"""LoopCD (arXiv 2610.02185): contrastive decoding between the looped core's first pass and its final pass.

Inkling loops layers [loop_start, loop_end) (6..17 in the release, 2..3 in the tiny model) `loop_num_passes` (3)
times. The state after pass k is h_k = (hidden_states, block_states) at the end of the core (paper Eq. 1); the
coda (post-loop layers, attention residual, final norm, lm_head, final logit softcap) maps a state to logits, the
paper's "output interface" (App. A.1). The reference is h_1 (deterministic prelude, App. B.4).

  hidden:  h' = h_R + w (h_R - h_1), one coda pass on h'                                      (Eq. 2)
  logits:  z_1 = coda(h_1), z_R = coda(h_R), z' = z_R + w (z_R - z_1), on softcapped logits    (Eq. 3)
  adaptive w (logits only, Sec. 3.2): w = w_max [1 - (p_(1) - p_(2))], p = softmax(z_R)        (Eq. 4)

There is no plausibility mask or truncation in the paper; temperature (if any) applies after guidance (App. A.3).
`passes` < loop_num_passes is the paper's reduced-depth setting (Sec. 4.3): the canvas runs fewer core passes,
reference still h_1; the tower prefix keeps full depth.
"""
from __future__ import annotations

import contextlib
from dataclasses import asdict, dataclass

import torch

import transformers.models.inkling2_loop.modeling_inkling2_loop as M

from .constants import S
from .tower import canvas_logits, parts

MODES = ("off", "logits", "hidden")


@dataclass(frozen=True)
class Spec:
    mode: str = "off"              # off | logits | hidden
    w: float = 0.0                 # fixed strength, or w_max when adaptive
    adaptive: bool = False         # Eq. 4 (logits only, as in the paper)
    passes: int | None = None      # core passes on the canvas (None: the model's own, 3)

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError(f"loopcd mode {self.mode!r} not in {MODES}")
        if self.w < 0:
            raise ValueError("loopcd w must be >= 0 (Eq. 2/3)")
        if self.adaptive and self.mode != "logits":
            raise ValueError("adaptive strength needs p_R, i.e. the logits form (paper Sec. 3.2)")
        if self.passes is not None and self.passes < (2 if self.mode != "off" else 1):
            raise ValueError("loopcd needs at least 2 core passes (h_1 and a later h_R)")

    @property
    def name(self) -> str:
        if self.mode == "off":
            core = "off"
        else:
            core = f"{self.mode}-{'ada' if self.adaptive else 'w'}{self.w:g}"
        return core + (f"-p{self.passes}" if self.passes is not None else "")

    def plain(self, n_passes: int) -> bool:
        """True when this spec is exactly the current sampler path (upstream forward)."""
        return self.mode == "off" and self.passes in (None, n_passes)

    def to_dict(self) -> dict:
        return asdict(self)


def spec_of(cfg) -> Spec:
    return Spec(cfg.loopcd, cfg.loopcd_w, cfg.loopcd_adaptive, cfg.loopcd_passes)


def sweep_of(cfg) -> list[Spec]:
    return [Spec(**d) for d in cfg.loopcd_sweep]


def _layer_kw(view, rel, layer):
    return dict(attention_mask=view.masks[layer.attention_type], relative_distance=rel[layer.attention_type],
                conv_mask=view.masks["linear_attention"], cache=view, position_ids=view.position_ids)


def _run(layers, hidden, blocks, view, rel, **extra):
    for layer in layers:
        new = layer(hidden, blocks, **_layer_kw(view, rel, layer), **extra)
        if layer.is_block_boundary:
            blocks = torch.cat([blocks.to(hidden.device), hidden.unsqueeze(-2)], dim=-2)
        hidden = new
    return hidden, blocks


def core_states(text, view, canvas_ids, passes=None):
    """Inkling2LoopTextModel.forward (transformers 5.24, 1308-1450) on a PrefixView canvas, stopping at the end of
    the looped core. Returns (h_1, h_R, ctx): h = (hidden_states, block_states), ctx = (rel, embedding dtype)."""
    c = text.config
    start, end = c.loop_layer_index_start, c.loop_layer_index_end
    n_loop, passes = end - start, passes or c.loop_num_passes
    emb = text.embed_tokens(canvas_ids)
    rel = {"full_attention": M.get_relative_distance(emb, view, is_sliding=False),
           "sliding_attention": M.get_relative_distance(emb, view, is_sliding=True)}
    hidden = emb
    blocks = emb.new_zeros(*emb.shape[:-1], 0, emb.shape[-1])
    hidden, blocks = _run(text.layers[:start], hidden, blocks, view, rel)
    pre_h, pre_b = hidden, blocks
    n_pre = pre_b.shape[-2]
    hidden, blocks = _run(text.layers[start:end], hidden, blocks, view, rel)
    first = (hidden, blocks)
    for li in range(1, passes):
        prev_b, prev_h = blocks[..., n_pre + 1:, :], hidden
        hidden, blocks = pre_h, pre_b
        off = c.num_hidden_layers + (li - 1) * n_loop - start
        for j, layer in enumerate(text.layers[start:end]):
            msg = text.message_input(prev_b, prev_h, j)
            new = layer(hidden, blocks, msg, **_layer_kw(view, rel, layer), cache_offset=off)
            if layer.is_block_boundary:
                blocks = torch.cat([blocks.to(hidden.device), hidden.unsqueeze(-2)], dim=-2)
            hidden = new
    return first, (hidden, blocks), (rel, emb.dtype)


def coda_logits(lm, text, view, state, ctx):
    """Post-loop layers -> attention residual -> norm -> lm_head -> float -> softcap, as upstream (1430-1450 and
    Inkling2LoopForCausalLM.forward 1531-1535). Slot rows only: (B, S, V)."""
    c = text.config
    rel, dtype = ctx
    hidden, blocks = _run(text.layers[c.loop_layer_index_end:], *state, view, rel)
    hidden = text.final_input(blocks, hidden)[0]
    hidden = text.norm(hidden).to(dtype)[:, :S]
    return lm.logit_softcap(lm.lm_head(hidden).float())


def adaptive_w(z_final, w_max: float):
    """Eq. 4 per token: w_max * (1 - (p_(1) - p_(2))) with p = softmax(z_R). Shape (..., 1)."""
    top = torch.softmax(z_final.float(), -1).topk(2, dim=-1).values
    return w_max * (1.0 - (top[..., :1] - top[..., 1:2]))


def guide(a_final, a_ref, w):
    """Eq. 2 / Eq. 3: x' = x_R + w (x_R - x_1)."""
    return a_final + w * (a_final - a_ref)


def guided_logits(peft_model, view, canvas_ids, spec: Spec | None = None, adapters=True):
    """(B, S, V) softcapped logits for every block slot under `spec`. Off at full depth is `canvas_logits`."""
    lm, text = parts(peft_model)
    spec = spec or Spec()
    if spec.plain(text.config.loop_num_passes):
        return canvas_logits(peft_model, view, canvas_ids, adapters=adapters)
    with contextlib.nullcontext() if adapters else peft_model.disable_adapter():
        h1, hr, ctx = core_states(text, view, canvas_ids, spec.passes)
        if spec.mode == "off":
            return coda_logits(lm, text, view, hr, ctx)
        if spec.mode == "hidden":
            h = (guide(hr[0], h1[0], spec.w), guide(hr[1], h1[1], spec.w))
            return coda_logits(lm, text, view, h, ctx)
        zr = coda_logits(lm, text, view, hr, ctx)
        z1 = coda_logits(lm, text, view, h1, ctx)
        w = adaptive_w(zr, spec.w) if spec.adaptive else spec.w
        return guide(zr, z1, w)
