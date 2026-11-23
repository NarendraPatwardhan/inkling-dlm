"""4-bit experts on an A100: the released NVFP4 / MXFP8 expert bytes stay packed and are decoded on the fly by
Unsloth's packed-expert kernels (`unsloth_zoo.packed_experts`).

Why this path (docs/RUNBOOK.md, "4-bit route"):
* The checkpoint already is 4-bit: routed experts are NVFP4 (layers 0-19) and MXFP8 (20-23), 15.67 GB; everything
  else is BF16/FP32, 3.22 GB. transformers 5.24 can only load it by dequantising the experts to BF16 (~52 GB).
* Re-quantising to bitsandbytes NF4 (Unsloth's stock Inkling loader with load_in_4bit) would add a second
  quantisation error to every tower state, so the frozen tower would no longer be Inkling.
* Unsloth's `Nvfp4ExpertParam` / `Mxfp8ExpertParam` hold the exact checkpoint bytes; `packed_expert_grouped_linear`
  decodes only the experts that received tokens (Triton on CUDA, sm80 included: E4M3 is decoded by hand), runs the same
  `grouped_mm` as transformers on the decoded rows and, in backward, decodes again for dX instead of keeping BF16
  weights. In BF16 the decoded values equal transformers' own `Inkling2LoopDequantizeExperts` on the checkpoint
  tensors in their stored dtypes (bitwise on real checkpoint slices on the CPU and on whole layers in G-dispatch),
  while holding 15.7 GB instead of 48.9 GB of expert weights. (transformers' BF16 from_pretrained additionally rounds
  the FP32 per-expert amax to BF16 before decoding; see `as_from_pretrained`.)

What Unsloth does not do for Inkling, and this module adds: reading Inkling's own expert layout (`w13_weight` +
`_scale` / `_amax` / `_outer_scale_denominator`, gate/up rows interleaved) into those packed stacks, loading the rest
of the model without ever materialising BF16 experts, and an experts-forward registered with transformers.
"""
from __future__ import annotations

import contextlib
import os

import torch

# unsloth_zoo refuses to import unless the `unsloth` package announced itself. `import unsloth` would also patch
# transformers' trainer, attention and generation globally, which this custom training loop does not use and which
# G-exact would have to re-prove; only the zoo's packed-expert kernels are used here.
os.environ.setdefault("UNSLOTH_IS_PRESENT", "1")
from unsloth_zoo.packed_experts import (  # noqa: E402
    Mxfp8ExpertParam,
    Nvfp4ExpertParam,
    is_new_packed_expert_param,
    packed_expert_grouped_linear,
)

IMPL = "unsloth_packed"
E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
NVFP4_DEN = 6.0 * 448.0          # ModelOpt's outer scale denominator (E2M1 max x E4M3 max)
STACKS = (("w13", "gate_up_proj", True), ("w2", "down_proj", False))   # checkpoint prefix, parameter, interleaved rows


# ------------------------------------------------------------------------------------------------ experts forward
def packed_experts_forward(self, hidden_states, top_k_index, top_k_weights):
    """transformers' `grouped_mm_experts_forward` (integrations/moe.py) with each `_grouped_linear` replaced by
    Unsloth's packed grouped linear: same sort, offsets, gate, weighting and fp32 reduction, so on identical decoded
    weights the result is bitwise identical."""
    num_top_k = top_k_index.size(-1)
    num_tokens, hidden_dim = hidden_states.shape
    sample_weights = top_k_weights.reshape(-1)
    expert_ids = top_k_index.reshape(-1)
    expert_ids_g, perm = torch.sort(expert_ids)
    x = hidden_states[perm // num_top_k]
    w = sample_weights[perm]
    counts = torch.bincount(expert_ids_g, minlength=self.num_experts)[: self.num_experts].to(torch.int32)
    h = packed_expert_grouped_linear(x, self.gate_up_proj, counts, True)
    h = self._apply_gate(h)
    h = packed_expert_grouped_linear(h, self.down_proj, counts, True)
    out = h * w.unsqueeze(-1)
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(perm.size(0), device=perm.device)
    out = out[inv]
    return out.view(num_tokens, num_top_k, hidden_dim).sum(dim=1).to(hidden_states.dtype)


def register() -> None:
    from transformers.integrations.moe import ALL_EXPERTS_FUNCTIONS
    if IMPL not in ALL_EXPERTS_FUNCTIONS.valid_keys():
        ALL_EXPERTS_FUNCTIONS.register(IMPL, packed_experts_forward)


# ------------------------------------------------------------------------------------------------ checkpoint layout
def deinterleave_rows(t: torch.Tensor) -> torch.Tensor:
    """transformers' Interleave(dim=1) on (E, 2I, C): row g*I + i <- checkpoint row 2i + g (gate rows, then up rows).
    Codes and block scales are per row, so the same permutation applies to the packed bytes."""
    E, R = t.shape[:2]
    return t.reshape(E, R // 2, 2, *t.shape[2:]).transpose(1, 2).reshape(t.shape).contiguous()


def is_nvfp4(raw: dict, prefix: str) -> bool:
    return f"{prefix}_weight_amax" in raw


def pack_stack(raw: dict, prefix: str, interleaved: bool, dtype=torch.bfloat16):
    """raw: checkpoint tensors of one experts module keyed by local name (`w13_weight`, `w13_weight_scale`, ...)."""
    codes, scale = raw[f"{prefix}_weight"], raw[f"{prefix}_weight_scale"]
    codes = codes.view(torch.uint8) if codes.dtype != torch.uint8 else codes
    scale = scale.view(torch.uint8)
    if interleaved:
        codes, scale = deinterleave_rows(codes), deinterleave_rows(scale)
    if is_nvfp4(raw, prefix):
        # transformers: (e2m1 * block_scale) * (amax / denominator) in fp32, then one cast; Unsloth "multiply" mode
        # forms e2m1 * (block_scale * global). Equal after the BF16 cast on every checked slice; in FP32 <= 1 ulp.
        glob = (raw[f"{prefix}_weight_amax"].float() / raw[f"{prefix}_weight_outer_scale_denominator"].float())
        return Nvfp4ExpertParam(codes.contiguous(), packed_scales=scale.contiguous(),
                                packed_global=glob.reshape(-1).contiguous(), packed_global_mode="multiply",
                                packed_dtype=dtype)
    return Mxfp8ExpertParam(codes.contiguous(), packed_scales=scale.contiguous(), packed_dtype=dtype)


def reference_dequant(raw: dict, prefix: str, interleaved: bool, dtype=torch.bfloat16) -> torch.Tensor:
    """transformers 5.24's own decode of the same tensors (the BF16-loaded parent's expert weights)."""
    from transformers.core_model_loading import Inkling2LoopDequantizeExperts, Interleave
    names = [f"{prefix}_weight", f"{prefix}_weight_scale", f"{prefix}_weight_amax",
             f"{prefix}_weight_outer_scale_denominator"]
    src = {n: raw[n] for n in names if n in raw}
    if not is_nvfp4(raw, prefix):
        src[names[0]] = raw[names[0]].view(torch.float8_e4m3fn)

    class _M:
        def get_parameter(self, _):
            return torch.empty(0, dtype=dtype)
    out = Inkling2LoopDequantizeExperts().convert(src, names, ["t"], model=_M(), full_layer_name="t")["t"]
    if interleaved:
        out = Interleave(dim=1).convert({"t": out}, ["t"], ["t"])["t"]
    return out


def as_from_pretrained(raw: dict, prefix: str, dtype) -> dict:
    """The sources as transformers 5.24's loader hands them to the converter: cast to the parameter dtype first
    (core_model_loading.py:1918-1925). For BF16 that rounds the FP32 per-expert amax, scaling each expert's weights
    by up to ~0.3% (measured); codes and FP8 scales survive the cast exactly."""
    out = dict(raw)
    for s in ("_amax", "_outer_scale_denominator"):
        k = f"{prefix}_weight{s}"
        if k in out:
            out[k] = out[k].to(dtype)
    return out


def install(experts, raw: dict, dtype=torch.bfloat16) -> int:
    """Put packed stacks into one Inkling2LoopExperts module; returns packed bytes."""
    n = 0
    for prefix, name, inter in STACKS:
        p = pack_stack(raw, prefix, inter, dtype)
        experts._parameters[name] = p
        n += p.nbytes_packed()
    return n


def experts_modules(model):
    from transformers.models.inkling2_loop.modeling_inkling2_loop import Inkling2LoopExperts
    return [(n, m) for n, m in model.named_modules() if isinstance(m, Inkling2LoopExperts)]


def packed_bytes(model) -> dict:
    out = {}
    for p in model.parameters():
        if is_new_packed_expert_param(p):
            out[p.packed_format] = out.get(p.packed_format, 0) + p.nbytes_packed()
    return out


# ------------------------------------------------------------------------------------------------ loading
def checkpoint_key(layer: int, local: str) -> str:
    return f"model.llm.layers.{layer}.mlp.experts.{local}"


def raw_layer(source, layer: int, device) -> dict:
    """Raw checkpoint tensors of one experts module. source: a safetensors path, or a dict of already-raw tensors
    keyed by checkpoint name (tiny models)."""
    names = [f"{p}_weight{s}" for p, _, _ in STACKS for s in ("", "_scale", "_amax", "_outer_scale_denominator")]
    if isinstance(source, dict):
        return {n: source[checkpoint_key(layer, n)].to(device) for n in names if checkpoint_key(layer, n) in source}
    from safetensors import safe_open
    out = {}
    with safe_open(source, framework="pt", device=str(device)) as f:
        keys = set(f.keys())
        for n in names:
            if checkpoint_key(layer, n) in keys:
                out[n] = f.get_tensor(checkpoint_key(layer, n))
    return out


@contextlib.contextmanager
def experts_left_out():
    """While loading: expert stacks are created empty and transformers' dequantising expert converters are removed,
    so no BF16 expert tensor is ever allocated; the raw expert keys are reported as unexpected and filled afterwards."""
    from transformers import conversion_mapping as cm
    from transformers.models.inkling2_loop import modeling_inkling2_loop as M

    cls = M.Inkling2LoopExperts
    init = cls.__init__

    def lean_init(self, config, *a, **k):
        init(self, config, *a, **k)
        self.gate_up_proj = torch.nn.Parameter(torch.empty(0), requires_grad=False)
        self.down_proj = torch.nn.Parameter(torch.empty(0), requires_grad=False)

    key = "inkling2_loop_text"
    before = cm.get_checkpoint_conversion_mapping(key)
    kept = [t for t in before
            if not any(str(p).startswith(r"mlp\.experts\.w") for p in getattr(t, "_original_source_patterns", ()))]
    cls.__init__ = lean_init
    cm.register_checkpoint_conversion_mapping(key, kept, overwrite=True)
    try:
        yield len(before) - len(kept)
    finally:
        cls.__init__ = init
        cm.register_checkpoint_conversion_mapping(key, before, overwrite=True)


def checkpoint_file(repo: str, revision: str | None) -> str:
    if os.path.isdir(repo):
        return os.path.join(repo, "model-00001-of-00001.safetensors") if os.path.exists(
            os.path.join(repo, "model-00001-of-00001.safetensors")) else os.path.join(repo, "model.safetensors")
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo, "model-00001-of-00001.safetensors", revision=revision)


def placement(device) -> tuple[torch.device, dict]:
    """(device, transformers device_map). Plain "cuda" has index None, and device_map {"": None} breaks loading."""
    dev = torch.device(device)
    if dev.type == "cuda" and dev.index is None:
        dev = torch.device("cuda", torch.cuda.current_device() if torch.cuda.is_available() else 0)
    return dev, {"": dev.index if dev.type == "cuda" else str(dev)}


def load_packed(repo: str, revision: str | None, device, dtype=torch.bfloat16):
    """Inkling2LoopForCausalLM with non-expert weights loaded by transformers and every experts module filled with the
    checkpoint's own packed bytes. Returns (model, load_info) where load_info lists only genuine problems."""
    from transformers.models.inkling2_loop.modeling_inkling2_loop import Inkling2LoopForCausalLM
    register()
    dev, device_map = placement(device)
    with experts_left_out() as dropped:
        model, info = Inkling2LoopForCausalLM.from_pretrained(
            repo, revision=revision, dtype=dtype, device_map=device_map, attn_implementation="eager",
            output_loading_info=True)
    path = checkpoint_file(repo, revision)
    mods = experts_modules(model)
    filled, consumed, nbytes = set(), set(), 0
    for name, mod in mods:
        layer = int(name.split("layers.")[1].split(".")[0])
        raw = raw_layer(path, layer, dev)
        nbytes += install(mod, raw, dtype)
        filled |= {f"{name}.gate_up_proj", f"{name}.down_proj"}
        consumed |= {f"{name}.{k}" for k in raw}
    model.requires_grad_(False)
    model.config._experts_implementation = IMPL
    model.set_experts_implementation(IMPL)
    missing = [k for k in info.get("missing_keys", []) if k not in filled]
    unexpected = [k for k in info.get("unexpected_keys", []) if k not in consumed]
    info = dict(info, missing_keys=missing, unexpected_keys=unexpected, packed_bytes=nbytes,
                packed_modules=len(mods), converters_dropped=dropped, checkpoint=path)
    return model, info


# ------------------------------------------------------------------------------------------------ synthetic checkpoints
def _e2m1_codes(y: torch.Tensor) -> torch.Tensor:
    grid = torch.tensor(E2M1, dtype=torch.float32)
    idx = (y.abs()[..., None] - grid).abs().argmin(-1)
    return (idx | ((y < 0) & (idx > 0)).long() << 3).to(torch.uint8)


def modelopt_nvfp4(w: torch.Tensor) -> dict:
    """ModelOpt-style NVFP4 of (E, R, C): codes (E, R, C/2) uint8 (even column in the low nibble), E4M3 block scales
    per 16, per-expert amax and the shared denominator. For building tiny checkpoints in the checkpoint's layout."""
    E, R, C = w.shape
    w = w.float()
    amax = w.abs().amax(dim=(1, 2)).clamp(min=1e-8)
    factor = amax / NVFP4_DEN
    blocks = w.view(E, R, C // 16, 16)
    s = (blocks.abs().amax(-1) / 6.0 / factor[:, None, None]).clamp(max=448.0).to(torch.float8_e4m3fn)
    sf = s.float() * factor[:, None, None]
    y = torch.where(sf[..., None] > 0, blocks / sf[..., None].clamp(min=1e-30), torch.zeros_like(blocks))
    q = _e2m1_codes(y.clamp(-6, 6)).view(E, R, C)
    codes = (q[..., 0::2] | (q[..., 1::2] << 4)).contiguous()
    return dict(weight=codes, weight_scale=s, weight_amax=amax.float(),
                weight_outer_scale_denominator=torch.tensor([NVFP4_DEN]))


def mxfp8(w: torch.Tensor) -> dict:
    """OCP MXFP8 of (E, R, C): E4M3 values and one E8M0 exponent per 32 (uint8)."""
    E, R, C = w.shape
    blocks = w.float().view(E, R, C // 32, 32)
    bmax = blocks.abs().amax(-1)
    e = torch.where(bmax > 0, torch.floor(torch.log2(bmax.clamp(min=2.0 ** -126))) - 8, torch.full_like(bmax, -127))
    e = e.clamp(-127, 127)
    vals = (blocks / torch.exp2(e)[..., None]).clamp(-448, 448).to(torch.float8_e4m3fn).view(E, R, C)
    return dict(weight=vals, weight_scale=(e + 127).to(torch.uint8))


def interleave_rows(t: torch.Tensor) -> torch.Tensor:
    """Inverse of deinterleave_rows: the checkpoint's gate/up row order."""
    E, R = t.shape[:2]
    return t.reshape(E, 2, R // 2, *t.shape[2:]).transpose(1, 2).reshape(t.shape).contiguous()


def quantize_like_checkpoint(model, mxfp8_last: int) -> dict:
    """Raw checkpoint-style expert tensors for every experts module of a (tiny) dense model: NVFP4, except the last
    `mxfp8_last` decoder layers in MXFP8, as in the release. Keys are checkpoint names."""
    mods = experts_modules(model)
    layers = sorted({int(n.split("layers.")[1].split(".")[0]) for n, _ in mods})
    mx = set(layers[-mxfp8_last:]) if mxfp8_last else set()
    out = {}
    for name, mod in mods:
        layer = int(name.split("layers.")[1].split(".")[0])
        for prefix, pname, inter in STACKS:
            w = getattr(mod, pname).detach().float().cpu()
            if inter:
                w = interleave_rows(w)
            q = mxfp8(w) if layer in mx else modelopt_nvfp4(w)
            for k, v in q.items():
                out[checkpoint_key(layer, f"{prefix}_{k}")] = v
    return out
