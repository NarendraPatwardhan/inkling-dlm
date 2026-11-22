"""Everything that changes how the upstream Inkling code behaves. No upstream file is edited; the patches are
applied at runtime by `apply_patches()` and are idempotent.

Line numbers refer to transformers 5.24.0 (`models/inkling2_loop/modeling_inkling2_loop.py` unless named).
"""
from __future__ import annotations

import torch
import torch.nn as nn
from transformers.cache_utils import Cache

import transformers.models.inkling2_loop.modeling_inkling2_loop as M

from .constants import S


# ---------------------------------------------------------------------------------------------------------------
# Read-only view of the tower's states
# ---------------------------------------------------------------------------------------------------------------
class PrefixView(Cache):
    """Per-canvas, read-only view of a filled tower cache.

    `update()` / `update_conv_state()` return `prefix (+) new` and store nothing, so a decoder layer becomes a pure
    function of its inputs and gradient checkpointing may replay it (`modeling_layers.py:104-110` would otherwise
    replace the cache by None and the denoiser would silently lose the prompt).

    keys/values[i]: (B_ex, kv_heads, T_i, head_dim) for cache layer i; rows: (Bc,) example of each canvas;
    seeds[i][s]: (Bc, C, k-1) conv inputs just before the anchor for conv state s
    (0 k_sconv, 1 v_sconv, 2 post-attention conv, 3 post-MLP conv).
    """

    def __init__(self, keys, values, seeds, rows, rel, masks, position_ids, n_layers: int):
        # Cache.__init__ is deliberately not called: this object owns no layers and never allocates.
        self.k, self.v, self.seeds, self.rows = keys, values, seeds, rows
        self.rel, self.masks, self.position_ids = rel, masks, position_ids
        self.reads = 0
        # len(layers) != num_hidden_layers, so Inkling2LoopTextModel.forward does not append loop copies (1355-1357)
        self.layers = [None] * n_layers

    def update(self, key_states, value_states, layer_idx, *args, **kwargs):
        self.reads += 1
        k = self.k[layer_idx].index_select(0, self.rows).to(key_states.dtype)
        v = self.v[layer_idx].index_select(0, self.rows).to(value_states.dtype)
        return torch.cat([k, key_states], dim=-2), torch.cat([v, value_states], dim=-2)

    def update_conv_state(self, conv_states, layer_idx, state_idx=0, **kwargs):
        return torch.cat([self.seeds[layer_idx][state_idx].to(conv_states.dtype), conv_states], dim=-1)

    def get_conv_state(self, layer_idx, state_idx=0):
        return None            # forces the concatenate-then-convolve branch (373-388)

    def is_recording_past(self, layer_idx):
        return False

    def get_seq_length(self, layer_idx: int = 0) -> int:   # only used when position_ids is None; we always pass it
        raise RuntimeError("PrefixView requires explicit position_ids")


def _geometry(kv_valid, kv_pos, anchors, window, causal, dtype):
    """Additive mask (Bc,1,Q,K) and relative distance (Bc,1,Q,K) for one layer type.

    Keys are [tower keys at absolute positions kv_pos] + [S+1 canvas slots]. The canvas is bidirectional, except that
    the anchor's own canvas key is hidden: position n is already exported by the tower (context-tower Prop. 1)."""
    Bc, T = kv_valid.shape
    dev, Q = kv_valid.device, S + 1
    qpos = anchors[:, None] + torch.arange(Q, device=dev)                       # slot j sits at position n+j
    kpos = torch.cat([kv_pos[None].expand(Bc, T), qpos], dim=1)
    rel = qpos[:, :, None] - kpos[:, None, :]                                    # (Bc,Q,T+Q): q - k as upstream
    allow_canvas = torch.ones(Q, Q, dtype=torch.bool, device=dev)
    allow_canvas[:, 0] = False
    if causal:
        allow_canvas &= torch.ones(Q, Q, dtype=torch.bool, device=dev).tril()
    allow = torch.cat([kv_valid[:, None, :].expand(Bc, Q, T), allow_canvas.expand(Bc, Q, Q)], dim=-1)
    if window is not None:
        allow &= rel < window                                                    # sliding layers only
    mask = torch.zeros(Bc, 1, Q, T + Q, dtype=dtype, device=dev)
    mask = mask.masked_fill(~allow[:, None], torch.finfo(dtype).min)
    return mask, rel[:, None]


def build_view(keys, values, seeds, rows, kv_valid, anchors, sliding_len, *, window, n_layers, dtype,
               causal=False) -> PrefixView:
    """kv_valid: (Bc, T) tower columns this canvas may read (real tokens at or before its anchor).
    sliding_len: number of tower keys the sliding layers hold (T in training; <= window-1 at inference)."""
    T = kv_valid.shape[1]
    dev = kv_valid.device
    full_pos = torch.arange(T, device=dev)
    slide_pos = torch.arange(T - sliding_len, T, device=dev)
    m_full, r_full = _geometry(kv_valid, full_pos, anchors, None, causal, dtype)
    m_slide, r_slide = _geometry(kv_valid[:, T - sliding_len:], slide_pos, anchors, window, causal, dtype)
    masks = {"full_attention": m_full, "sliding_attention": m_slide, "linear_attention": None}
    rel = {"full": r_full, "sliding": r_slide}
    position_ids = anchors[:, None] + torch.arange(S + 1, device=dev)
    return PrefixView(keys, values, seeds, rows, rel, masks, position_ids, n_layers)


# ---------------------------------------------------------------------------------------------------------------
# Patches
# ---------------------------------------------------------------------------------------------------------------
def _rpb_forward(self, hidden_states, relative_distance, position_ids):
    # upstream body (260-270) plus a learned bank for forward distances, used only on canvas (4-D distance) calls
    rs = self.linear(hidden_states).view(*hidden_states.shape[:-1], self.num_heads, -1)
    rs = self.norm(rs, position_ids)
    back = (rs @ self.proj).transpose(1, 2)                                          # (B,H,Q,R)
    rd = relative_distance
    bias = back.gather(-1, rd.clamp(0, self.rel_extent - 1).expand(*back.shape[:2], -1, -1))
    bias = bias.masked_fill((rd < 0) | (rd >= self.rel_extent), 0.0)
    if rd.dim() == 4 and getattr(self, "proj_fwd", None) is not None:
        fwd = (rs @ self.proj_fwd.to(rs.dtype)).transpose(1, 2)
        fd = -rd                                                                     # forward distance
        fb = fwd.gather(-1, fd.clamp(0, self.rel_extent - 1).expand(*fwd.shape[:2], -1, -1))
        bias = bias + fb.masked_fill((fd <= 0) | (fd >= self.rel_extent), 0.0)
    return bias


_ORIG = {}


def _get_relative_distance(inputs_embeds, cache, is_sliding):
    if isinstance(cache, PrefixView):
        return cache.rel["sliding" if is_sliding else "full"]
    return _ORIG["get_relative_distance"](inputs_embeds, cache, is_sliding)


LAYER_CLASSES = (M.Inkling2LoopSlidingDecoderLayer, M.Inkling2LoopFullDecoderLayer,
                 M.Inkling2LoopSlidingLoopDecoderLayer, M.Inkling2LoopFullLoopDecoderLayer)


def apply_patches() -> None:
    if _ORIG:
        return
    _ORIG["get_relative_distance"] = M.get_relative_distance
    _ORIG["rpb_forward"] = M.Inkling2LoopRelativePositionBias.forward
    M.Inkling2LoopRelativePositionBias.forward = _rpb_forward
    M.get_relative_distance = _get_relative_distance          # looked up as a module global (1345-1348)
    for cls in LAYER_CLASSES:
        cls._can_checkpoint_with_cache = True                  # safe ONLY because PrefixView never mutates


def add_forward_bias_bank(model) -> list[nn.Parameter]:
    """Attach `proj_fwd` (FP32 master, mirror-initialised from `proj`) to every relative-position-bias module."""
    new = []
    for mod in model.modules():
        if isinstance(mod, M.Inkling2LoopRelativePositionBias) and getattr(mod, "proj_fwd", None) is None:
            mod.proj_fwd = nn.Parameter(mod.proj.detach().float().clone())
            new.append(mod.proj_fwd)
    return new
