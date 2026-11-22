"""Frozen tower (the parent model with adapters disabled) and the denoiser call on a 33-slot canvas."""
from __future__ import annotations

import contextlib

import torch
from transformers.cache_utils import DynamicCache

from .constants import S
from .patches import PrefixView, build_view


def parts(peft_model):
    lm = peft_model.base_model.model           # Inkling2LoopForCausalLM
    return lm, lm.model                         # (head owner, Inkling2LoopTextModel)


def geometry(peft_model) -> dict:
    cfg = parts(peft_model)[1].config
    n_layers = cfg.num_hidden_layers + (cfg.loop_num_passes - 1) * (cfg.loop_layer_index_end - cfg.loop_layer_index_start)
    return dict(n_layers=n_layers, window=cfg.sliding_window_size, kernel=cfg.conv_kernel_size)


@torch.no_grad()
def tower_fill(peft_model, input_ids, attention_mask=None, cache=None, record_past=False, position_ids=None):
    """One causal pass of the frozen parent. record_past keeps the full KV and conv-input history for every cache
    layer (cache_utils DynamicSlidingWindowLayer/LinearAttentionLayer with record_past)."""
    lm, text = parts(peft_model)
    if cache is None:
        cache = DynamicCache(config=text.config)
        if record_past:
            cache.activate_past_recording()
    with peft_model.disable_adapter():
        text(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, cache=cache,
             use_cache=True)
    return cache


def compact_training_states(cache, rows, anchors, kernel: int):
    """After a record_past tower pass over whole examples: KV stays per example; conv inputs are cut down to the
    kernel-1 positions before each canvas anchor (zeros before position 0, as upstream pads), then the history is
    dropped to free memory."""
    keys, values, seeds = [], [], []
    k1 = kernel - 1
    idx = anchors[:, None] - k1 + torch.arange(k1, device=anchors.device)           # (Bc,k-1)
    valid = (idx >= 0)[:, None, :]
    idx = idx.clamp(min=0)
    for layer in cache.layers:
        keys.append(layer.keys)
        values.append(layer.values)
        per_state = []
        for si in range(layer.number_of_states):                                     # conv_states is a dict
            h = layer.conv_states[si]                                                # (B_ex, C, T)
            g = h.permute(0, 2, 1)[rows[:, None], idx].transpose(1, 2)                # (Bc, C, k-1)
            per_state.append(torch.where(valid, g, torch.zeros_like(g)).contiguous())
        seeds.append(per_state)
        for si in range(layer.number_of_states):
            layer.conv_states[si] = None                                             # free the history now
    return keys, values, seeds


def slice_seeds(seeds, idx):
    return [[s[idx] for s in per_layer] for per_layer in seeds]


def view_from_inference_cache(peft_model, cache, n_tokens, causal=False):
    """Batch-1 sampler: the tower cache holds positions 0..n_tokens-1; the anchor is the last one."""
    g = geometry(peft_model)
    dev = cache.layers[0].keys.device
    keys = [l.keys for l in cache.layers]
    values = [l.values for l in cache.layers]
    k = g["kernel"]
    seeds = [[l.conv_states[si][..., -k:-1] for si in range(l.number_of_states)]      # last k inputs minus anchor
             for l in cache.layers]
    sliding_len = min(l.keys.shape[-2] for l in cache.layers)
    rows = torch.zeros(1, dtype=torch.long, device=dev)
    kv_valid = torch.ones(1, n_tokens, dtype=torch.bool, device=dev)
    anchors = torch.tensor([n_tokens - 1], device=dev)
    return build_view(keys, values, seeds, rows, kv_valid, anchors, sliding_len, window=g["window"],
                      n_layers=g["n_layers"], dtype=keys[0].dtype, causal=causal)


def view_for_batch(peft_model, keys, values, seeds, b, idx, causal=False) -> PrefixView:
    g = geometry(peft_model)
    return build_view(keys, values, slice_seeds(seeds, idx), b["rows"][idx], b["kv_valid"][idx], b["anchors"][idx],
                      sliding_len=b["ids"].shape[1], window=g["window"], n_layers=g["n_layers"],
                      dtype=keys[0].dtype, causal=causal)


def canvas_logits(peft_model, view, canvas_ids, select=None, adapters=True):
    """canvas_ids: (Bc, S+1) = [anchor, S block slots]. Slot j's output predicts block position j (token n+j+1).
    Returns softcapped float logits for the selected (Bc, S) slots, or all S if select is None."""
    lm, text = parts(peft_model)
    ctx = contextlib.nullcontext() if adapters else peft_model.disable_adapter()
    with ctx:
        h = text(input_ids=canvas_ids, attention_mask=view.masks, position_ids=view.position_ids,
                 cache=view, use_cache=False).last_hidden_state[:, :S]
    if select is not None:
        h = h[select]
    return lm.logit_softcap(lm.lm_head(h).float())     # same order as Inkling2LoopForCausalLM.forward (1531-1535)
