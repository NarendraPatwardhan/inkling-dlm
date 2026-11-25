"""Held-out masked-token loss on fixed validation blocks and fixed masks (used during training and in sanity)."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .constants import MASK_ID
from .data import collate
from .loopcd import guided_logits
from .tower import canvas_ce, compact_training_states, geometry, tower_fill, view_for_batch

T_GRID = (0.1, 0.3, 0.5, 0.7, 0.9)


def tower_states(pm, b):
    cache = tower_fill(pm, b["ids"], b["attn"], record_past=True)
    return compact_training_states(cache, b["rows"], b["anchors"], geometry(pm)["kernel"])


def masked_canvases(pm, groups, cfg, micro):
    """Fixed held-out masks: yields (t, view, canvas, mask, targets at the masked slots) per micro-batch."""
    dev = next(pm.parameters()).device
    for i, ex in enumerate(groups):
        b = collate(ex, 10_000_000 + i, cfg.seed, dev, cfg.block)
        k, v, s = tower_states(pm, b)
        for tt in T_GRID:
            g = torch.Generator().manual_seed(int(tt * 100) + 1000 * i)
            m = (torch.rand(b["masked"].shape, generator=g) < tt).to(dev)
            m[~m.any(1), 0] = True
            canvas = torch.cat([b["canvas"][:, :1], torch.where(m, MASK_ID, b["tgt"])], 1)
            Bc = canvas.shape[0]
            for c0 in range(0, Bc, micro):
                idx = torch.arange(c0, min(Bc, c0 + micro), device=dev)
                yield tt, view_for_batch(pm, k, v, s, b, idx), canvas[idx], m[idx], b["tgt"][idx][m[idx]]


def _summary(acc) -> dict:
    out = {f"ce_t{tt:.1f}": c / max(1, n) for tt, (c, n) in acc.items()}
    out["ce"] = sum(c for c, _ in acc.values()) / max(1, sum(n for _, n in acc.values()))
    return out


@torch.no_grad()
def heldout_ce(pm, groups, cfg, adapters=True, micro=None) -> dict:
    """Mean masked-token CE (no 1/t) over fixed groups, at fixed mask rates. Returns overall and per-t values."""
    was_training = pm.training
    pm.eval()
    acc = {t: [0.0, 0] for t in T_GRID}                    # summed CE, masked tokens
    for tt, view, canvas, m, tgt in masked_canvases(pm, groups, cfg, micro or cfg.micro_blocks):
        ce = canvas_ce(pm, view, canvas, m, tgt, adapters=adapters)
        acc[tt][0] += ce.sum().item()
        acc[tt][1] += int(m.sum())
    pm.train(was_training)
    return _summary(acc)


@torch.no_grad()
def heldout_guided(pm, groups, cfg, spec, adapters=True, micro=None) -> dict:
    """Same masks as heldout_ce, scored with LoopCD's guided distribution softmax(z'): masked CE and top-1 accuracy
    (the token the greedy sampler would commit)."""
    was_training = pm.training
    pm.eval()
    acc = {t: [0.0, 0] for t in T_GRID}
    hits = 0
    for tt, view, canvas, m, tgt in masked_canvases(pm, groups, cfg, micro or cfg.loopcd_micro):
        z = guided_logits(pm, view, canvas, spec, adapters=adapters)[m]
        acc[tt][0] += F.cross_entropy(z, tgt, reduction="sum").item()
        acc[tt][1] += int(m.sum())
        hits += int((z.argmax(-1) == tgt).sum())
    pm.train(was_training)
    return _summary(acc) | {"top1": hits / max(1, sum(n for _, n in acc.values()))}
