"""Held-out masked-token loss on fixed validation blocks and fixed masks (used during training and in sanity)."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .constants import MASK_ID
from .data import collate
from .tower import canvas_logits, compact_training_states, geometry, tower_fill, view_for_batch

T_GRID = (0.1, 0.3, 0.5, 0.7, 0.9)


def tower_states(pm, b):
    cache = tower_fill(pm, b["ids"], b["attn"], record_past=True)
    return compact_training_states(cache, b["rows"], b["anchors"], geometry(pm)["kernel"])


@torch.no_grad()
def heldout_ce(pm, groups, cfg, adapters=True, micro=None) -> dict:
    """Mean masked-token CE (no 1/t) over fixed groups, at fixed mask rates. Returns overall and per-t values."""
    was_training = pm.training
    pm.eval()
    dev = next(pm.parameters()).device
    acc = {t: [0.0, 0] for t in T_GRID}                    # summed CE, masked tokens
    micro = micro or cfg.micro_blocks
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
                view = view_for_batch(pm, k, v, s, b, idx)
                lg = canvas_logits(pm, view, canvas[idx], select=m[idx], adapters=adapters)
                acc[tt][0] += F.cross_entropy(lg, b["tgt"][idx][m[idx]], reduction="sum").item()
                acc[tt][1] += int(m[idx].sum())
    pm.train(was_training)
    out = {f"ce_t{tt:.1f}": c / max(1, n) for tt, (c, n) in acc.items()}
    out["ce"] = sum(c for c, _ in acc.values()) / max(1, sum(n for _, n in acc.values()))
    return out
