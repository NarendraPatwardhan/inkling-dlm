"""BYOD masked-diffusion objective per block: 0.9 answer / 0.1 EOS padding, 1/t weight, per-region normalised."""
from __future__ import annotations

import torch

W_ANSWER, W_PAD = 0.9, 0.1


def block_loss(ce_sel, t_sel, region_sel, blk_sel, n_ans, n_pad):
    """ce_sel: per-slot cross entropy over masked slots (flat; tower.canvas_ce); blk_sel the block of each; n_ans/n_pad
    (Bc,) eligible slots per block. Returns (loss, answer term mean, padding term mean), the last two detached."""
    ce = ce_sel / t_sel
    Bc = n_ans.shape[0]
    la = torch.zeros(Bc, device=ce.device).index_add_(0, blk_sel, ce * (region_sel == 0))
    lp = torch.zeros(Bc, device=ce.device).index_add_(0, blk_sel, ce * (region_sel == 1))
    la = torch.where(n_ans > 0, la / n_ans.clamp(min=1), torch.zeros_like(la))
    lp = torch.where(n_pad > 0, lp / n_pad.clamp(min=1), torch.zeros_like(lp))
    loss = (W_ANSWER * la + W_PAD * lp).mean()
    return loss, la.detach().mean(), lp.detach().mean()
