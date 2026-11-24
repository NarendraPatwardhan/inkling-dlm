"""Block loss against a direct per-slot reference, Unsloth's chunked softcapped CE against the plain logits path,
and block collation rules."""
import math

import torch
import torch.nn.functional as F

from inkling_dlm.constants import EOS_ID, MASK_ID, S
from inkling_dlm.data import collate, n_blocks
from inkling_dlm.loss import block_loss


def _reference(logits, tgt, t, region, masked):
    """Naive: per block, sum CE/t over masked answer slots / #answer slots, same for padding; 0.9/0.1; mean."""
    out = []
    for i in range(tgt.shape[0]):
        n_a = int((region[i] == 0).sum())
        n_p = int((region[i] == 1).sum())
        la = lp = 0.0
        for j in range(S):
            if masked[i, j]:
                ce = F.cross_entropy(logits[i, j][None], tgt[i, j][None]).item() / t[i].item()
                if region[i, j] == 0:
                    la += ce
                else:
                    lp += ce
        out.append(0.9 * (la / n_a if n_a else 0) + 0.1 * (lp / n_p if n_p else 0))
    return sum(out) / len(out)


def check_block_loss_matches_reference():
    g = torch.Generator().manual_seed(0)
    Bc, V = 5, 50
    logits = torch.randn(Bc, S, V, generator=g)
    tgt = torch.randint(0, V, (Bc, S), generator=g)
    region = torch.zeros(Bc, S, dtype=torch.long)
    for i, L in enumerate([32, 20, 1, 31, 32]):
        region[i, L:] = 1
    t = torch.rand(Bc, generator=g) * 0.9 + 0.1
    masked = torch.rand(Bc, S, generator=g) < t[:, None]
    masked[:, 0] = True
    blk = masked.nonzero()[:, 0]
    ce = F.cross_entropy(logits[masked], tgt[masked], reduction="none")
    loss, la, lp = block_loss(ce, t[blk], region[masked], blk,
                              (region == 0).sum(1).float(), (region == 1).sum(1).float())
    assert math.isclose(loss.item(), _reference(logits, tgt, t, region, masked), rel_tol=1e-5)
    assert la.item() >= 0 and lp.item() >= 0


def check_collate_blocks_targets_and_anchors():
    p1, a1 = list(range(10, 30)), list(range(100, 170))          # ended: 70 answer tokens -> 3 blocks, last EOS-filled
    p2, a2 = list(range(5, 25)), list(range(200, 280))           # truncated: 80 tokens -> 2 full blocks only
    b = collate([(p1, a1, True), (p2, a2, False)], step=7, seed=42, device="cpu")
    assert n_blocks(70, True) == 3 and n_blocks(80, False) == 2
    assert b["rows"].tolist() == [0, 0, 0, 1, 1]
    assert b["anchors"].tolist() == [19, 51, 83, 19, 51]
    assert b["tgt"][2].tolist() == a1[64:] + [EOS_ID] * 26
    assert b["region"][2].tolist() == [0] * 6 + [1] * 26
    assert b["n_ans"].tolist() == [32, 32, 6, 32, 32] and b["n_pad"].tolist() == [0, 0, 26, 0, 0]
    ids = b["ids"]
    assert ids[0, :90].tolist() == p1 + a1 and b["attn"][0].sum() == 90       # right padding, positions from 0
    assert (b["canvas"][:, 0] == ids[b["rows"], b["anchors"]]).all()           # slot 0 is the anchor token
    m = b["masked"]
    assert m.any(1).all()
    assert torch.equal(b["canvas"][:, 1:], torch.where(m, MASK_ID, b["tgt"]))
    for i in range(5):                                                         # tower columns <= anchor, real only
        assert b["kv_valid"][i].nonzero().max().item() == b["anchors"][i].item()
    b2 = collate([(p1, a1, True), (p2, a2, False)], step=7, seed=42, device="cpu")
    assert torch.equal(b["masked"], b2["masked"]) and torch.equal(b["t"], b2["t"])   # (seed, step) determinism
    b3 = collate([(p1, a1, True), (p2, a2, False)], step=8, seed=42, device="cpu")
    assert not torch.equal(b["masked"], b3["masked"])


def check_train_step_loss_is_finite_and_near_uniform(cfg, bundle, groups):
    from inkling_dlm.data import collate as c
    from inkling_dlm.train import train_step
    b = c(groups[0], 0, cfg.seed, cfg.device, cfg.block)
    out = train_step(bundle.pm, b, cfg.micro_blocks)
    bundle.pm.zero_grad(set_to_none=True)
    assert math.isfinite(out["loss"]) and out["blocks"] == b["rows"].shape[0]
    # a random tiny model is near-uniform over the 200,058 logits: per-token CE ~ ln V, inflated by 1/t on average
    assert out["loss_answer"] > 0.5 * math.log(200_058)


def check_chunked_softcap_ce_matches_logits_path(cfg, bundle, groups, fresh_lora):
    """tower.canvas_ce (Unsloth's chunked CE with the final logit softcap) gives the same per-slot CE and the same
    gradients as F.cross_entropy on canvas_logits (lm_head -> float -> softcap), with non-zero LoRA-B."""
    from inkling_dlm.evaluate import tower_states
    from inkling_dlm.tower import canvas_ce, canvas_logits, view_for_batch
    pm = bundle.pm
    with torch.no_grad():
        for n, p in pm.named_parameters():
            if "lora_B" in n:
                p.normal_(0, 1e-2, generator=torch.Generator().manual_seed(1))
    b = collate(groups[0], 5, cfg.seed, cfg.device, cfg.block)
    k, v, s = tower_states(pm, b)
    idx = torch.arange(min(6, b["rows"].shape[0]))
    sel, tgt = b["masked"][idx], b["tgt"][idx][b["masked"][idx]]
    params = [p for p in pm.parameters() if p.requires_grad]
    out = []
    for fused in (False, True):
        view = view_for_batch(pm, k, v, s, b, idx)
        if fused:
            ce = canvas_ce(pm, view, b["canvas"][idx], sel, tgt)
        else:
            ce = F.cross_entropy(canvas_logits(pm, view, b["canvas"][idx], select=sel), tgt, reduction="none")
        w = torch.linspace(0.5, 2.0, ce.shape[0])
        grads = torch.autograd.grad((ce * w).sum(), params, allow_unused=True)
        out.append((ce.detach(), torch.cat([g.flatten() for g in grads if g is not None])))
    (ce0, g0), (ce1, g1) = out
    assert ce0.shape == ce1.shape and ce0.numel() == int(sel.sum())
    assert (ce0 - ce1).abs().max().item() < 1e-4 * max(1.0, ce0.abs().max().item())
    assert ((g0 - g1).norm() / g0.norm()).item() < 1e-4
