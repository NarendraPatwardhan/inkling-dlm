"""LoRA-off logit equivalence (G-exact): causal canvas over the frozen tower reproduces the parent's full-sequence
logits, through the training path and the sampler's inference path, with and without sliding truncation, and still
after the trainable tensors move."""
import torch

from inkling_dlm.gates import g_exact
from inkling_dlm.model import build_tiny


def check_exact_at_init(cfg, bundle, groups):
    r = g_exact(bundle, cfg, groups)
    assert r["passed"], r["detail"]
    m = r["metrics"]
    assert m["top1"] == 1.0 and m["inf_top1"] == 1.0
    assert m["dlogit"] < 1e-4 and m["inf_dlogit"] < 1e-4 and m["lora_on_dlogit"] < 1e-4


def check_exact_after_trainables_move(cfg, fresh_lora, groups):
    with torch.no_grad():
        for n, p in fresh_lora.pm.named_parameters():
            if p.requires_grad:
                p.add_(torch.randn_like(p) * 1e-2)
    r = g_exact(fresh_lora, cfg, groups)
    assert r["metrics"]["dlogit"] < 1e-4 and r["metrics"]["inf_dlogit"] < 1e-4, r["detail"]
    assert r["metrics"]["lora_on_dlogit"] > 1e-4             # adapters on now changes the output


def check_exact_with_window_longer_than_rows(cfg, groups):
    b = build_tiny(cfg, window=1024)                           # the real window: no truncation in 256-token rows
    r = g_exact(b, cfg, groups)
    assert r["passed"] and r["metrics"]["dlogit"] < 1e-4, r["detail"]
