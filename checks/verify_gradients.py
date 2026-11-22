"""Gradient flow (G-grad) and gradient-through-cache under checkpointing (G-ckpt)."""
from inkling_dlm.gates import g_ckpt, g_grad


def check_gradient_flow(cfg, fresh_lora, groups):
    r = g_grad(fresh_lora, cfg, groups)
    assert r["passed"], r["detail"]


def check_checkpointing_gives_identical_gradients(cfg, fresh_lora, groups):
    r = g_ckpt(fresh_lora, cfg, groups)
    assert r["passed"], r["detail"]
    assert r["metrics"]["rel"] < 1e-5                          # fp32 on CPU: replay is exact up to rounding
