"""Shared fixtures for the CPU verification suite: a tiny random Inkling (real config fields, real modeling code,
real tokenizer; no weights downloaded) and real LAD rows. Set INKLING_LAD_DIR to a local LAD snapshot to avoid the
Hub download of the validation split."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
VERIFY = ROOT / ".verify"
os.environ.setdefault("WANDB_MODE", "offline")
os.environ.setdefault("WANDB_DIR", str(VERIFY / "wandb"))
os.environ.setdefault("WANDB_SILENT", "true")
(VERIFY / "wandb").mkdir(parents=True, exist_ok=True)
torch.use_deterministic_algorithms(False)


def tiny_cfg(**over):
    from inkling_dlm.config import load_config
    cfg = load_config(ROOT / "configs" / "cpu-tiny.yaml")
    cfg.data_local_dir = os.environ.get("INKLING_LAD_DIR") or cfg.data_local_dir
    for k, v in over.items():
        setattr(cfg, k, v)
    return cfg


@pytest.fixture(scope="session")
def cfg():
    return tiny_cfg()


@pytest.fixture(scope="session")
def bundle(cfg):
    from inkling_dlm.model import build_tiny
    return build_tiny(cfg, window=64)          # window 64 < typical row length: sliding truncation is exercised


@pytest.fixture(scope="session")
def groups(cfg, bundle):
    from inkling_dlm.data import eval_batches
    g = eval_batches(cfg, bundle.tok, 2, first=400)
    assert len(g) == 2
    return g


@pytest.fixture()
def fresh_lora(bundle):
    """Restore every trainable tensor after a check that perturbs them."""
    saved = {n: p.detach().clone() for n, p in bundle.pm.named_parameters() if p.requires_grad}
    yield bundle
    with torch.no_grad():
        for n, p in bundle.pm.named_parameters():
            if n in saved:
                p.copy_(saved[n])
    bundle.pm.zero_grad(set_to_none=True)
