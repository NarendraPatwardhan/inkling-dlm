"""Preemption-safe state: trainable tensors, optimizer, scheduler, step, RNG, W&B id. A state directory is complete
only after the final rename; `.partial` directories are ignored and cleaned. The last `keep` states are kept."""
from __future__ import annotations

import random
import shutil
from pathlib import Path

import numpy as np
import torch


def trainable_state(pm) -> dict:
    return {n: p.detach().cpu().clone() for n, p in pm.named_parameters() if p.requires_grad}


def completed_states(run_dir) -> list[Path]:
    root = Path(run_dir)
    return sorted(p for p in root.glob("state-*") if p.is_dir() and not p.name.endswith(".partial"))


def save_state(cfg, pm, opt, sched, step: int, extra: dict | None = None) -> Path:
    stage = Path(cfg.stage_dir) / f"state-{step:06d}"
    shutil.rmtree(stage, ignore_errors=True)
    stage.mkdir(parents=True, exist_ok=True)
    rng = dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
               cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
    torch.save({"trainable": trainable_state(pm), "opt": opt.state_dict(), "sched": sched.state_dict(),
                "step": step, "rng": rng, "config": cfg.to_dict(), "extra": extra or {}}, stage / "state.pt")
    root = Path(cfg.run_dir)
    root.mkdir(parents=True, exist_ok=True)
    part = root / f"state-{step:06d}.partial"
    shutil.rmtree(part, ignore_errors=True)
    shutil.copytree(stage, part)
    final = root / f"state-{step:06d}"
    shutil.rmtree(final, ignore_errors=True)
    part.rename(final)                                       # complete only after the rename
    shutil.rmtree(stage, ignore_errors=True)
    for old in completed_states(root)[:-cfg.keep]:
        shutil.rmtree(old, ignore_errors=True)
    for stale in root.glob("state-*.partial"):
        shutil.rmtree(stale, ignore_errors=True)
    return final


def load_state(cfg, pm, opt=None, sched=None, path: str | Path | None = None) -> dict | None:
    """Restore the newest complete state (or `path`). Returns the state dict (minus tensors) or None."""
    if path is None:
        done = completed_states(cfg.run_dir)
        if not done:
            return None
        path = done[-1]
    st = torch.load(Path(path) / "state.pt", map_location="cpu", weights_only=False)
    params = dict(pm.named_parameters())
    missing = [n for n in st["trainable"] if n not in params]
    if missing:
        raise KeyError(f"state has tensors absent from the model: {missing[:3]}")
    absent = [n for n, p in params.items() if p.requires_grad and n not in st["trainable"]]
    if absent:
        raise KeyError(f"model has trainable tensors absent from the state: {absent[:3]}")
    with torch.no_grad():
        for n, v in st["trainable"].items():
            params[n].copy_(v.to(params[n].device, params[n].dtype))
    if opt is not None:
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        r = st["rng"]
        random.setstate(r["python"])
        np.random.set_state(r["numpy"])
        torch.set_rng_state(r["torch"])
        if r.get("cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(r["cuda"])
    return {k: v for k, v in st.items() if k not in ("trainable", "opt", "sched", "rng")} | {"path": str(path)}
