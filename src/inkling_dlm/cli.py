"""Command line: `python -m inkling_dlm <job> --config configs/a100.yaml [key=value ...]`.

Jobs: gates | train | sanity | pipeline (gates -> train -> sanity) | data-stats.
Real training refuses to start unless the gates passed for this run_dir and `--approved` is given."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import load_config


def _bundle(cfg):
    from .model import build
    return build(cfg)


def _tracker(cfg, job, resume_id=None, extra=None):
    from .tracking import Tracker
    return Tracker(cfg, job, resume_id=resume_id, extra_config=extra)


def job_gates(cfg, bundle=None) -> dict:
    from .gates import run_gates
    bundle = bundle or _bundle(cfg)
    tr = _tracker(cfg, "gates")
    out = run_gates(cfg, bundle, tr)
    tr.finish(exit_code=0 if out["passed"] else 1)
    return out


def apply_gate_setting(cfg) -> dict | None:
    p = Path(cfg.run_dir) / "gates.json"
    if not p.exists():
        return None
    g = json.loads(p.read_text())
    if g.get("passed"):
        cfg.grad_ckpt = g["setting"]["grad_ckpt"]
        cfg.micro_blocks = g["setting"]["micro_blocks"]
    return g


def job_train(cfg, approved: bool, bundle=None) -> dict:
    from .train import run_training
    g = apply_gate_setting(cfg)
    if not cfg.tiny:
        if not g or not g.get("passed"):
            raise SystemExit(f"refusing to train: gates have not passed for {cfg.run_dir} (run `gates` first)")
        if not approved:
            raise SystemExit("refusing to train: the main run needs explicit approval (--approved)")
    bundle = bundle or _bundle(cfg)
    extra = {"gates": g and {"projection": g.get("projection"), "setting": g.get("setting")}}
    return run_training(cfg, bundle, lambda rid: _tracker(cfg, "train", rid, extra))


def job_sanity(cfg, state=None, bundle=None) -> dict:
    from .sanity import run_sanity
    bundle = bundle or _bundle(cfg)
    tr = _tracker(cfg, "sanity")
    out = run_sanity(cfg, bundle, tr, state_path=state)
    tr.finish()
    return out


def job_data_stats(cfg) -> dict:
    from .data import data_stats, load_training_examples
    from .model import load_tokenizer
    data, info = load_training_examples(cfg, load_tokenizer(cfg))
    st = data_stats(data, cfg.block) | info
    print(json.dumps(st, indent=1))
    return st


def main(argv=None):
    ap = argparse.ArgumentParser(prog="inkling-dlm")
    ap.add_argument("job", choices=["gates", "train", "sanity", "pipeline", "data-stats"])
    ap.add_argument("--config", default=None)
    ap.add_argument("--approved", action="store_true", help="the owner approved the main training run")
    ap.add_argument("--state", default=None, help="state directory for sanity (default: newest)")
    ap.add_argument("overrides", nargs="*", help="key=value config overrides")
    a = ap.parse_args(argv)
    cfg = load_config(a.config, a.overrides)
    if a.job == "gates":
        sys.exit(0 if job_gates(cfg)["passed"] else 1)
    if a.job == "train":
        job_train(cfg, a.approved)
    elif a.job == "sanity":
        job_sanity(cfg, a.state)
    elif a.job == "data-stats":
        job_data_stats(cfg)
    elif a.job == "pipeline":
        bundle = _bundle(cfg)
        if not job_gates(cfg, bundle)["passed"]:
            sys.exit(1)
        if not a.approved:
            print("[pipeline] gates passed; training not approved, stopping here", flush=True)
            return
        job_train(cfg, True, bundle)
        job_sanity(cfg, None, bundle)


if __name__ == "__main__":
    main()
