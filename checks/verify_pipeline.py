"""The notebook's flow on the tiny model: gates (all of them, incl. the speed/cost projection and its stop rule)
-> training -> post-training sanity checks, each logged to W&B (offline), plus the CLI's training guards."""
import json

import pytest

from inkling_dlm import cli
from inkling_dlm.gates import g_speed, run_gates, speed_batches
from inkling_dlm.model import build_tiny
from inkling_dlm.tracking import Tracker


def _cfg(cfg, tmp_path, **over):
    c = type(cfg)(**cfg.to_dict())
    c.run_dir, c.stage_dir, c.cache_dir = str(tmp_path / "run"), str(tmp_path / "stage"), str(tmp_path / "cache")
    for k, v in over.items():
        setattr(c, k, v)
    return c


def check_gates_train_sanity_end_to_end(cfg, tmp_path, groups):
    c = _cfg(cfg, tmp_path, updates=4, save_every=2, eval_every=2, max_examples=32, sample_max_blocks=1,
             copy_cases=4)
    b = build_tiny(c, window=64)
    tr = Tracker(c, "gates")
    out = run_gates(c, b, tr, groups=groups, speed=speed_batches(c, b.tok, 3))
    tr.finish()
    assert out["passed"], [g["detail"] for g in out["gates"] if not g["passed"]]
    names = [g["gate"] for g in out["gates"]]
    assert names == ["G-load", "G-dispatch", "G-exact", "G-readonly", "G-grad", "G-ckpt", "G-mem/speed"]
    g = json.loads((tmp_path / "run" / "gates.json").read_text())
    assert g["passed"] and g["projection"]["train_cu"] > 0
    res = cli.job_train(c, approved=False, bundle=b)          # tiny: no approval needed
    assert res["step"] == 4
    s = cli.job_sanity(c, bundle=b)
    assert s["step"] == 4 and len(s["final"]["samples"]) == 8
    assert set(s["init"]["heldout"]) == set(s["final"]["heldout"])
    assert (tmp_path / "run" / "sanity.json").exists()


def check_hardware_gate_stops_before_loading(cfg, tmp_path, monkeypatch):
    c = _cfg(cfg, tmp_path, tiny=False, device="cuda")
    called = []
    monkeypatch.setattr(cli, "_bundle", lambda cfg: called.append(1))
    out = cli.job_gates(c)
    assert not out["passed"] and out["gates"][0]["gate"] == "G-hardware" and not called


def check_speed_gate_stops_above_the_cu_limit(cfg, tmp_path, bundle):
    c = _cfg(cfg, tmp_path, cu_limit=1e-6, updates=25_000)
    r = g_speed(bundle, c, speed_batches(c, bundle.tok, 2))
    assert not r["passed"] and "projected" in r["detail"]


def check_real_training_refuses_without_gates_or_approval(cfg, tmp_path):
    c = _cfg(cfg, tmp_path, tiny=False)
    with pytest.raises(SystemExit, match="gates have not passed"):
        cli.job_train(c, approved=True)
    (tmp_path / "run").mkdir(parents=True)
    (tmp_path / "run" / "gates.json").write_text(json.dumps({"passed": True, "setting": {"grad_ckpt": True,
                                                                                            "micro_blocks": 8}}))
    with pytest.raises(SystemExit, match="explicit approval"):
        cli.job_train(c, approved=False)
    assert c.grad_ckpt is True and c.micro_blocks == 8          # the gate-chosen setting is applied


def check_config_overrides():
    from inkling_dlm.config import load_config
    c = load_config("configs/a100.yaml", ["updates=10", "grad_ckpt=true", "lr=2e-5", "cache_dir=none",
                                          "tags=a,b"])
    assert (c.updates, c.grad_ckpt, c.lr, c.cache_dir, c.tags) == (10, True, 2e-5, None, ["a", "b"])
    with pytest.raises(KeyError):
        load_config("configs/a100.yaml", ["nope=1"])
