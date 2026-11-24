"""The notebook's flow on the tiny model: gates (all of them, incl. the speed/cost projection and its stop rule)
-> training -> post-training sanity checks, each logged to W&B (offline), plus the CLI's training guards."""
import json

import pytest

from inkling_dlm import cli
from inkling_dlm.gates import choose_setting, g_speed, peak_batch, result, run_gates, speed_batches
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
    out = run_gates(c, b, tr, groups=groups, speed=speed_batches(c, b.tok, 3), peak=peak_batch(c, b.tok))
    tr.finish()
    assert out["passed"], [g["detail"] for g in out["gates"] if not g["passed"]]
    names = [g["gate"] for g in out["gates"] if not g["gate"].startswith("G-mem/speed (probe")]
    assert names == ["G-load", "G-dispatch", "G-exact", "G-readonly", "G-grad", "G-ckpt", "G-mem/speed"]
    probes = [g for g in out["gates"] if g["gate"].startswith("G-mem/speed (probe")]
    assert len(probes) == len(c.gate_settings) and all(p["metrics"]["fits"] for p in probes)
    g = json.loads((tmp_path / "run" / "gates.json").read_text())
    assert g["passed"] and g["projection"]["train_cu"] > 0
    fastest = min(probes, key=lambda p: p["metrics"]["s_per_update"])["metrics"]
    assert g["setting"] == {k: fastest[k] for k in ("grad_ckpt", "micro_blocks", "dequant_chunk_mb")}
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
    (tmp_path / "run" / "gates.json").write_text(json.dumps({"passed": True, "setting": {
        "grad_ckpt": True, "micro_blocks": 8, "dequant_chunk_mb": 192}}))
    with pytest.raises(SystemExit, match="explicit approval"):
        cli.job_train(c, approved=False)
    assert (c.grad_ckpt, c.micro_blocks, c.dequant_chunk_mb) == (True, 8, 192)   # the gate-chosen setting is applied


def check_config_overrides():
    from inkling_dlm.config import load_config
    c = load_config("configs/a100.yaml", ["updates=10", "grad_ckpt=true", "lr=2e-5", "cache_dir=none",
                                          "tags=a,b"])
    assert (c.updates, c.grad_ckpt, c.lr, c.cache_dir, c.tags) == (10, True, 2e-5, None, ["a", "b"])
    with pytest.raises(KeyError):
        load_config("configs/a100.yaml", ["nope=1"])


def check_speed_batches_follow_training_order_and_peak_batch_is_worst(cfg, bundle):
    from inkling_dlm.data import n_blocks
    sb = speed_batches(cfg, bundle.tok, 3)
    assert len(sb) == 3 and all(len(x) == cfg.examples_per_update for x in sb)
    pk = peak_batch(cfg, bundle.tok)
    blocks = lambda grp: sum(n_blocks(len(a), e, cfg.block) for _, a, e in grp)
    assert len(pk) == cfg.examples_per_update and all(blocks(pk) >= blocks(x) for x in sb)


def check_choose_setting_keeps_the_fastest_that_fits(cfg, bundle, monkeypatch, tmp_path):
    from inkling_dlm import gates
    c = _cfg(cfg, tmp_path, gate_settings=[[True, 64], [True, 32], [False, 16]], dequant_fallback_mb=192)
    timing = {(True, 64): (3.0, False), (True, 32): (4.0, True), (False, 16): (2.0, True)}

    def fake(bundle, cfg, batches, tracker=None, peak=None, name="", check_cost=True, warm=5):
        s, fits = timing[(cfg.grad_ckpt, cfg.micro_blocks)]
        if cfg.dequant_chunk_mb:
            fits = True
        return result(name, fits, "fake", fits=fits, s_per_update=s, **gates.setting(cfg))
    monkeypatch.setattr(gates, "g_speed", fake)
    _, best = choose_setting(bundle, c, [], None)
    assert best == (False, 16, None)                       # fastest among those that fit
    timing.update({(False, 16): (2.0, False), (True, 32): (4.0, False)})
    res, best = choose_setting(bundle, c, [], None)
    assert best == (False, 16, 192) and len(res) == 6      # nothing fits: same ladder with the decode bounded
    gates.apply_setting(c, bundle, False, c.micro_blocks, None)
