"""W&B offline logging: config with git commit, metrics, gate table, sample table, plots; the key never reaches
the run files."""
import json
import os
from pathlib import Path

from inkling_dlm.tracking import Tracker, git_info, load_wandb_key


def check_offline_run_records_everything(cfg):
    from wandb_records import summarize
    tr = Tracker(cfg, "verify", extra_config={"note": "verification"})
    tr.log({"train/loss": 1.0, "train/grad_norm": 0.5}, update=1)
    tr.log({"train/loss": 0.9, "train/grad_norm": 0.4}, update=2)
    tr.gates([{"gate": "G-x", "passed": True, "detail": "ok", "metrics": {}}])
    tr.table("sanity/samples", ["prompt", "out"], [["p", "o"]])
    tr.line("plots/line", [0.1, 0.5], [1.0, 2.0], "t", "ce", "ce vs t")
    tr.bar("plots/bar", ["a", "b"], [1, 2], "k", "v", "bars")
    run_dir = Path(tr.run.dir).parent
    tr.finish()
    files = [f for f in run_dir.rglob("*") if f.is_file()]
    rec = summarize(next(f for f in files if f.suffix == ".wandb"))
    conf = rec["config"]
    val = {k: json.loads(v) for k, v in conf.items()}
    val = {k: v["value"] if isinstance(v, dict) and "value" in v else v for k, v in val.items()}
    assert val["git_commit"] == git_info()["git_commit"] and len(val["git_commit"]) == 40
    assert val["note"] == "verification" and val["lora_r"] == cfg.lora_r and "torch" in val["env"]
    assert [h["train/loss"] for h in rec["history"] if "train/loss" in h] == ["1", "0.9"]
    assert [h["update"] for h in rec["history"] if "train/loss" in h] == ["1", "2"]
    logged = {k for h in rec["history"] for k in h}
    for key in ("gates/results", "sanity/samples", "plots/line_table", "plots/bar_table"):
        assert f"{key}/_type" in logged and f"{key}/nrows" in logged, key
    assert rec["summary"]["gates/all_passed"] == "true"
    assert sum(f.name.endswith(".table.json") for f in files) >= 4                 # table media written
    if load_wandb_key(cfg):                                   # compare in memory only; never printed
        key = os.environ["WANDB_API_KEY"].encode()
        for f in files:
            assert key not in f.read_bytes(), "W&B key leaked into run files"
    assert "WANDB_API_KEY" not in conf and not any("api_key" in k.lower() for k in conf)


def check_online_mode_requires_a_key(cfg, monkeypatch):
    import pytest
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.setattr("inkling_dlm.tracking.KEY_FILES", ())
    monkeypatch.delenv("WANDB_KEY_FILE", raising=False)
    c = type(cfg)(**cfg.to_dict())
    c.wandb_mode, c.wandb_key_file = "online", "/nonexistent/key"
    with pytest.raises(RuntimeError, match="W&B is mandatory"):
        Tracker(c, "verify")
