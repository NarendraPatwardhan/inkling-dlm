"""Weights & Biases for every job (gates, training, sanity, verification). The API key comes from WANDB_API_KEY or a
key file read at runtime; it is never printed, logged or stored in the config."""
from __future__ import annotations

import os
import platform
import subprocess
from pathlib import Path

KEY_FILES = ("~/.config/wandb/key",)


def load_wandb_key(cfg) -> bool:
    """Set WANDB_API_KEY from a key file if not already set. Returns whether a key is available (never prints it)."""
    if os.environ.get("WANDB_API_KEY"):
        return True
    for f in (cfg.wandb_key_file, os.environ.get("WANDB_KEY_FILE"), *KEY_FILES):
        if f and Path(os.path.expanduser(f)).is_file():
            key = Path(os.path.expanduser(f)).read_text().strip()
            if key:
                os.environ["WANDB_API_KEY"] = key
                return True
    return False


def git_info() -> dict:
    root = Path(__file__).resolve().parents[2]
    try:
        sha = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True,
                             check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
                                    capture_output=True, text=True).stdout.strip())
        return {"git_commit": sha, "git_dirty": dirty}
    except Exception:
        return {"git_commit": os.environ.get("INKLING_GIT_COMMIT", "unknown"), "git_dirty": None}


def env_info() -> dict:
    import torch
    import transformers
    import peft
    d = dict(python=platform.python_version(), torch=torch.__version__, transformers=transformers.__version__,
             peft=peft.__version__, cuda=torch.version.cuda)
    if torch.cuda.is_available():
        d.update(gpu=torch.cuda.get_device_name(0), capability=".".join(map(str, torch.cuda.get_device_capability())))
    return d


class Tracker:
    def __init__(self, cfg, job_type: str, resume_id: str | None = None, extra_config: dict | None = None):
        import wandb
        self.wandb = wandb
        os.environ.setdefault("WANDB_BASE_URL", cfg.wandb_host)
        has_key = load_wandb_key(cfg)
        mode = cfg.wandb_mode
        if mode == "online" and not has_key:
            raise RuntimeError("W&B is mandatory: set WANDB_API_KEY or provide wandb_key_file (no key found)")
        conf = cfg.to_dict() | git_info() | {"env": env_info()} | (extra_config or {})
        self.run = wandb.init(
            entity=cfg.wandb_entity, project=cfg.wandb_project, job_type=job_type, name=f"{job_type}-{cfg.name}",
            group=cfg.name, config=conf, tags=list(cfg.tags) + [job_type] + (["tiny"] if cfg.tiny else []),
            mode=mode, id=resume_id, resume="allow" if resume_id else None,
            dir=os.environ.get("WANDB_DIR"), settings=wandb.Settings(console="off", quiet=True))
        self.run.define_metric("update", hidden=True)
        for prefix in ("train/*", "eval/*", "sys/*"):
            self.run.define_metric(prefix, step_metric="update")

    @property
    def id(self) -> str:
        return self.run.id

    def log(self, metrics: dict, update: int | None = None):
        if update is not None:
            metrics = {"update": update, **metrics}
        self.run.log(metrics)

    def summary(self, d: dict):
        for k, v in d.items():
            self.run.summary[k] = v

    def table(self, key: str, columns: list[str], rows: list[list]):
        self.run.log({key: self.wandb.Table(columns=columns, data=rows)})

    def line(self, key: str, xs, ys, x: str, y: str, title: str):
        t = self.wandb.Table(columns=[x, y], data=[[a, b] for a, b in zip(xs, ys)])
        self.run.log({key: self.wandb.plot.line(t, x, y, title=title)})

    def bar(self, key: str, labels, values, label: str, value: str, title: str):
        t = self.wandb.Table(columns=[label, value], data=[[a, b] for a, b in zip(labels, values)])
        self.run.log({key: self.wandb.plot.bar(t, label, value, title=title)})

    def gates(self, results: list[dict]):
        self.table("gates/results", ["gate", "passed", "detail"],
                   [[r["gate"], r["passed"], r["detail"]] for r in results])
        self.summary({f"gate/{r['gate']}": r["passed"] for r in results})
        self.summary({"gates/all_passed": all(r["passed"] for r in results)})

    def finish(self, exit_code: int = 0):
        self.run.finish(exit_code=exit_code)
