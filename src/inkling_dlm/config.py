"""Run configuration: one dataclass, loaded from YAML plus `key=value` overrides. Secrets never live here."""
from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

from . import constants as K


@dataclass
class RunConfig:
    name: str = "main"                    # run name suffix; W&B name = f"{job}-{name}"
    # model
    model_repo: str = K.MODEL_REPO
    model_revision: str = K.MODEL_REVISION
    tiny: bool = False                    # CPU verification: random tiny Inkling built from the real config/code
    dtype: str = "bfloat16"
    device: str = "cuda"
    experts_impl: str = "unsloth_packed"  # packed NVFP4/MXFP8 experts (Unsloth kernels); "grouped_mm" = BF16 (80 GB)
    # data
    data_repo: str = K.DATA_REPO
    data_revision: str = K.DATA_REVISION
    data_local_dir: str | None = None     # directory holding data/{split}-*.parquet (skips the Hub download)
    cache_dir: str | None = None          # tokenised-data cache (defaults to run_dir/cache)
    maxlen: int = 256
    block: int = K.S
    max_examples: int | None = None       # cap the selection (verification only)
    # optimisation (BYOD Table 7 unless noted)
    updates: int = 25_000
    examples_per_update: int = 16
    lr: float = 1e-5
    lr_fwd: float = 1e-4                  # forward bias bank (guide's choice, untested)
    weight_decay: float = 0.01
    warmup: int = 100
    clip: float = 0.5
    seed: int = 42
    lora_r: int = 1024
    lora_alpha: int = 1024
    micro_blocks: int = 16
    grad_ckpt: bool = False
    # bookkeeping
    run_dir: str = "/content/drive/MyDrive/inkling-dlm/runs/main"
    stage_dir: str = "/content/stage"
    save_every: int = 500
    keep: int = 2
    eval_every: int = 1000
    eval_batches: int = 4
    stop_after: int | None = None         # stop (cleanly, after a save) at this update: pilots and resume checks
    # gates and cost
    gate_updates: int = 30
    cu_per_hour: float = 8.88
    cu_limit: float = 200.0
    mem_limit_gb: float = 38.0            # also capped at 92% of the device by the speed gate
    exact_top1_min: float = 0.995
    exact_dlogit_max: float = 0.1
    ckpt_grad_rel_max: float = 1e-2
    # sanity
    sample_max_blocks: int = 8
    copy_cases: int = 200
    # tracking (the key comes from WANDB_API_KEY or wandb_key_file, never from this config)
    wandb_entity: str = "narendra-patwardhan-independent"
    wandb_project: str = "inkling-dlm"
    wandb_host: str = "https://api.forge.coreweave.com"
    wandb_mode: str = "online"            # online | offline (offline only for CPU verification)
    wandb_key_file: str = "/content/wandb.key"
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @property
    def torch_dtype(self):
        import torch
        return getattr(torch, self.dtype)


def _coerce(f: dataclasses.Field, raw: str):
    if raw.lower() in ("none", "null"):
        return None
    t = str(f.type)
    if "bool" in t:
        return raw.lower() in ("1", "true", "yes", "on")
    if "int" in t and "float" not in t:
        return int(raw)
    if "float" in t:
        return float(raw)
    if "list" in t:
        return json.loads(raw) if raw.startswith("[") else [x for x in raw.split(",") if x]
    return raw


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> RunConfig:
    data: dict[str, Any] = {}
    if path:
        data = yaml.safe_load(Path(path).read_text()) or {}
    known = {f.name: f for f in fields(RunConfig)}
    unknown = set(data) - set(known)
    if unknown:
        raise KeyError(f"unknown config keys: {sorted(unknown)}")
    cfg = RunConfig(**data)
    for ov in overrides or []:
        k, _, v = ov.partition("=")
        if k not in known:
            raise KeyError(f"unknown override {k!r}")
        setattr(cfg, k, _coerce(known[k], v))
    return cfg
