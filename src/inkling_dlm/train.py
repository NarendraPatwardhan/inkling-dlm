"""Training: one update = one length-bucketed group of examples -> one shared tower pass -> one canvas per block."""
from __future__ import annotations

import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from transformers import get_cosine_schedule_with_warmup

from .checkpoint import load_state, save_state
from .cost import projection
from .data import batch_order, collate, data_stats, eval_batches, load_training_examples
from .evaluate import heldout_ce, tower_states
from .loss import block_loss
from .tower import canvas_ce, view_for_batch


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def sync(dev):
    if torch.device(dev).type == "cuda":
        torch.cuda.synchronize()


def peak_mem_gb(dev) -> float:
    return torch.cuda.max_memory_allocated() / 1e9 if torch.device(dev).type == "cuda" else 0.0


def peak_reserved_gb(dev) -> float:
    return torch.cuda.max_memory_reserved() / 1e9 if torch.device(dev).type == "cuda" else 0.0


def apply_memory_setting(cfg, bundle) -> None:
    """Make the model follow cfg (grad_ckpt, dequant_chunk_mb), whatever state the gates or a previous job left."""
    from .unsloth_ops import set_dequant_chunk_mb
    lm = bundle.pm.base_model.model
    if cfg.grad_ckpt:
        lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    else:
        lm.gradient_checkpointing_disable()
    set_dequant_chunk_mb(cfg.dequant_chunk_mb)


def train_step(pm, b, micro: int, limit: int | None = None) -> dict:
    """Forward + backward for one batch (gradients accumulate on the trainable tensors). limit: gate use only."""
    k, v, s = tower_states(pm, b)
    Bc = b["rows"].shape[0] if limit is None else min(b["rows"].shape[0], limit)
    dev = b["rows"].device
    tot, ans, pad = 0.0, 0.0, 0.0
    for c0 in range(0, Bc, micro):
        idx = torch.arange(c0, min(Bc, c0 + micro), device=dev)
        view = view_for_batch(pm, k, v, s, b, idx)
        sel = b["masked"][idx]
        ce = canvas_ce(pm, view, b["canvas"][idx], sel, b["tgt"][idx][sel])
        blk = sel.nonzero()[:, 0]
        loss, la, lp = block_loss(ce, b["t"][idx][blk], b["region"][idx][sel], blk, b["n_ans"][idx], b["n_pad"][idx])
        w = len(idx) / Bc
        (loss * w).backward()
        tot += float(loss.detach()) * w
        ans += float(la) * w
        pad += float(lp) * w
    return dict(loss=tot, loss_answer=ans, loss_pad=pad, blocks=Bc, tower_width=int(b["ids"].shape[1]),
                masked_slots=int(b["masked"][:Bc].sum()))


def make_optimizer(cfg, bundle):
    opt = torch.optim.AdamW([{"params": bundle.lora, "lr": cfg.lr, "weight_decay": cfg.weight_decay},
                             {"params": bundle.fwd, "lr": cfg.lr_fwd, "weight_decay": 0.0}])
    sched = get_cosine_schedule_with_warmup(opt, cfg.warmup, cfg.updates)
    return opt, sched


def grad_norm(params) -> float:
    gs = [p.grad.detach().float().norm() for p in params if p.grad is not None]
    return float(torch.stack(gs).norm()) if gs else 0.0


class NonFinite(RuntimeError):
    pass


def run_training(cfg, bundle, tracker_factory, data=None) -> dict:
    """tracker_factory(resume_id) -> Tracker. Resumes from the newest complete state in cfg.run_dir."""
    seed_all(cfg.seed)
    pm, tok, dev = bundle.pm, bundle.tok, cfg.device
    run_dir = Path(cfg.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    if data is None:
        data, dinfo = load_training_examples(cfg, tok)
    else:
        dinfo = {"data_id": "provided", "cached": False}
    stats = data_stats(data, cfg.block)
    lengths = data.lengths if hasattr(data, "lengths") else np.array([len(p) + len(a) for p, a, _ in data])
    order = batch_order(lengths, cfg.seed, cfg.examples_per_update)
    opt, sched = make_optimizer(cfg, bundle)
    st = load_state(cfg, pm, opt, sched)
    step = st["step"] if st else 0
    id_file = run_dir / "wandb_id"
    tracker = tracker_factory(id_file.read_text().strip() if id_file.exists() else None)
    id_file.write_text(tracker.id)
    tracker.summary({f"data/{k}": v for k, v in stats.items()} | {"data/data_id": dinfo["data_id"],
                    "data/batches_per_epoch": len(order), "resumed_from": step})
    print(f"[train] start at update {step}/{cfg.updates}; {stats['examples']} examples, "
          f"{len(order)} batches/epoch; W&B run {tracker.id}", flush=True)
    evals = eval_batches(cfg, tok, cfg.eval_batches) if cfg.eval_every else []
    if step == 0 and evals:
        ev = heldout_ce(pm, evals, cfg, adapters=True)
        tracker.log({f"eval/heldout_{k}": v for k, v in ev.items()}, update=0)
    params = bundle.lora + bundle.fwd
    pm.train()
    if torch.device(dev).type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    times, bad = [], 0
    last = {}
    while step < cfg.updates:
        t0 = time.time()
        b = collate([data[i] for i in order[step % len(order)]], step, cfg.seed, dev, cfg.block)
        out = train_step(pm, b, cfg.micro_blocks)
        if not math.isfinite(out["loss"]):
            bad += 1
            opt.zero_grad(set_to_none=True)
            if bad >= 3:
                tracker.summary({"abort": f"non-finite loss at update {step}"})
                tracker.finish(exit_code=1)
                raise NonFinite(f"non-finite loss 3x at update {step}")
            step += 1
            sched.step()
            continue
        bad = 0
        gn_lora, gn_fwd = grad_norm(bundle.lora), grad_norm(bundle.fwd)
        gn = float(torch.nn.utils.clip_grad_norm_(params, cfg.clip))
        lr = sched.get_last_lr()
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        sync(dev)
        dt = time.time() - t0
        times.append(dt)
        med = float(np.median(times[-50:]))
        proj = projection(med, cfg.updates, cfg.cu_per_hour)
        last = dict(loss=out["loss"], loss_answer=out["loss_answer"], loss_pad=out["loss_pad"])
        tracker.log({"train/loss": out["loss"], "train/loss_answer": out["loss_answer"],
                     "train/loss_pad": out["loss_pad"], "train/grad_norm": gn, "train/grad_norm_lora": gn_lora,
                     "train/grad_norm_fwd": gn_fwd, "train/lr_lora": lr[0], "train/lr_fwd": lr[1],
                     "sys/s_per_update": dt, "sys/blocks": out["blocks"], "sys/tower_width": out["tower_width"],
                     "sys/masked_slots": out["masked_slots"], "sys/peak_mem_gb": peak_mem_gb(dev),
                     "sys/peak_reserved_gb": peak_reserved_gb(dev),
                     "sys/projected_train_cu": proj["train_cu"],
                     "sys/remaining_h": med * (cfg.updates - step) / 3600}, update=step)
        stop_now = cfg.stop_after is not None and step >= cfg.stop_after
        if step % cfg.save_every == 0 or step == cfg.updates or stop_now:
            save_state(cfg, pm, opt, sched, step, extra={"wandb_id": tracker.id})
        if evals and cfg.eval_every and step % cfg.eval_every == 0:
            ev = heldout_ce(pm, evals, cfg, adapters=True)
            tracker.log({f"eval/heldout_{k}": v for k, v in ev.items()}, update=step)
        if stop_now:
            break
    tracker.summary({"final_update": step} | {f"final/{k}": v for k, v in last.items()})
    tracker.finish()
    return dict(step=step, **last)
