"""Pre-training gates. Every gate returns {gate, passed, detail, metrics}; `run_gates` stops at the first hard
failure that makes the rest meaningless, logs everything to W&B and writes gates.json (the chosen GRAD_CKPT /
MICRO_BLOCKS setting and the cost projection) next to the run's checkpoints."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from .constants import S
from .cost import projection
from .data import collate, encode_all, eval_batches, lad_rows, selection_index
from .evaluate import tower_states
from .tower import canvas_logits, tower_fill, view_for_batch, view_from_inference_cache
from .train import peak_mem_gb, sync, train_step


def result(name, passed, detail, **metrics):
    tag = "PASS" if passed else "FAIL"
    print(f"[{tag}] {name}: {detail}", flush=True)
    return dict(gate=name, passed=bool(passed), detail=detail, metrics=metrics)


# ------------------------------------------------------------------------------------------------ load & dispatch
def g_hardware(cfg, min_gpu_gb: float = 70.0):
    """Before any download: the BF16 text model is ~52 GB, so only an 80 GB card can hold it (a 40 GB A100 cannot)."""
    if cfg.tiny or not cfg.device.startswith("cuda"):
        return result("G-hardware", True, f"device {cfg.device} (tiny verification)", skipped=True)
    if not torch.cuda.is_available():
        return result("G-hardware", False, "no CUDA device")
    p = torch.cuda.get_device_properties(0)
    gb = p.total_memory / 1e9
    cap = torch.cuda.get_device_capability()
    return result("G-hardware", gb >= min_gpu_gb and cap >= (8, 0),
                  f"{p.name}, {gb:.0f} GB, capability {cap} (need >= {min_gpu_gb:.0f} GB, sm80+)", gpu_gb=gb)


def g_load(bundle, cfg):
    if cfg.tiny:
        return result("G-load", True, "tiny random model (no checkpoint to load)", skipped=True)
    info = bundle.load_info or {}
    missing = list(info.get("missing_keys", []))
    unexpected = [k for k in info.get("unexpected_keys", []) if "visual" not in k and "mtp" not in k]
    mism = list(info.get("mismatched_keys", []))
    lm = bundle.pm.base_model.model
    dtypes = sorted({str(p.dtype) for p in lm.parameters()})
    return result("G-load", not missing and not unexpected and not mism,
                  f"missing {len(missing)}, unexpected non-vision/MTP {len(unexpected)}, mismatched {len(mism)}; "
                  f"dtypes {dtypes}", missing=missing[:20], unexpected=unexpected[:20])


def g_dispatch(bundle, cfg):
    """grouped_mm is selected (not the silent Python-loop fallback) and agrees with the reference expert loop."""
    lm = bundle.pm.base_model.model
    impl = getattr(lm.config, "_experts_implementation", None)
    ok, parts = impl == cfg.experts_impl, [f"experts_implementation={impl}"]
    if torch.cuda.is_available() and cfg.device.startswith("cuda"):
        cap = torch.cuda.get_device_capability()
        has = hasattr(torch.nn.functional, "grouped_mm") or hasattr(torch, "_grouped_mm")
        ok &= cap >= (8, 0) and has
        parts.append(f"capability {cap}, torch {torch.__version__}, grouped_mm available {has}")
    # functional comparison on one MoE block: grouped_mm vs eager
    moe = next(m for n, m in lm.named_modules() if n.endswith("mlp.experts"))
    x = torch.randn(64, lm.config.hidden_size, device=moe.gate_up_proj.device, dtype=moe.gate_up_proj.dtype)
    E, k = lm.config.n_routed_experts, lm.config.num_experts_per_tok
    g = torch.Generator(device="cpu").manual_seed(0)
    top_k_index = torch.stack([torch.randperm(E, generator=g)[:k] for _ in range(64)]).to(x.device)
    top_k_weights = torch.rand(64, k, generator=g).to(x.device, x.dtype)
    outs = {}
    with torch.no_grad():
        for name in (cfg.experts_impl, "eager"):
            lm.set_experts_implementation(name)
            outs[name] = moe(x, top_k_index, top_k_weights).float()
    lm.set_experts_implementation(cfg.experts_impl)
    rel = ((outs[cfg.experts_impl] - outs["eager"]).norm() / outs["eager"].norm().clamp(min=1e-12)).item()
    tol = 2e-2 if x.dtype == torch.bfloat16 else 1e-4
    ok &= rel < tol
    parts.append(f"grouped_mm vs eager rel diff {rel:.2e} (tol {tol})")
    return result("G-dispatch", ok, "; ".join(parts), rel_diff=rel)


# ------------------------------------------------------------------------------------------------ exactness
@torch.no_grad()
def g_exact(bundle, cfg, groups):
    """Adapters off + causal canvas + clean inputs must reproduce the parent's full-sequence logits, through the
    training path (record_past + compaction, batched canvases) and the inference path (sampler cache view)."""
    pm, dev = bundle.pm, cfg.device
    lm = pm.base_model.model
    pm.eval()
    b = collate(groups[0], 0, cfg.seed, dev, cfg.block)
    with pm.disable_adapter():
        ref = lm(input_ids=b["ids"], attention_mask=b["attn"]).logits
    keep = (b["anchors"] + S) < b["lens"][b["rows"]]
    idx = keep.nonzero()[:, 0][:16]
    k, v, s = tower_states(pm, b)
    view = view_for_batch(pm, k, v, s, b, idx, causal=True)
    r, a = b["rows"][idx], b["anchors"][idx]
    ar = torch.arange(S + 1, device=dev)
    clean = b["ids"][r[:, None], a[:, None] + ar]
    got = canvas_logits(pm, view, clean, adapters=False)
    want = ref[r[:, None], a[:, None] + ar[:S]]
    agree = (got.argmax(-1) == want.argmax(-1)).float().mean().item()
    dmax = (got - want).abs().max().item()
    # LoRA on at initialisation (B = 0) must equal LoRA off
    got_on = canvas_logits(pm, view, clean, adapters=True)
    lora_on_dmax = (got_on - got).abs().max().item()
    # inference path: tower over prompt+answer prefix of row 0, sampler view, causal canvas
    r0, a0 = int(r[0]), int(a[0])
    cache = tower_fill(pm, b["ids"][r0:r0 + 1, :a0 + 1], record_past=True)   # as the sampler
    iview = view_from_inference_cache(pm, cache, a0 + 1, causal=True)
    got_inf = canvas_logits(pm, iview, b["ids"][r0:r0 + 1, a0:a0 + S + 1], adapters=False)
    want_inf = ref[r0, a0:a0 + S][None]
    inf_agree = (got_inf.argmax(-1) == want_inf.argmax(-1)).float().mean().item()
    inf_dmax = (got_inf - want_inf).abs().max().item()
    lo = min(agree, inf_agree)
    ok = (lo >= cfg.exact_top1_min and max(dmax, inf_dmax) <= cfg.exact_dlogit_max
          and lora_on_dmax <= cfg.exact_dlogit_max)
    return result("G-exact", ok, f"train path top-1 {agree:.4f} max|dlogit| {dmax:.3g}; inference path top-1 "
                  f"{inf_agree:.4f} max|dlogit| {inf_dmax:.3g}; LoRA-on(B=0) vs off {lora_on_dmax:.3g}; "
                  f"{len(idx)} canvases", top1=agree, dlogit=dmax, inf_top1=inf_agree, inf_dlogit=inf_dmax,
                  lora_on_dlogit=lora_on_dmax)


@torch.no_grad()
def g_readonly(bundle, cfg, groups):
    """A canvas forward never writes to the tower states, and is a pure function of its inputs."""
    pm, dev = bundle.pm, cfg.device
    b = collate(groups[0], 1, cfg.seed, dev, cfg.block)
    k, v, s = tower_states(pm, b)
    snap = [t.clone() for t in k + v] + [t.clone() for L in s for t in L]
    idx = torch.arange(min(8, b["rows"].shape[0]), device=dev)
    view = view_for_batch(pm, k, v, s, b, idx)
    o1 = canvas_logits(pm, view, b["canvas"][idx])
    o2 = canvas_logits(pm, view, b["canvas"][idx])
    after = [t for t in k + v] + [t for L in s for t in L]
    same = all(torch.equal(x, y) for x, y in zip(snap, after))
    det = torch.equal(o1, o2)
    shapes_ok = all(x.shape == y.shape for x, y in zip(snap, after))
    return result("G-readonly", same and det and shapes_ok and view.reads > 0,
                  f"states unchanged {same}, repeat forward identical {det}, cache reads {view.reads}",
                  reads=view.reads)


def _grads(pm, b, cfg, limit):
    pm.zero_grad(set_to_none=True)
    train_step(pm, b, cfg.micro_blocks, limit=limit)
    return {n: p.grad.detach().float().clone() for n, p in pm.named_parameters()
            if p.requires_grad and p.grad is not None}


def g_grad(bundle, cfg, groups):
    """Gradients reach every LoRA-B tensor and every forward-bias bank; nothing frozen receives a gradient; after
    one optimizer-free perturbation of LoRA-B, LoRA-A tensors receive gradients too."""
    pm, dev = bundle.pm, cfg.device
    pm.train()
    b = collate(groups[0], 2, cfg.seed, dev, cfg.block)
    g = _grads(pm, b, cfg, limit=8)
    names = [n for n, p in pm.named_parameters() if p.requires_grad]
    lora_b = [n for n in names if "lora_B" in n]
    fwd = [n for n, p in pm.named_parameters() if p.requires_grad and n.endswith("proj_fwd")]
    zero_b = [n for n in lora_b if n not in g or g[n].abs().max() == 0]
    zero_f = [n for n in fwd if n not in g or g[n].abs().max() == 0]
    frozen = [n for n, p in pm.named_parameters() if not p.requires_grad and p.grad is not None]
    finite = all(torch.isfinite(t).all() for t in g.values())
    saved = {n: p.detach().clone() for n, p in pm.named_parameters() if "lora_B" in n}
    with torch.no_grad():
        for n, p in pm.named_parameters():
            if "lora_B" in n:
                p.normal_(0, 1e-3)
    g2 = _grads(pm, b, cfg, limit=8)
    with torch.no_grad():
        for n, p in pm.named_parameters():
            if n in saved:
                p.copy_(saved[n])
    pm.zero_grad(set_to_none=True)
    lora_a = [n for n in names if "lora_A" in n]
    zero_a = [n for n in lora_a if n not in g2 or g2[n].abs().max() == 0]
    ok = not zero_b and not zero_f and not frozen and finite and not zero_a
    return result("G-grad", ok, f"LoRA-B without grad {len(zero_b)}/{len(lora_b)}, LoRA-A without grad (after "
                  f"perturbing B) {len(zero_a)}/{len(lora_a)}, proj_fwd without grad {len(zero_f)}/{len(fwd)}, "
                  f"frozen tensors with grad {len(frozen)}, finite {finite}")


def g_ckpt(bundle, cfg, groups):
    """Same canvases with gradient checkpointing off vs on: identical gradients (read-only cache replay is safe,
    and the prompt is not silently dropped under checkpointing)."""
    pm = bundle.pm
    lm = pm.base_model.model
    b = collate(groups[0], 3, cfg.seed, cfg.device, cfg.block)
    saved = {n: p.detach().clone() for n, p in pm.named_parameters() if "lora_B" in n}
    with torch.no_grad():                                   # non-zero B so every LoRA tensor carries gradient
        for n, p in pm.named_parameters():
            if "lora_B" in n:
                p.normal_(0, 1e-3, generator=torch.Generator(device=p.device).manual_seed(0))
    was = lm.is_activation_checkpointing
    pm.train()
    grads = []
    for on in (False, True):
        if on:
            lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        else:
            lm.gradient_checkpointing_disable()
        grads.append(_grads(pm, b, cfg, limit=8))
    if not was:
        lm.gradient_checkpointing_disable()
    with torch.no_grad():
        for n, p in pm.named_parameters():
            if n in saved:
                p.copy_(saved[n])
    pm.zero_grad(set_to_none=True)
    keys = sorted(grads[0])
    a = torch.cat([grads[0][n].flatten() for n in keys])
    c = torch.cat([grads[1][n].flatten() for n in keys])
    rel = ((a - c).norm() / a.norm().clamp(min=1e-12)).item()
    same_keys = set(grads[0]) == set(grads[1])
    return result("G-ckpt", same_keys and rel < cfg.ckpt_grad_rel_max,
                  f"relative grad difference {rel:.2e} over {len(keys)} tensors", rel=rel)


# ------------------------------------------------------------------------------------------------ speed & cost
def speed_batches(cfg, tok, n):
    sel = selection_index()[: n * cfg.examples_per_update]
    data = encode_all(tok, lad_rows(cfg, "train", sel), cfg.maxlen)
    G = cfg.examples_per_update
    return [data[i * G:(i + 1) * G] for i in range(n)]


def g_speed(bundle, cfg, batches, tracker=None):
    """Real-data updates (forward+backward, no optimizer step) in the current setting; project the full run.
    Stops (FAIL) if the projection exceeds cfg.cu_limit or peak memory exceeds cfg.mem_limit_gb."""
    pm, dev = bundle.pm, cfg.device
    pm.train()
    if dev.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    times = []
    for i, ex in enumerate(batches):
        b = collate(ex, i, cfg.seed, dev, cfg.block)
        sync(dev)
        t0 = time.time()
        out = train_step(pm, b, cfg.micro_blocks)
        sync(dev)
        times.append(time.time() - t0)
        pm.zero_grad(set_to_none=True)
        if tracker:
            tracker.log({"gate_speed/s_per_update": times[-1], "gate_speed/loss": out["loss"],
                         "gate_speed/blocks": out["blocks"], "gate_speed/peak_mem_gb": peak_mem_gb(dev)})
    warm = min(5, len(times) - 1)
    s = float(np.median(times[warm:]))
    mem = peak_mem_gb(dev)
    p = projection(s, cfg.updates, cfg.cu_per_hour)
    ok = mem < cfg.mem_limit_gb and p["train_cu"] <= cfg.cu_limit
    return result("G-mem/speed", ok, f"grad_ckpt={cfg.grad_ckpt} micro={cfg.micro_blocks}: median {s:.2f} s/update, "
                  f"peak {mem:.1f} GB -> projected {p['train_hours']:.1f} h = {p['train_cu']:.0f} CU (limit "
                  f"{cfg.cu_limit:.0f})", peak_mem_gb=mem, **p)


# ------------------------------------------------------------------------------------------------ driver
def run_gates(cfg, bundle, tracker, groups=None, speed=None, pre=None) -> dict:
    t0 = time.time()
    res = list(pre or []) + [g_load(bundle, cfg)]
    if not res[-1]["passed"]:
        return _finish(cfg, tracker, res, t0)
    res.append(g_dispatch(bundle, cfg))
    groups = groups or eval_batches(cfg, bundle.tok, 2)
    for fn in (g_exact, g_readonly, g_grad, g_ckpt):
        res.append(fn(bundle, cfg, groups))
    if all(r["passed"] for r in res):
        speed = speed if speed is not None else speed_batches(cfg, bundle.tok, cfg.gate_updates)
        try:
            r = g_speed(bundle, cfg, speed, tracker)
        except torch.OutOfMemoryError:
            r = result("G-mem/speed", False, f"OOM with grad_ckpt={cfg.grad_ckpt} micro={cfg.micro_blocks}")
        if not r["passed"] and not cfg.grad_ckpt:
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            print("[gates] retrying the speed gate with gradient checkpointing on", flush=True)
            cfg.grad_ckpt = True
            bundle.pm.base_model.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
            res.append(r | {"gate": "G-mem/speed (no ckpt)"})
            try:
                r = g_speed(bundle, cfg, speed, tracker)
            except torch.OutOfMemoryError:
                r = result("G-mem/speed", False, "OOM with gradient checkpointing on")
        res.append(r)
    return _finish(cfg, tracker, res, t0)


def _finish(cfg, tracker, res, t0):
    passed = all(r["passed"] for r in res if not r["gate"].endswith("(no ckpt)"))
    speed = next((r for r in reversed(res) if r["gate"] == "G-mem/speed"), None)
    out = dict(passed=passed, gates=res, seconds=time.time() - t0,
               setting=dict(grad_ckpt=cfg.grad_ckpt, micro_blocks=cfg.micro_blocks),
               projection=speed["metrics"] if speed else None)
    tracker.gates(res)
    for r in res:
        for k, v in r["metrics"].items():
            if isinstance(v, (int, float, bool)):
                tracker.summary({f"gate_metrics/{r['gate']}/{k}": v})
    tracker.summary({"gates/seconds": out["seconds"], "gates/grad_ckpt": cfg.grad_ckpt,
                     "gates/micro_blocks": cfg.micro_blocks})
    Path(cfg.run_dir).mkdir(parents=True, exist_ok=True)
    (Path(cfg.run_dir) / "gates.json").write_text(json.dumps(out, indent=1, default=str))
    print(f"[gates] {'ALL PASSED' if passed else 'FAILED'} in {out['seconds']:.0f} s", flush=True)
    return out
