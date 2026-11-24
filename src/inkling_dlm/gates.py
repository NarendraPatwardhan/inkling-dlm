"""Pre-training gates. Every gate returns {gate, passed, detail, metrics}; `run_gates` stops at the first hard
failure that makes the rest meaningless, logs everything to W&B and writes gates.json (the chosen GRAD_CKPT /
MICRO_BLOCKS setting and the cost projection) next to the run's checkpoints."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from . import quant, unsloth_ops
from .constants import S
from .cost import projection
from .data import batch_order, collate, encode_all, eval_batches, lad_rows, n_blocks, selection_index
from .evaluate import tower_states
from .tower import canvas_logits, tower_fill, view_for_batch, view_from_inference_cache
from .train import peak_mem_gb, peak_reserved_gb, sync, train_step
from .unsloth_ops import set_dequant_chunk_mb


def result(name, passed, detail, **metrics):
    tag = "PASS" if passed else "FAIL"
    print(f"[{tag}] {name}: {detail}", flush=True)
    return dict(gate=name, passed=bool(passed), detail=detail, metrics=metrics)


# ------------------------------------------------------------------------------------------------ load & dispatch
def g_hardware(cfg, min_gpu_gb: float = 39.0):
    """Before any download. With the experts packed (15.7 GB) the model needs ~19 GB resident, so a 40 GB A100
    (42.4e9 bytes) qualifies; the BF16-dequantised alternative (~52 GB) would need an 80 GB card."""
    if cfg.tiny or not cfg.device.startswith("cuda"):
        return result("G-hardware", True, f"device {cfg.device} (tiny verification)", skipped=True)
    if not torch.cuda.is_available():
        return result("G-hardware", False, "no CUDA device")
    p = torch.cuda.get_device_properties(0)
    gb = p.total_memory / 1e9
    cap = torch.cuda.get_device_capability()
    need = min_gpu_gb if cfg.experts_impl == quant.IMPL else 70.0
    alloc = os.environ.get("PYTORCH_ALLOC_CONF") or os.environ.get("PYTORCH_CUDA_ALLOC_CONF") or ""
    return result("G-hardware", gb >= need and cap >= (8, 0),
                  f"{p.name}, {gb:.1f} GB, capability {cap} (need >= {need:.0f} GB, sm80+); allocator {alloc!r}",
                  gpu_gb=gb, gpu=p.name, alloc_conf=alloc)


def _message_parts(bundle, lm):
    """(message-gate weights, those not 16/32-bit or packed, message residuals under Unsloth's checkpoint)."""
    gates_w = [(n, p) for n, p in lm.named_parameters() if "message_gate" in n]
    bad = [n for n, p in gates_w if quant.is_new_packed_expert_param(p)
           or p.dtype not in (torch.bfloat16, torch.float16, torch.float32)]
    return gates_w, bad, unsloth_ops.message_inputs_checkpointed(lm)


def g_load(bundle, cfg):
    if cfg.tiny:
        lm = bundle.pm.base_model.model
        gates_w, bad, n_msg = _message_parts(bundle, lm)
        return result("G-load", bool(gates_w) and not bad and n_msg == bundle.message_ckpt > 0,
                      f"tiny random model (no checkpoint to load); message gates 16-bit {len(gates_w) - len(bad)}/"
                      f"{len(gates_w)}; message residuals under Unsloth's checkpoint {n_msg}", skipped=True,
                      message_residuals_checkpointed=n_msg)
    info = bundle.load_info or {}
    missing = list(info.get("missing_keys", []))
    unexpected = [k for k in info.get("unexpected_keys", []) if "visual" not in k and "mtp" not in k]
    mism = list(info.get("mismatched_keys", []))
    lm = bundle.pm.base_model.model
    dtypes = sorted({str(p.dtype) for p in lm.parameters()})
    packed = quant.packed_bytes(lm)
    n_mod = len(quant.experts_modules(lm))
    ok = not missing and not unexpected and not mism
    if cfg.experts_impl == quant.IMPL:
        ok &= info.get("packed_modules") == n_mod and set(packed) == {"nvfp4", "mxfp8"}
    # Unsloth guide, "Correct loss": the loop message gates must stay out of 4-bit (quantised, the first losses are
    # 3.4-3.9 instead of 0.4-1.2). Only routed experts are packed here; check every message-gate weight is 16/32-bit.
    gates_w, bad_gates, n_msg = _message_parts(bundle, lm)
    ok &= bool(gates_w) and not bad_gates and n_msg == bundle.message_ckpt > 0
    mem = torch.cuda.memory_allocated() / 1e9 if cfg.device.startswith("cuda") else 0.0
    return result("G-load", ok,
                  f"missing {len(missing)}, unexpected non-vision/MTP {len(unexpected)}, mismatched {len(mism)}; "
                  f"dtypes {dtypes}; packed experts {', '.join(f'{k} {v / 1e9:.2f} GB' for k, v in packed.items())} "
                  f"in {info.get('packed_modules')}/{n_mod} modules; message gates 16-bit "
                  f"{len(gates_w) - len(bad_gates)}/{len(gates_w)}; message residuals under Unsloth's checkpoint "
                  f"{n_msg}; allocated after load {mem:.2f} GB",
                  missing=missing[:20], unexpected=unexpected[:20], allocated_gb=mem, message_gates=len(gates_w),
                  message_gates_not_16bit=bad_gates[:20], message_residuals_checkpointed=n_msg,
                  **{f"packed_{k}_gb": v / 1e9 for k, v in packed.items()})


def _layer(name: str) -> int:
    return int(name.split("layers.")[1].split(".")[0])


def _eager_moe(x, gate_up, down, act, idx, w):
    """transformers' eager expert loop (Inkling2LoopExperts.forward) on given dense weights."""
    out = torch.zeros_like(x, dtype=torch.float32)
    for e in range(gate_up.shape[0]):
        tok, pos = torch.where(idx == e)
        if tok.numel() == 0:
            continue
        g, u = torch.nn.functional.linear(x[tok], gate_up[e]).chunk(2, dim=-1)
        h = torch.nn.functional.linear(act(g, u), down[e]) * w[tok, pos, None]
        out.index_add_(0, tok, h.to(out.dtype))
    return out.to(x.dtype)


def g_dispatch(bundle, cfg):
    """Packed experts: (1) Unsloth's decode of the packed bytes equals transformers' own BF16 dequantisation of the
    raw checkpoint tensors, bitwise, on NVFP4 and MXFP8 layers (the 4-bit model IS the BF16-loaded parent);
    (2) the packed grouped forward and its input gradient agree with the eager expert loop on those dense weights.
    BF16 path: grouped_mm is selected and agrees with the eager loop."""
    lm = bundle.pm.base_model.model
    impl = getattr(lm.config, "_experts_implementation", None)
    ok, parts, metrics = impl == cfg.experts_impl, [f"experts_implementation={impl}"], {}
    if torch.cuda.is_available() and cfg.device.startswith("cuda"):
        cap = torch.cuda.get_device_capability()
        has = hasattr(torch.nn.functional, "grouped_mm") or hasattr(torch, "_grouped_mm")
        ok &= cap >= (8, 0) and has
        parts.append(f"capability {cap}, torch {torch.__version__}, grouped_mm available {has}")
    mods = quant.experts_modules(lm)
    E, k = lm.config.n_routed_experts, lm.config.num_experts_per_tok
    g = torch.Generator(device="cpu").manual_seed(0)
    if impl == quant.IMPL:
        by_layer = {_layer(n): m for n, m in mods}
        nv = sorted(L for L, m in by_layer.items() if m.down_proj.packed_format == "nvfp4")
        mx = sorted(L for L, m in by_layer.items() if m.down_proj.packed_format == "mxfp8")
        loop0 = lm.config.loop_layer_index_start
        pick = sorted(set(nv[:1] + nv[-1:] + mx[:1] + mx[-1:] + [loop0]) & set(by_layer))
        mism, total, worst = 0, 0, 0.0
        fwd_rel, dx_rel = 0.0, 0.0
        tf_diff, tf_rel = 0, 0.0           # transformers' from_pretrained casts amax to the model dtype first
        for L in pick:
            mod = by_layer[L]
            dev = mod.down_proj.device
            dt = mod.down_proj.packed_dtype
            raw = quant.raw_layer(bundle.expert_source, L, dev)
            dense = {}
            for prefix, pname, inter in quant.STACKS:
                ref = quant.reference_dequant(raw, prefix, inter, dt)
                got = getattr(mod, pname).dequantize(dt)
                d = ref != got
                mism += int(d.sum())
                total += ref.numel()
                worst = max(worst, float(((ref.float() - got.float()).abs() / ref.float().abs().clamp(min=1e-30))
                                         [d].max()) if d.any() else 0.0)
                dense[pname] = ref
                if quant.is_nvfp4(raw, prefix) and dt != torch.float32:
                    tf = quant.reference_dequant(quant.as_from_pretrained(raw, prefix, dt), prefix, inter, dt)
                    tf_diff += int((tf != ref).sum())
                    tf_rel = max(tf_rel, float(((tf.float() - ref.float()).abs()
                                                / ref.float().abs().clamp(min=1e-30)).max()))
                    del tf
                del got
            del raw
            x = torch.randn(64, lm.config.hidden_size, generator=g).to(dev, dt).requires_grad_(True)
            idx = torch.stack([torch.randperm(E, generator=g)[:k] for _ in range(64)]).to(dev)
            w = torch.rand(64, k, generator=g).to(dev, dt)
            y = mod(x, idx, w)
            xr = x.detach().clone().requires_grad_(True)
            yr = _eager_moe(xr, dense["gate_up_proj"], dense["down_proj"], mod.act_fn, idx, w)
            gy = torch.randn(y.shape, generator=g).to(dev, dt)
            y.backward(gy)
            yr.backward(gy)
            rel = lambda a, b: ((a.float() - b.float()).norm() / b.float().norm().clamp(min=1e-12)).item()
            fwd_rel, dx_rel = max(fwd_rel, rel(y, yr)), max(dx_rel, rel(x.grad, xr.grad))
            del dense, y, yr
        # fp32 (tiny) decodes in a different order than transformers (e2m1*(s*g) vs (e2m1*s)*g): <= 1 fp32 ulp.
        exact = mism == 0 if dt != torch.float32 else worst <= 2.4e-7
        tol = 2e-2 if dt == torch.bfloat16 else 1e-5
        ok &= exact and fwd_rel < tol and dx_rel < tol
        parts.append(f"decode vs transformers dequant on layers {pick}: {mism}/{total} values differ "
                     f"(max rel {worst:.1e}); packed vs eager loop rel diff fwd {fwd_rel:.2e}, dX {dx_rel:.2e} "
                     f"(tol {tol}); for reference, transformers' BF16 from_pretrained (amax rounded to BF16) "
                     f"differs in {tf_diff}/{total} values, max rel {tf_rel:.1e}")
        metrics = dict(decode_mismatches=mism, decode_values=total, decode_max_rel=worst, rel_diff=fwd_rel,
                       dx_rel_diff=dx_rel, layers=pick, bf16_load_mismatches=tf_diff, bf16_load_max_rel=tf_rel)
        return result("G-dispatch", ok, "; ".join(parts), **metrics)
    # BF16 path: grouped_mm vs eager on one MoE block
    moe = mods[0][1]
    x = torch.randn(64, lm.config.hidden_size, device=moe.gate_up_proj.device, dtype=moe.gate_up_proj.dtype)
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
    the prompt is not silently dropped under checkpointing, and Unsloth's message-residual recompute is exact)."""
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
        with unsloth_ops.count_message_math() as msg:
            grads.append(_grads(pm, b, cfg, limit=8))
        if on:
            msg_calls = msg.calls
        else:
            msg_off = msg.calls
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
    # Unsloth's message-residual recompute must engage under checkpointing (forward + backward recompute) and only there
    msg_ok = msg_off == 0 and msg_calls > 0
    return result("G-ckpt", same_keys and rel < cfg.ckpt_grad_rel_max and msg_ok,
                  f"relative grad difference {rel:.2e} over {len(keys)} tensors; checkpointed message-residual calls "
                  f"{msg_calls} with checkpointing, {msg_off} without", rel=rel, message_calls=msg_calls)


# ------------------------------------------------------------------------------------------------ speed & cost
def speed_batches(cfg, tok, n):
    """n updates shaped like training: length-bucketed groups (data.batch_order, as run_training builds them) from the
    head of the selection, so canvases per update and tower width follow the training distribution."""
    G = cfg.examples_per_update
    data = encode_all(tok, lad_rows(cfg, "train", selection_index()[: max(n, 50) * G]), cfg.maxlen)
    order = batch_order([len(p) + len(a) for p, a, _ in data], cfg.seed, G)
    return [[data[i] for i in grp] for grp in order[:n]]


def peak_batch(cfg, tok, pool: int | None = None):
    """The worst update for memory: the examples_per_update rows with the most answer blocks (then the longest) among
    the first `pool` selected rows, i.e. the most canvases per micro-batch and the widest tower."""
    pool = pool or (400 if cfg.tiny else 4000)
    data = encode_all(tok, lad_rows(cfg, "train", selection_index()[:pool]), cfg.maxlen)
    key = sorted(range(len(data)), key=lambda i: (n_blocks(len(data[i][1]), data[i][2], cfg.block),
                                                   len(data[i][0]) + len(data[i][1])), reverse=True)
    return [data[i] for i in key[: cfg.examples_per_update]]


def mem_limit_gb(cfg) -> float:
    """cfg.mem_limit_gb, capped at 92% of the device (allocator caching, CUDA context and fragmentation live above
    max_memory_allocated)."""
    if cfg.device.startswith("cuda") and torch.cuda.is_available():
        return min(cfg.mem_limit_gb, 0.92 * torch.cuda.get_device_properties(0).total_memory / 1e9)
    return cfg.mem_limit_gb


def reserved_limit_gb(cfg) -> float:
    if cfg.device.startswith("cuda") and torch.cuda.is_available():
        return 0.95 * torch.cuda.get_device_properties(0).total_memory / 1e9
    return float("inf")


def setting(cfg) -> dict:
    return dict(grad_ckpt=cfg.grad_ckpt, micro_blocks=cfg.micro_blocks, dequant_chunk_mb=cfg.dequant_chunk_mb)


def apply_setting(cfg, bundle, grad_ckpt, micro_blocks, dequant_chunk_mb=None):
    cfg.grad_ckpt, cfg.micro_blocks, cfg.dequant_chunk_mb = bool(grad_ckpt), int(micro_blocks), dequant_chunk_mb
    lm = bundle.pm.base_model.model
    if cfg.grad_ckpt:
        lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    else:
        lm.gradient_checkpointing_disable()
    set_dequant_chunk_mb(dequant_chunk_mb)


def g_speed(bundle, cfg, batches, tracker=None, peak=None, name="G-mem/speed", check_cost=True, warm=5):
    """Real-data updates in the current setting, each a full update (forward + backward + AdamW step on the
    trainable tensors, so optimizer state is in the memory peak); project the full run. `peak` (peak_batch) runs first:
    the worst update for memory, so the measured peak covers every training update at this micro-batch. FAIL if peak
    allocated memory exceeds mem_limit_gb, peak reserved exceeds 95 % of the card, or (check_cost) the projection exceeds
    cfg.cu_limit. Run last: it moves the trainable tensors."""
    from .train import make_optimizer
    pm, dev = bundle.pm, cfg.device
    pm.train()
    opt, _ = make_optimizer(cfg, bundle)
    if dev.startswith("cuda"):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    times, toks, slots = [], [], []
    peak_blocks = 0
    for i, ex in enumerate(([peak] if peak else []) + list(batches)):
        b = collate(ex, i, cfg.seed, dev, cfg.block)
        sync(dev)
        t0 = time.time()
        out = train_step(pm, b, cfg.micro_blocks)
        torch.nn.utils.clip_grad_norm_(bundle.lora + bundle.fwd, cfg.clip)
        opt.step()
        opt.zero_grad(set_to_none=True)
        sync(dev)
        if peak and i == 0:
            peak_blocks = out["blocks"]
            continue
        times.append(time.time() - t0)
        toks.append(int(b["lens"].sum()))
        slots.append(out["blocks"] * (S + 1))
        if tracker:
            tracker.log({"gate_speed/s_per_update": times[-1], "gate_speed/loss": out["loss"],
                         "gate_speed/blocks": out["blocks"], "gate_speed/tower_tokens": toks[-1],
                         "gate_speed/tokens_per_s": toks[-1] / times[-1],
                         "gate_speed/peak_mem_gb": peak_mem_gb(dev), "gate_speed/peak_reserved_gb":
                         peak_reserved_gb(dev), "gate_speed/micro_blocks": cfg.micro_blocks,
                         "gate_speed/grad_ckpt": int(cfg.grad_ckpt)})
    del opt
    warm = min(warm, len(times) - 1)
    s = float(np.median(times[warm:]))
    tps = float(np.sum(toks[warm:]) / np.sum(times[warm:]))
    cps = float(np.sum(slots[warm:]) / np.sum(times[warm:]))
    mem, res = peak_mem_gb(dev), peak_reserved_gb(dev)
    lim, rlim = mem_limit_gb(cfg), reserved_limit_gb(cfg)
    p = projection(s, cfg.updates, cfg.cu_per_hour)
    fits = mem < lim and res < rlim
    ok = fits and (p["train_cu"] <= cfg.cu_limit or not check_cost)
    st = setting(cfg)
    return result(name, ok, f"grad_ckpt={cfg.grad_ckpt} micro={cfg.micro_blocks} dequant_chunk_mb="
                  f"{cfg.dequant_chunk_mb}: median {s:.2f} s/update, {tps:.0f} tower tokens/s, {cps:.0f} canvas "
                  f"slots/s, peak {mem:.1f} GB allocated (limit {lim:.1f}), {res:.1f} GB reserved (limit {rlim:.1f}), "
                  f"peak update {peak_blocks} canvases -> projected {p['train_hours']:.1f} h = {p['train_cu']:.0f} "
                  f"CU at {cfg.cu_per_hour} CU/h (limit {cfg.cu_limit:.0f})", peak_mem_gb=mem, peak_reserved_gb=res,
                  mem_limit_gb=lim, reserved_limit_gb=rlim, fits=fits, peak_update_blocks=peak_blocks,
                  tokens_per_s=tps, canvas_slots_per_s=cps, **st, **p)


def choose_setting(bundle, cfg, probe, peak, tracker=None):
    """Probe every (grad_ckpt, micro_blocks) in cfg.gate_settings on the peak update plus `probe` timed updates and
    keep the fastest one that fits; if none fits, the same ladder again with the expert decode bounded to
    cfg.dequant_fallback_mb. Returns (probe results, chosen setting or None)."""
    res, fit = [], []
    ladders = [[(ck, m, cfg.dequant_chunk_mb) for ck, m in cfg.gate_settings]]
    if cfg.dequant_fallback_mb:
        ladders.append([(ck, m, cfg.dequant_fallback_mb) for ck, m in cfg.gate_settings])
    for ladder in ladders:
        for ck, micro, chunk in ladder:
            apply_setting(cfg, bundle, ck, micro, chunk)
            tag = f"G-mem/speed (probe ckpt={ck} micro={micro} chunk={chunk})"
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            try:
                r = g_speed(bundle, cfg, probe, tracker, peak=peak, name=tag, check_cost=False, warm=1)
            except torch.OutOfMemoryError:
                bundle.pm.zero_grad(set_to_none=True)
                r = result(tag, False, f"OOM with grad_ckpt={ck} micro={micro} dequant_chunk_mb={chunk}",
                           fits=False, grad_ckpt=ck, micro_blocks=micro, dequant_chunk_mb=chunk)
            res.append(r)
            if r["metrics"].get("fits"):
                fit.append(r)
        if fit:
            break
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if not fit:
        return res, None
    best = min(fit, key=lambda r: r["metrics"]["s_per_update"])["metrics"]
    return res, (best["grad_ckpt"], best["micro_blocks"], best["dequant_chunk_mb"])


# ------------------------------------------------------------------------------------------------ driver
def run_gates(cfg, bundle, tracker, groups=None, speed=None, pre=None, peak=None) -> dict:
    t0 = time.time()
    res = list(pre or []) + [g_load(bundle, cfg)]
    if not res[-1]["passed"]:
        return _finish(cfg, tracker, res, t0)
    res.append(g_dispatch(bundle, cfg))
    groups = groups or eval_batches(cfg, bundle.tok, 2)
    for fn in (g_exact, g_readonly, g_grad, g_ckpt):
        res.append(fn(bundle, cfg, groups))
    if all(r["passed"] for r in res):
        speed = speed if speed is not None else speed_batches(cfg, bundle.tok, cfg.gate_updates + cfg.probe_updates)
        peak = peak if peak is not None else peak_batch(cfg, bundle.tok)
        probe, timed = speed[:cfg.probe_updates], speed[cfg.probe_updates:] or speed
        probes, best = choose_setting(bundle, cfg, probe, peak, tracker)
        res += probes
        if best is None:
            res.append(result("G-mem/speed", False, "no setting in gate_settings fits the memory limits"))
        else:
            apply_setting(cfg, bundle, *best)
            print(f"[gates] fastest fitting setting: grad_ckpt={best[0]} micro={best[1]} dequant_chunk_mb={best[2]}",
                  flush=True)
            try:
                res.append(g_speed(bundle, cfg, timed, tracker, peak=peak))
            except torch.OutOfMemoryError:
                bundle.pm.zero_grad(set_to_none=True)
                res.append(result("G-mem/speed", False, f"OOM in the full run of the chosen setting {best}"))
    return _finish(cfg, tracker, res, t0)


def _finish(cfg, tracker, res, t0):
    passed = all(r["passed"] for r in res if not r["gate"].startswith("G-mem/speed ("))
    speed = next((r for r in reversed(res) if r["gate"] == "G-mem/speed"), None)
    out = dict(passed=passed, gates=res, seconds=time.time() - t0,
               setting=setting(cfg),
               projection=speed["metrics"] if speed else None)
    tracker.gates(res)
    for r in res:
        for k, v in r["metrics"].items():
            if isinstance(v, (int, float, bool)):
                tracker.summary({f"gate_metrics/{r['gate']}/{k}": v})
    tracker.summary({"gates/seconds": out["seconds"], "gates/grad_ckpt": cfg.grad_ckpt,
                     "gates/micro_blocks": cfg.micro_blocks, "gates/dequant_chunk_mb": cfg.dequant_chunk_mb or 0})
    Path(cfg.run_dir).mkdir(parents=True, exist_ok=True)
    (Path(cfg.run_dir) / "gates.json").write_text(json.dumps(out, indent=1, default=str))
    print(f"[gates] {'ALL PASSED' if passed else 'FAILED'} in {out['seconds']:.0f} s", flush=True)
    return out
