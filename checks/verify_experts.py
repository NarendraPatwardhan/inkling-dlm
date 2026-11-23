"""Packed 4-bit experts (Unsloth's NVFP4/MXFP8 stacks):
* decode equals transformers' own dequantisation on slices of the real checkpoint (HTTP range reads, ~40 MB, cached);
* the packed forward and input gradient equal transformers' grouped_mm on the decoded weights, bitwise;
* a checkpoint in the release's layout loads with no BF16 expert ever allocated and matches transformers' load;
* G-dispatch passes on the tiny model."""
import json
import struct
import urllib.request

import pytest
import torch

from inkling_dlm import quant
from inkling_dlm.constants import MODEL_REPO, MODEL_REVISION
from inkling_dlm.gates import g_dispatch
from inkling_dlm.model import tiny_text_config

from conftest import VERIFY

URL = f"https://huggingface.co/{MODEL_REPO}/resolve/{MODEL_REVISION}/model-00001-of-00001.safetensors"
SLICES = {0: 3, 6: 64, 19: 127, 20: 0, 23: 77}       # layer -> expert: NVFP4 first/loop/last, MXFP8 first/last
ROWS = 256                                            # leading rows of that expert (even: gate/up pairs)
DT = {"U8": torch.uint8, "F8_E4M3": torch.float8_e4m3fn, "F32": torch.float32}


def _get(a, b):
    return urllib.request.urlopen(urllib.request.Request(URL, headers={"Range": f"bytes={a}-{b}"}), timeout=60).read()


def _slices():
    path = VERIFY / "expert-slices.pt"
    if path.exists():
        return torch.load(path)
    n = struct.unpack("<Q", _get(0, 7))[0]
    head = json.loads(_get(8, 8 + n - 1))
    out = {}
    for layer, e in SLICES.items():
        raw = {}
        for p, _, _ in quant.STACKS:
            for s in ("", "_scale", "_amax", "_outer_scale_denominator"):
                m = head.get(quant.checkpoint_key(layer, f"{p}_weight{s}"))
                if m is None:
                    continue
                dt, shape, (a, b) = DT[m["dtype"]], m["shape"], m["data_offsets"]
                size = torch.empty(0, dtype=dt).element_size()
                if len(shape) == 3:
                    row = shape[2] * size
                    a = 8 + n + a + (e * shape[1]) * row
                    t = torch.frombuffer(bytearray(_get(a, a + ROWS * row - 1)), dtype=torch.uint8)
                    t = t.view(dt).reshape(1, ROWS, shape[2])
                else:
                    t = torch.frombuffer(bytearray(_get(8 + n + a, 8 + n + b - 1)), dtype=torch.uint8).view(dt)
                    t = t[e:e + 1] if shape[0] > 1 else t
                raw[f"{p}_weight{s}"] = t.clone()
        out[layer] = raw
    torch.save(out, path)
    return out


@pytest.fixture(scope="module")
def slices():
    try:
        return _slices()
    except OSError as e:                     # no network: the GPU gate G-dispatch repeats this on whole layers
        pytest.skip(f"checkpoint slices unavailable: {e}")


def check_decode_matches_transformers_on_real_checkpoint(slices):
    formats = set()
    for layer, raw in slices.items():
        for p, _, inter in quant.STACKS:
            packed = quant.pack_stack(raw, p, inter)
            formats.add(packed.packed_format)
            ref = quant.reference_dequant(raw, p, inter, torch.bfloat16)
            got = packed.dequantize(torch.bfloat16)
            assert got.shape == ref.shape == (1, ROWS, raw[f"{p}_weight"].shape[-1] * (2 if quant.is_nvfp4(raw, p) else 1))
            assert torch.equal(got.view(torch.int16), ref.view(torch.int16)), (layer, p)
            r32, g32 = (quant.reference_dequant(raw, p, inter, torch.float32),
                        packed.dequantize(torch.float32))
            assert ((r32 - g32).abs() <= r32.abs() * 2.4e-7).all(), (layer, p)      # <= 1 fp32 ulp
    assert formats == {"nvfp4", "mxfp8"}


def check_from_pretrained_bf16_rounds_the_expert_scale(slices):
    """What transformers' BF16 load actually holds: amax rounded to BF16 first. Different from the checkpoint's
    values, by up to ~0.3% per expert, on the real tensors."""
    raw = slices[0]
    ref = quant.reference_dequant(raw, "w13", True, torch.bfloat16)
    tf = quant.reference_dequant(quant.as_from_pretrained(raw, "w13", torch.bfloat16), "w13", True, torch.bfloat16)
    rel = ((tf.float() - ref.float()).abs() / ref.float().abs().clamp(min=1e-30)).max().item()
    assert 0 < rel < 1e-2


def check_interleave_round_trip():
    t = torch.arange(2 * 8 * 3).reshape(2, 8, 3)
    assert torch.equal(quant.deinterleave_rows(quant.interleave_rows(t)), t)
    d = quant.deinterleave_rows(t)
    assert torch.equal(d[:, :4], t[:, 0::2]) and torch.equal(d[:, 4:], t[:, 1::2])


def check_packed_forward_and_dx_equal_grouped_mm(cfg, bundle):
    from transformers.integrations.moe import grouped_mm_experts_forward
    lm = bundle.pm.base_model.model
    E, k = lm.config.n_routed_experts, lm.config.num_experts_per_tok
    g = torch.Generator().manual_seed(1)
    seen = set()
    for _, mod in quant.experts_modules(lm):
        seen.add(mod.down_proj.packed_format)
        x = torch.randn(48, lm.config.hidden_size, generator=g, requires_grad=True)
        idx = torch.stack([torch.randperm(E, generator=g)[:k] for _ in range(48)])
        idx[:, 0] = 0                                       # expert 0 crowded, some experts empty
        w = torch.rand(48, k, generator=g)
        y = mod(x, idx, w)
        dense = type("D", (), {})()
        dense.num_experts, dense.has_gate, dense.has_bias, dense.is_transposed = E, True, False, False
        dense._is_expert_parallel, dense.has_post_expert_norm = False, False
        dense.gate_up_proj = mod.gate_up_proj.dequantize(torch.float32)
        dense.down_proj = mod.down_proj.dequantize(torch.float32)
        dense._apply_gate = mod._apply_gate
        xr = x.detach().clone().requires_grad_(True)
        yr = grouped_mm_experts_forward(dense, xr, idx, w)
        assert torch.equal(y, yr)
        gy = torch.randn(y.shape, generator=g)
        y.backward(gy)
        yr.backward(gy)
        assert torch.equal(x.grad, xr.grad)
        assert not mod.gate_up_proj.requires_grad and mod.gate_up_proj.grad is None
    assert seen == {"nvfp4", "mxfp8"}


def check_dispatch_gate(cfg, bundle):
    r = g_dispatch(bundle, cfg)
    assert r["passed"], r["detail"]
    assert r["metrics"]["dx_rel_diff"] < 1e-5 and r["metrics"]["rel_diff"] < 1e-5


@pytest.fixture(scope="module")
def release_layout(cfg, tmp_path_factory):
    """A tiny checkpoint written in the release's own layout: transformers' reverse mapping for the names, the
    experts replaced by ModelOpt NVFP4 (+ MXFP8 for the last 2 layers) tensors, a modelopt quantization_config."""
    from safetensors.torch import load_file, save_file
    from transformers.models.inkling2_loop.modeling_inkling2_loop import Inkling2LoopForCausalLM
    d = tmp_path_factory.mktemp("release")
    tc = tiny_text_config(cfg)
    tc._attn_implementation = "eager"
    torch.manual_seed(0)
    Inkling2LoopForCausalLM(tc).save_pretrained(d)
    m = Inkling2LoopForCausalLM.from_pretrained(d, dtype=torch.float32)
    sd = {k: v for k, v in load_file(d / "model.safetensors").items() if ".mlp.experts." not in k}
    sd.update(quant.quantize_like_checkpoint(m, 2))
    save_file(sd, d / "model.safetensors", metadata={"format": "pt"})
    c = json.loads((d / "config.json").read_text())
    c["quantization_config"] = {"quant_method": "modelopt", "quant_algo": "NVFP4"}
    (d / "config.json").write_text(json.dumps(c))
    return d


def check_release_layout_loads_packed(release_layout):
    from transformers.models.inkling2_loop.modeling_inkling2_loop import Inkling2LoopForCausalLM
    ids = torch.randint(0, 1000, (2, 40), generator=torch.Generator().manual_seed(0))
    for dt in (torch.float32, torch.bfloat16):
        ref = Inkling2LoopForCausalLM.from_pretrained(release_layout, dtype=dt, attn_implementation="eager",
                                                      experts_implementation="grouped_mm")
        pk, info = quant.load_packed(str(release_layout), None, "cpu", dt)
        assert not info["missing_keys"] and not info["unexpected_keys"] and not info["mismatched_keys"]
        assert info["packed_modules"] == 6 and info["converters_dropped"] == 2
        mods = quant.experts_modules(pk)
        assert [m.down_proj.packed_format for _, m in mods] == ["nvfp4"] * 4 + ["mxfp8"] * 2
        assert not any(p.dtype == dt and p.dim() == 3 and p.shape[0] == 8 for p in pk.parameters())  # no dense stacks
        assert pk.config._experts_implementation == quant.IMPL
        for (name, r), (_, p) in zip(quant.experts_modules(ref), mods):
            layer = int(name.split("layers.")[1].split(".")[0])
            src = quant.raw_layer(str(release_layout) + "/model.safetensors", layer, "cpu")
            for prefix, pname, inter in quant.STACKS:
                # transformers' load == its decode of the dtype-cast sources; ours == its decode of the stored ones
                tf = quant.reference_dequant(quant.as_from_pretrained(src, prefix, dt), prefix, inter, dt)
                assert torch.equal(getattr(r, pname), tf)
                exact = quant.reference_dequant(src, prefix, inter, dt)
                got = getattr(p, pname).dequantize(dt)
                if dt == torch.bfloat16:
                    assert torch.equal(got, exact)
                else:
                    assert ((got - exact).abs() <= exact.abs() * 2.4e-7).all()
        if dt == torch.float32:                 # fp32: the amax cast is lossless, the models agree to rounding
            with torch.no_grad():
                a, b = ref(input_ids=ids).logits, pk(input_ids=ids).logits
            assert (a - b).abs().max() < 1e-5
