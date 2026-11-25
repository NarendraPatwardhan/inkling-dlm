"""LoopCD (loopcd.py) on the tiny model: off and w=0 are bit-identical to the current sampler; Eq. 2/3/4 match
references built from the upstream forward (fewer passes, a hook on the first coda layer); reduced depth equals the
upstream model run with fewer passes."""
import contextlib

import pytest
import torch

from inkling_dlm.config import load_config
from inkling_dlm.constants import END_MSG_ID, EOS_ID, MASK_ID, S
from inkling_dlm.evaluate import heldout_ce, heldout_guided
from inkling_dlm.loopcd import Spec, adaptive_w, coda_logits, core_states, guided_logits, spec_of, sweep_of
from inkling_dlm.sampler import generate
from inkling_dlm.sanity import prompt_ids
from inkling_dlm.tower import canvas_logits, parts, tower_fill, view_from_inference_cache


@pytest.fixture()
def lora_on(fresh_lora):
    """Non-zero adapters so the trained path (not the parent) is what is compared."""
    g = torch.Generator().manual_seed(7)
    with torch.no_grad():
        for p in fresh_lora.lora:
            p.add_(torch.randn(p.shape, generator=g) * 0.05)
    return fresh_lora


def _canvas(bundle):
    pm = bundle.pm
    ids = prompt_ids(bundle.tok, "Write a Python function that adds two numbers.")
    cache = tower_fill(pm, torch.tensor([ids]), record_past=True)
    view = view_from_inference_cache(pm, cache, len(ids))
    x = torch.full((1, S + 1), MASK_ID)
    x[0, 0] = ids[-1]
    x[0, 1 + torch.arange(0, S, 3)] = torch.arange(1000, 1000 + len(range(0, S, 3)))
    return view, x


@contextlib.contextmanager
def passes(text, n):
    old = text.config.loop_num_passes
    text.config.loop_num_passes = n
    try:
        yield
    finally:
        text.config.loop_num_passes = old


@contextlib.contextmanager
def coda_input(text):
    """Capture the state (hidden, block_states) entering the first post-loop layer."""
    seen = []

    def pre(mod, args, kwargs):
        seen.append((args[0], args[1]))

    h = text.layers[text.config.loop_layer_index_end].register_forward_pre_hook(pre, with_kwargs=True)
    try:
        yield seen
    finally:
        h.remove()


def check_default_is_off(cfg):
    sp = spec_of(load_config())
    assert sp == Spec() and sp.plain(3)
    assert spec_of(cfg).mode == "off"


@torch.no_grad()
def check_off_and_w0_logits_are_bit_identical(lora_on):
    pm = lora_on.pm
    lm, text = parts(pm)
    view, x = _canvas(lora_on)
    ref = canvas_logits(pm, view, x)
    _, hr, ctx = core_states(text, view, x)
    assert torch.equal(coda_logits(lm, text, view, hr, ctx), ref)          # our core/coda split == upstream
    for sp in (Spec(), Spec("logits", 0.0), Spec("hidden", 0.0), Spec("logits", 0.0, True), Spec("off", passes=3)):
        assert torch.equal(guided_logits(pm, view, x, sp), ref), sp.name
    for ad in (False, True):
        assert torch.equal(guided_logits(pm, view, x, Spec(), adapters=ad), canvas_logits(pm, view, x, adapters=ad))


@torch.no_grad()
def check_w0_sampler_is_bit_identical(lora_on):
    pm = lora_on.pm
    ids = prompt_ids(lora_on.tok, "Write a Python function that reverses a string.")
    t0 = []
    base = generate(pm, ids, max_blocks=2, steps_per_block=8, trace=t0)
    for sp in (Spec("logits", 0.0), Spec("hidden", 0.0), Spec("logits", 0.0, True)):
        t1 = []
        assert generate(pm, ids, max_blocks=2, steps_per_block=8, trace=t1, loopcd=sp) == base and t1 == t0
    gs = [torch.Generator().manual_seed(3) for _ in range(2)]
    a = generate(pm, ids, max_blocks=1, steps_per_block=4, temperature=1.0, generator=gs[0])
    b = generate(pm, ids, max_blocks=1, steps_per_block=4, temperature=1.0, generator=gs[1], loopcd=Spec("logits", 0.0))
    assert a == b


@torch.no_grad()
def check_states_are_upstream_pass1_and_pass3(lora_on):
    """core_states' h_1 / h_R are exactly what the upstream forward hands the coda with 1 / 3 core passes."""
    pm = lora_on.pm
    text = parts(pm)[1]
    view, x = _canvas(lora_on)
    h1, hr, _ = core_states(text, view, x)
    with coda_input(text) as seen:
        canvas_logits(pm, view, x)
        with passes(text, 1):
            canvas_logits(pm, view, x)
    (u3h, u3b), (u1h, u1b) = seen
    assert torch.equal(hr[0], u3h) and torch.equal(hr[1], u3b)
    assert torch.equal(h1[0], u1h) and torch.equal(h1[1], u1b)
    assert not torch.equal(h1[0], hr[0])


@torch.no_grad()
def check_logits_formula_eq3_and_probability_form(lora_on):
    pm = lora_on.pm
    text = parts(pm)[1]
    view, x = _canvas(lora_on)
    z3 = canvas_logits(pm, view, x)
    with passes(text, 1):
        z1 = canvas_logits(pm, view, x)                  # lm_head(coda(h_1)), softcap included
    for w in (0.3, 1.0):
        zg = guided_logits(pm, view, x, Spec("logits", w))
        assert torch.allclose(zg, z3 + w * (z3 - z1), atol=1e-5, rtol=0)
        # App. A.3: softmax(z') == p3 (p3/p1)^w normalised
        lp3, lp1 = z3.log_softmax(-1), z1.log_softmax(-1)
        ref = ((1 + w) * lp3 - w * lp1).softmax(-1)
        assert torch.allclose(zg.softmax(-1), ref, atol=1e-6)
    # guidance moves the logits beyond the final pass along z3 - z1, by exactly w
    zg = guided_logits(pm, view, x, Spec("logits", 0.5))
    d = (z3 - z1).flatten()
    assert torch.allclose(((zg - z3).flatten() @ d) / (d @ d), torch.tensor(0.5), atol=1e-5)


@torch.no_grad()
def check_hidden_formula_eq2(lora_on):
    """h' = h_R + w (h_R - h_1) on (hidden, block_states), then one coda pass. The coda is the upstream one: on h_1 /
    h_R it gives exactly the upstream logits with 1 / 3 core passes. (A pre-hook on the first coda layer cannot be the
    reference: the upstream loop appends the unmodified hidden state to block_states outside the layer.)"""
    pm = lora_on.pm
    lm, text = parts(pm)
    view, x = _canvas(lora_on)
    h1, hr, ctx = core_states(text, view, x)
    with passes(text, 1):
        assert torch.equal(coda_logits(lm, text, view, h1, ctx), canvas_logits(pm, view, x))
    w = 0.5
    k = text.config.n_layers_per_attn_res_block
    pre = -(-text.config.loop_layer_index_start // k) + 1    # pre-loop blocks + each pass's first entry (pre_h)
    assert torch.equal(h1[1][..., :pre, :], hr[1][..., :pre, :])        # these cancel in the contrast
    hg = (hr[0] + w * (hr[0] - h1[0]), hr[1] + w * (hr[1] - h1[1]))
    out = guided_logits(pm, view, x, Spec("hidden", w))
    assert torch.equal(out, coda_logits(lm, text, view, hg, ctx))
    assert not torch.allclose(out, canvas_logits(pm, view, x), atol=1e-4)
    assert not torch.allclose(out, guided_logits(pm, view, x, Spec("logits", w)), atol=1e-4)   # nonlinear coda


@torch.no_grad()
def check_adaptive_gate_eq4(lora_on):
    z = torch.tensor([[50.0, 0.0, 0.0], [0.0, 0.0, 0.0], [2.0, 2.0, -9.0], [1.0, 0.0, 0.0]])
    w = adaptive_w(z, 1.5)[:, 0]
    p = z.softmax(-1).sort(-1, descending=True).values
    assert torch.allclose(w, 1.5 * (1 - (p[:, 0] - p[:, 1])))
    assert w[0] < 1e-6 and abs(w[1] - 1.5) < 1e-6 and abs(w[2] - 1.5) < 1e-6 and 0 < w[3] < 1.5
    pm = lora_on.pm
    text = parts(pm)[1]
    view, x = _canvas(lora_on)
    z3 = canvas_logits(pm, view, x)
    with passes(text, 1):
        z1 = canvas_logits(pm, view, x)
    top = z3.softmax(-1).topk(2, -1).values
    wt = 1.0 * (1 - (top[..., 0] - top[..., 1]))[..., None]       # per token
    zg = guided_logits(pm, view, x, Spec("logits", 1.0, adaptive=True))
    assert torch.allclose(zg, z3 + wt * (z3 - z1), atol=1e-5, rtol=0)
    assert wt.std() > 0 and (wt <= 1).all() and (wt >= 0).all()


@torch.no_grad()
def check_reduced_depth_is_upstream_with_fewer_passes(lora_on):
    pm = lora_on.pm
    text = parts(pm)[1]
    view, x = _canvas(lora_on)
    with passes(text, 2):
        z2 = canvas_logits(pm, view, x)
    with passes(text, 1):
        z1 = canvas_logits(pm, view, x)
    assert torch.equal(guided_logits(pm, view, x, Spec("off", passes=2)), z2)
    assert torch.allclose(guided_logits(pm, view, x, Spec("logits", 0.3, passes=2)), z2 + 0.3 * (z2 - z1), atol=1e-5)


@torch.no_grad()
def check_guided_choice_and_confidence_drive_unmasking(lora_on):
    """With w > 0 the sampler's first commit is the argmax/confidence of softmax(z'), not of the plain logits."""
    pm = lora_on.pm
    ids = prompt_ids(lora_on.tok, "Write a Python function that reverses a string.")
    sp = Spec("logits", 1.0)
    cache = tower_fill(pm, torch.tensor([ids]), record_past=True)
    view = view_from_inference_cache(pm, cache, len(ids))
    x = torch.full((1, S + 1), MASK_ID)
    x[0, 0] = ids[-1]
    p = guided_logits(pm, view, x, sp)[0].softmax(-1)
    conf, pred = p.max(-1)
    is_end = (pred == EOS_ID) | (pred == END_MSG_ID)
    conf = torch.where(is_end & (torch.arange(S) != 0), torch.full_like(conf, -0.5), conf)   # sampler's end rule
    plain = canvas_logits(pm, view, x)[0].softmax(-1).max(-1)
    trace = []
    out = generate(pm, ids, max_blocks=1, steps_per_block=S, trace=trace, loopcd=sp)
    j = trace[0][2][0]
    assert j == int(conf.argmax()) and out[j] == int(pred[j])
    assert not torch.equal(plain.indices, pred) or not torch.allclose(plain.values, p.max(-1).values)


def check_spec_validation_and_sweep(cfg):
    for bad in (dict(mode="hidden", w=0.5, adaptive=True), dict(mode="logits", w=-0.1),
                dict(mode="logits", w=0.3, passes=1), dict(mode="contrast")):
        with pytest.raises(ValueError):
            Spec(**bad)
    grid = sweep_of(load_config())
    names = [s.name for s in grid]
    assert len(names) == len(set(names)) and names[0] == "off"
    assert {s.w for s in grid if s.mode == "logits" and not s.adaptive and s.passes is None} >= {0.2, 0.3, 0.5}
    assert {s.w for s in grid if s.adaptive} == {0.25, 0.5, 1.0, 1.5}
    assert any(s.mode == "hidden" for s in grid) and any(s.passes == 2 for s in grid)
    c = load_config(overrides=["loopcd=logits", "loopcd_w=0.3", "loopcd_adaptive=true", "loopcd_passes=2"])
    assert spec_of(c) == Spec("logits", 0.3, True, 2)
    assert spec_of(load_config(overrides=["loopcd_passes=none"])).passes is None


@torch.no_grad()
def check_heldout_guided_off_matches_heldout_ce(cfg, bundle, groups):
    a = heldout_ce(bundle.pm, groups[:1], cfg)
    b = heldout_guided(bundle.pm, groups[:1], cfg, Spec())
    assert abs(a["ce"] - b["ce"]) < 1e-4 and 0 <= b["top1"] <= 1
    c = heldout_guided(bundle.pm, groups[:1], cfg, Spec("logits", 0.5))
    assert c["ce"] != b["ce"]
