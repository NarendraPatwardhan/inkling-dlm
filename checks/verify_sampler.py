"""Sampler end to end on the tiny model, and inference-view == training-view logits for the same prefix."""
import torch

from inkling_dlm.constants import EOS_ID, END_MSG_ID, MASK_ID, S
from inkling_dlm.data import collate
from inkling_dlm.evaluate import tower_states
from inkling_dlm.sampler import generate, per_step
from inkling_dlm.sanity import prompt_ids
from inkling_dlm.tower import canvas_logits, tower_fill, view_for_batch, view_from_inference_cache


def check_per_step_schedule():
    assert per_step(32, 32) == [1] * 32
    assert per_step(32, 16) == [2] * 16
    assert sum(per_step(20, 6)) == 20


def check_generate_runs_and_commits_blocks(bundle):
    ids = prompt_ids(bundle.tok, "Write a Python function that adds two numbers.")
    trace = []
    out = generate(bundle.pm, ids, max_blocks=3, steps_per_block=8, trace=trace)
    assert 1 <= len(out) <= 3 * S and MASK_ID not in out
    ends = [i for i, t in enumerate(out) if t in (EOS_ID, END_MSG_ID)]
    assert not ends or ends[0] == len(out) - 1                 # stops at the first end token
    first_block = [x for x in trace if x[0] == 0]
    assert len(first_block) == 8 and sorted(sum((x[2] for x in first_block), [])) == list(range(S))
    out16 = generate(bundle.pm, ids, max_blocks=2, steps_per_block=16, block=16)
    assert len(out16) <= 32
    g = torch.Generator().manual_seed(0)
    out_t = generate(bundle.pm, ids, max_blocks=1, steps_per_block=4, temperature=1.0, generator=g)
    assert len(out_t) <= S


@torch.no_grad()
def check_inference_view_matches_training_view(cfg, bundle, groups):
    """Same example, same anchor, bidirectional masked canvas, adapters on: the sampler's cache view and the
    training path (record_past + compaction) give the same logits."""
    pm = bundle.pm
    b = collate(groups[0], 9, cfg.seed, cfg.device, cfg.block)
    k, v, s = tower_states(pm, b)
    j = 1
    idx = torch.tensor([j])
    tv = view_for_batch(pm, k, v, s, b, idx)
    lt = canvas_logits(pm, tv, b["canvas"][idx])
    r, a = int(b["rows"][j]), int(b["anchors"][j])
    cache = tower_fill(pm, b["ids"][r:r + 1, :a + 1], record_past=True)
    iv = view_from_inference_cache(pm, cache, a + 1)
    li = canvas_logits(pm, iv, b["canvas"][idx])
    assert (lt - li).abs().max().item() < 1e-4


@torch.no_grad()
def check_commit_extends_tower_like_a_full_pass(cfg, bundle):
    """Prompt tower + committed block (positions n..n+31) == one tower pass over prompt+block."""
    pm = bundle.pm
    ids = prompt_ids(bundle.tok, "Explain recursion briefly.")
    blk = list(range(1000, 1000 + S))
    c1 = tower_fill(pm, torch.tensor([ids]), record_past=True)
    tower_fill(pm, torch.tensor([blk]), cache=c1, position_ids=torch.arange(len(ids), len(ids) + S)[None])
    c2 = tower_fill(pm, torch.tensor([ids + blk]), record_past=True)
    n = len(ids) + S
    v1, v2 = view_from_inference_cache(pm, c1, n), view_from_inference_cache(pm, c2, n)
    x = torch.full((1, S + 1), MASK_ID)
    x[0, 0] = blk[-1]
    assert (canvas_logits(pm, v1, x) - canvas_logits(pm, v2, x)).abs().max().item() < 1e-4


@torch.no_grad()
def check_plain_sliding_cache_would_break_the_anchor_slot(cfg, bundle, groups):
    """Documents why the sampler records the past: with a plain sliding cache and context > window, slot 0 loses
    key n-(window-1) and drifts; with record_past it is exact."""
    pm = bundle.pm
    lm = pm.base_model.model
    b = collate(groups[0], 0, cfg.seed, cfg.device, cfg.block)
    r, a = 0, int(b["anchors"][b["rows"] == 0][-1])
    assert a + 1 > lm.config.sliding_window_size
    ref = lm(input_ids=b["ids"][:1, :a + S + 1]).logits[:, a:a + S]
    errs = {}
    for rec in (False, True):
        c = tower_fill(pm, b["ids"][:1, :a + 1], record_past=rec)
        v = view_from_inference_cache(pm, c, a + 1, causal=True)
        errs[rec] = (canvas_logits(pm, v, b["ids"][:1, a:a + S + 1], adapters=False) - ref).abs().max().item()
    assert errs[True] < 1e-4 < errs[False]
