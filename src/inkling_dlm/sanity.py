"""Post-training sanity checks: held-out masked loss (parent init vs trained), bidirectional copy test, diffusion
samples on code prompts against the parent's AR decode. Everything goes to W&B (tables + plots) and sanity.json."""
from __future__ import annotations

import ast
import json
import re
import time
from pathlib import Path

import numpy as np
import torch

from .checkpoint import load_state
from .constants import MASK_ID, S
from .data import collate, encode, eval_batches
from .evaluate import T_GRID, heldout_ce, tower_states
from .sampler import generate
from .tower import canvas_logits, view_for_batch

CODE_PROMPTS = [
    "Write a Python function `is_palindrome(s: str) -> bool` that ignores case and non-alphanumeric characters.",
    "Write a Python function that returns the n-th Fibonacci number iteratively.",
    "Write a Python function `merge_sorted(a, b)` that merges two sorted lists into one sorted list.",
    "Write a Python function that counts word frequencies in a string and returns a dict.",
    "Write a Python function `flatten(xs)` that flattens an arbitrarily nested list.",
    "Write a Python class `Stack` with push, pop, peek and is_empty methods.",
    "Write a Python function that checks whether a number is prime.",
    "Write a Python function `chunk(xs, n)` that splits a list into chunks of size n.",
]


@torch.no_grad()
def copy_test(pm, tok, groups, cfg, adapters=True, n=200) -> dict:
    """Mask the first occurrence of a token that recurs later in the same block. Accuracy with the later copy clean
    (clue) vs also masked (no clue). A bidirectional model gains from the clue; the AR parent cannot."""
    dev = cfg.device
    hits = {"clue": [], "noclue": []}
    for i, ex in enumerate(groups):
        if len(hits["clue"]) >= n:
            break
        b = collate(ex, 20_000_000 + i, cfg.seed, dev, cfg.block)
        k, v, s = tower_states(pm, b)
        for j in range(b["tgt"].shape[0]):
            if len(hits["clue"]) >= n:
                break
            blk = b["tgt"][j].tolist()
            pos = [(a, c) for a in range(S) for c in range(a + 2, S) if blk[a] == blk[c]
                   and int(b["region"][j, c]) == 0 and len(tok.decode([blk[a]]).strip()) > 2]
            if not pos:
                continue
            a, c = pos[0]
            r = torch.tensor([j], device=dev)
            view = view_for_batch(pm, k, v, s, b, r)
            for kind in ("clue", "noclue"):
                x = torch.tensor(blk, device=dev)
                m = torch.zeros(S, dtype=torch.bool, device=dev)
                m[a] = True
                if kind == "noclue":
                    m[c] = True
                canvas = torch.cat([b["canvas"][j:j + 1, :1], torch.where(m, MASK_ID, x)[None]], 1)
                lg = canvas_logits(pm, view, canvas, adapters=adapters)[0]
                hits[kind].append(int(lg[a].argmax() == blk[a]))
    return {k: float(np.mean(v)) if v else float("nan") for k, v in hits.items()} | {"cases": len(hits["clue"])}


def prompt_ids(tok, text):
    return encode(tok, {"instruction": text, "input": "", "output": "", "system": None}, 10_000)[0]


def parses(text: str) -> bool:
    m = re.search(r"```(?:python)?\n(.*?)```", text, re.S)
    try:
        ast.parse(m.group(1) if m else text)
        return True
    except SyntaxError:
        return False


@torch.no_grad()
def samples(pm, tok, cfg, prompts=CODE_PROMPTS, ar_tokens=256) -> list[dict]:
    lm = pm.base_model.model
    dev = cfg.device
    rows = []
    for p in prompts:
        ids = prompt_ids(tok, p)
        t0 = time.time()
        with pm.disable_adapter():
            ar = lm.generate(torch.tensor([ids], device=dev), max_new_tokens=ar_tokens, do_sample=False)[0, len(ids):]
        row = {"prompt": p, "ar": tok.decode(ar, skip_special_tokens=True), "s_ar": time.time() - t0}
        for steps in (32, 16):
            t0 = time.time()
            out = generate(pm, ids, max_blocks=cfg.sample_max_blocks, steps_per_block=steps)
            row[f"dlm_nfe{steps}"] = tok.decode(out, skip_special_tokens=True)
            row[f"s_nfe{steps}"] = time.time() - t0
            row[f"tokens_nfe{steps}"] = len(out)
        for k in ("ar", "dlm_nfe32", "dlm_nfe16"):
            row[f"{k}_parses"] = parses(row[k])
        rows.append(row)
        print(json.dumps(row)[:300], flush=True)
    return rows


def run_sanity(cfg, bundle, tracker, state_path=None, groups=None, prompts=CODE_PROMPTS) -> dict:
    pm, tok = bundle.pm, bundle.tok
    pm.eval()
    groups = groups or eval_batches(cfg, tok, 8)
    copy_groups = groups if len(groups) >= 16 else (eval_batches(cfg, tok, 16) if not cfg.tiny else groups)
    init = dict(heldout=heldout_ce(pm, groups, cfg, adapters=False),
                copy=copy_test(pm, tok, copy_groups, cfg, adapters=False, n=cfg.copy_cases))
    st = load_state(cfg, pm, path=state_path)
    if st is None:
        raise FileNotFoundError(f"no complete state in {cfg.run_dir}")
    final = dict(heldout=heldout_ce(pm, groups, cfg), copy=copy_test(pm, tok, copy_groups, cfg, n=cfg.copy_cases))
    rows = samples(pm, tok, cfg, prompts)
    final["parse_rate"] = {k: float(np.mean([r[f"{k}_parses"] for r in rows])) for k in ("ar", "dlm_nfe32",
                                                                                         "dlm_nfe16")}
    # W&B: summary numbers, a held-out-CE-vs-t plot for both models, copy-test bars, samples table
    tracker.summary({"sanity/state_step": st["step"], "sanity/heldout_ce_init": init["heldout"]["ce"],
                     "sanity/heldout_ce_final": final["heldout"]["ce"],
                     "sanity/copy_clue_init": init["copy"]["clue"], "sanity/copy_noclue_init": init["copy"]["noclue"],
                     "sanity/copy_clue_final": final["copy"]["clue"],
                     "sanity/copy_noclue_final": final["copy"]["noclue"]}
                    | {f"sanity/parse_rate_{k}": v for k, v in final["parse_rate"].items()})
    tracker.table("sanity/heldout_by_t", ["t", "parent (LoRA off)", "trained"],
                  [[t, init["heldout"][f"ce_t{t:.1f}"], final["heldout"][f"ce_t{t:.1f}"]] for t in T_GRID])
    wb = tracker.wandb
    tracker.run.log({"sanity/heldout_ce_vs_t": wb.plot.line_series(
        xs=list(T_GRID), ys=[[init["heldout"][f"ce_t{t:.1f}"] for t in T_GRID],
                             [final["heldout"][f"ce_t{t:.1f}"] for t in T_GRID]],
        keys=["parent (LoRA off)", "trained"], title="Held-out masked CE vs mask rate", xname="mask rate t")})
    tracker.bar("sanity/copy_test", ["parent clue", "parent no clue", "trained clue", "trained no clue"],
                [init["copy"]["clue"], init["copy"]["noclue"], final["copy"]["clue"], final["copy"]["noclue"]],
                "case", "accuracy", "Bidirectional copy test")
    cols = ["prompt", "ar", "dlm_nfe32", "dlm_nfe16", "ar_parses", "dlm_nfe32_parses", "dlm_nfe16_parses",
            "s_ar", "s_nfe32", "s_nfe16"]
    tracker.table("sanity/samples", cols, [[r[c] for c in cols] for r in rows])
    out = dict(state=st["path"], step=st["step"], init=init, final=final | {"samples": rows})
    Path(cfg.run_dir).mkdir(parents=True, exist_ok=True)
    (Path(cfg.run_dir) / "sanity.json").write_text(json.dumps(out, indent=1))
    return out
