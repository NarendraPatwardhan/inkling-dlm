"""LAD data: the exact 399,998-row BYOD selection, chat-template encoding, length-bucketed order, block collation.

Data order and corruption noise are pure functions of (seed, update), so a resumed run replays the same batches."""
from __future__ import annotations

import glob
import hashlib
import json
import math
from importlib import resources
from pathlib import Path

import numpy as np
import torch

from . import constants as K


def selection_index() -> np.ndarray:
    with resources.as_file(resources.files("inkling_dlm") / "resources" / "byod_selection_index.npy") as p:
        sel = np.load(p)
    if sel.shape != (K.SELECTION_ROWS,):
        raise ValueError(f"selection index has shape {sel.shape}, expected ({K.SELECTION_ROWS},)")
    return sel


def parquet_files(cfg, split: str) -> list[str]:
    if cfg.data_local_dir:
        files = sorted(glob.glob(f"{cfg.data_local_dir}/data/{split}-*.parquet"))
    else:
        from huggingface_hub import snapshot_download
        d = snapshot_download(cfg.data_repo, repo_type="dataset", revision=cfg.data_revision,
                              allow_patterns=[f"data/{split}-*.parquet"])
        files = sorted(glob.glob(f"{d}/data/{split}-*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet files for split {split!r}")
    return files


COLUMNS = ["instruction", "input", "output", "system"]


def lad_rows(cfg, split: str, index=None) -> list[dict]:
    """Rows of a LAD split in file order (the selection index refers to the concatenated train files)."""
    import pyarrow.dataset as ds
    d = ds.dataset(parquet_files(cfg, split), format="parquet")
    if index is None:
        return d.to_table(columns=COLUMNS).to_pylist()
    return d.take(np.asarray(index, dtype=np.int64), columns=COLUMNS).to_pylist()


def prompt_text(instruction: str, inp: str | None = None, system: str | None = None) -> str:
    user = instruction if not inp else f"{instruction}\n\n{inp}"
    system = (system or K.DEFAULT_SYSTEM).strip()
    return (f"<|message_system|><|content_text|>{system}<|end_message|>"
            f"<|message_system|><|content_text|>Thinking effort level: 0<|end_message|>"
            f"<|message_user|><|content_text|>{user}<|end_message|><|message_model|><|content_text|>")


def encode(tok, r: dict, maxlen: int):
    """Returns (prompt_ids, answer_ids, ended). The answer ends with <|end_message|><|content_model_end_sampling|>;
    rows longer than maxlen are truncated and flagged not ended (they contribute full blocks only)."""
    p = tok(prompt_text(r["instruction"], r["input"], r["system"]), add_special_tokens=False)["input_ids"]
    a = tok(r["output"], add_special_tokens=False)["input_ids"] + [K.END_MSG_ID, K.EOS_ID]
    ended = len(p) + len(a) <= maxlen
    return p, a[: max(0, maxlen - len(p))], ended


def n_blocks(answer_len: int, ended: bool, block: int = K.S) -> int:
    return math.ceil(answer_len / block) if ended else answer_len // block


def encode_all(tok, rows, maxlen: int):
    return [encode(tok, r, maxlen) for r in rows]


def load_training_examples(cfg, tok):
    """Encode the selection once and cache it (flat int32 arrays) under cache_dir, keyed by the data identity."""
    sel = selection_index()
    if cfg.max_examples:
        sel = sel[: cfg.max_examples]
    ident = json.dumps([cfg.data_repo, cfg.data_revision, cfg.model_repo, cfg.model_revision, cfg.maxlen,
                        hashlib.sha1(sel.tobytes()).hexdigest()])
    key = hashlib.sha1(ident.encode()).hexdigest()[:12]
    cache_dir = Path(cfg.cache_dir or Path(cfg.run_dir) / "cache")
    path = cache_dir / f"lad-train-{key}.npz"
    if path.exists():
        z = np.load(path)
        return _unflatten(z), {"data_id": key, "cached": True}
    data = encode_all(tok, lad_rows(cfg, "train", sel), cfg.maxlen)
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".partial.npz")
    np.savez(tmp, **_flatten(data))
    tmp.rename(path)
    return data, {"data_id": key, "cached": False}


def _flatten(data):
    pl = np.array([len(p) for p, _, _ in data], dtype=np.int32)
    al = np.array([len(a) for _, a, _ in data], dtype=np.int32)
    return dict(prompt=np.concatenate([np.asarray(p, np.int32) for p, _, _ in data]),
                answer=np.concatenate([np.asarray(a, np.int32) for _, a, _ in data]),
                plen=pl, alen=al, ended=np.array([e for _, _, e in data], dtype=bool))


def _unflatten(z):
    po = np.concatenate([[0], np.cumsum(z["plen"])])
    ao = np.concatenate([[0], np.cumsum(z["alen"])])
    P, A = z["prompt"], z["answer"]
    return [(P[po[i]:po[i + 1]].tolist(), A[ao[i]:ao[i + 1]].tolist(), bool(z["ended"][i]))
            for i in range(len(z["plen"]))]


def data_stats(data, block: int = K.S) -> dict:
    lens = np.array([len(p) + len(a) for p, a, _ in data])
    ended = np.array([e for _, _, e in data])
    blocks = sum(n_blocks(len(a), e, block) for _, a, e in data)
    return dict(examples=len(data), truncated_frac=float(1 - ended.mean()), mean_len=float(lens.mean()),
                mean_prompt=float(np.mean([len(p) for p, _, _ in data])),
                mean_answer=float(np.mean([len(a) for _, a, _ in data])), blocks=int(blocks),
                zero_block_examples=int(sum(n_blocks(len(a), e, block) == 0 for _, a, e in data)))


def batch_order(lengths, seed: int, per_update: int):
    """Length-bucketed batches: shuffle, sort inside windows of 50 batches, shuffle the batches."""
    rng = np.random.default_rng(seed)
    perm, W, out = rng.permutation(len(lengths)), per_update * 50, []
    for i in range(0, len(perm), W):
        w = perm[i:i + W]
        w = w[np.argsort(np.asarray(lengths)[w], kind="stable")]
        bs = [w[j:j + per_update] for j in range(0, len(w) - per_update + 1, per_update)]
        rng.shuffle(bs)
        out += bs
    return out


def collate(examples, step: int, seed: int, device, block: int = K.S):
    """examples: list of (prompt_ids, answer_ids, ended). Right-padded tower inputs (positions start at 0 for every
    example, exactly as at inference) and one canvas per answer block."""
    g = torch.Generator().manual_seed(seed * 1_000_003 + step)
    T = max(len(p) + len(a) for p, a, _ in examples)
    ids = torch.full((len(examples), T), K.PAD_ID, dtype=torch.long)
    attn = torch.zeros(len(examples), T, dtype=torch.long)
    rows, anchors, tgt, region, n_ans, n_pad = [], [], [], [], [], []
    lens = []
    for i, (p, a, ended) in enumerate(examples):
        seq = p + a
        ids[i, :len(seq)] = torch.tensor(seq)
        attn[i, :len(seq)] = 1
        lens.append(len(seq))
        for k in range(n_blocks(len(a), ended, block)):
            blk = a[block * k:block * k + block]
            rows.append(i)
            anchors.append(len(p) + block * k - 1)
            tgt.append(blk + [K.EOS_ID] * (block - len(blk)))
            region.append([0] * len(blk) + [1] * (block - len(blk)))
            n_ans.append(len(blk))
            n_pad.append(block - len(blk))
    if not rows:
        raise ValueError("batch has no trainable blocks")
    rows_t, anchors_t = torch.tensor(rows), torch.tensor(anchors)
    tgt_t, region_t = torch.tensor(tgt), torch.tensor(region)
    Bc = len(rows)
    t = torch.rand(Bc, generator=g) * (1 - 1e-3) + 1e-3                  # t ~ U[1e-3, 1] per block
    masked = torch.rand(Bc, block, generator=g) < t[:, None]
    none = ~masked.any(1)
    masked[none, torch.randint(0, block, (int(none.sum()),), generator=g)] = True   # >= 1 loss slot per block
    canvas = torch.cat([ids[rows_t, anchors_t][:, None], torch.where(masked, K.MASK_ID, tgt_t)], dim=1)
    kv_valid = attn.bool()[rows_t] & (torch.arange(T)[None] <= anchors_t[:, None])
    d = dict(ids=ids, attn=attn, lens=torch.tensor(lens), rows=rows_t, anchors=anchors_t, tgt=tgt_t,
             region=region_t, t=t, masked=masked, canvas=canvas, kv_valid=kv_valid,
             n_ans=torch.tensor(n_ans, dtype=torch.float), n_pad=torch.tensor(n_pad, dtype=torch.float))
    return {k: v.to(device) for k, v in d.items()}


def eval_batches(cfg, tok, n: int, split: str = "validation", first: int = 2000, min_blocks: int = 3):
    """Fixed held-out groups of examples_per_update ended rows with at least `min_blocks` answer blocks."""
    data = encode_all(tok, lad_rows(cfg, split, list(range(first))), cfg.maxlen)
    data = [d for d in data if d[2] and len(d[1]) > (min_blocks - 1) * cfg.block]
    G = cfg.examples_per_update
    out = [data[i * G:(i + 1) * G] for i in range(n)]
    return [b for b in out if len(b) == G]
