"""Data pipeline on real LAD rows: the shipped selection index, chat encoding (equal to Inkling's own chat template),
batched == single encoding, truncation, compact storage and length-bucketed order."""
import numpy as np

from inkling_dlm import constants as K
from inkling_dlm.data import (EncodedSet, batch_order, data_stats, encode, encode_all, lad_rows, prompt_text,
                              selection_index)


def check_selection_index():
    sel = selection_index()
    assert sel.shape == (K.SELECTION_ROWS,) and sel.dtype.kind == "i"
    assert len(np.unique(sel)) == len(sel) and sel.min() >= 0


def check_selection_rows_exist_in_pinned_train_split(cfg):
    import pyarrow.dataset as ds
    from inkling_dlm.data import parquet_files
    n = ds.dataset(parquet_files(cfg, "train"), format="parquet").count_rows()
    assert selection_index().max() < n


def check_encoding_of_real_selected_rows(cfg, bundle):
    tok = bundle.tok
    rows = lad_rows(cfg, "train", selection_index()[:64])
    single = [encode(tok, r, cfg.maxlen) for r in rows]
    assert encode_all(tok, rows, cfg.maxlen) == single
    for r, (p, a, ended) in zip(rows, single):
        assert p[-2:] == [K.MESSAGE_MODEL_ID, K.CONTENT_TEXT_ID]
        assert tok.decode(p) == prompt_text(r["instruction"], r["input"], r["system"])
        assert len(p) + len(a) <= cfg.maxlen
        if ended:
            assert a[-2:] == [K.END_MSG_ID, K.EOS_ID]
            assert tok.decode(a[:-2]) == r["output"]
        else:
            assert K.EOS_ID not in a


def check_compact_storage_round_trip(cfg, bundle, tmp_path):
    rows = lad_rows(cfg, "validation", list(range(50)))
    data = encode_all(bundle.tok, rows, cfg.maxlen)
    es = EncodedSet.from_examples(data)
    assert list(es) == data
    es.save(tmp_path / "x.npz")
    assert list(EncodedSet.load(tmp_path / "x.npz")) == data
    assert data_stats(es) == data_stats(data)


def check_batch_order_buckets_lengths():
    rng = np.random.default_rng(0)
    lengths = rng.integers(20, 256, size=16 * 120)
    order = batch_order(lengths, 42, 16)
    assert len(order) == 120 and all(len(b) == 16 for b in order)
    assert sorted(np.concatenate(order).tolist()) == list(range(16 * 120))
    spread = np.mean([lengths[b].max() - lengths[b].min() for b in order])
    assert spread < 0.25 * (lengths.max() - lengths.min())            # bucketing keeps groups homogeneous
    assert [b.tolist() for b in batch_order(lengths, 42, 16)] == [b.tolist() for b in order]


def check_full_selection_matches_the_guide(cfg, bundle):
    """Encodes all 399,998 selected rows (minutes; cached under .verify/cache). Enabled with INKLING_FULL_DATA=1.
    The guide's prep reports 399,998 rows, 5.264% truncated, prompt 70.1, answer 68.5, 55,427,029 real tokens."""
    import json
    import os
    from pathlib import Path

    import pytest

    from inkling_dlm.data import load_training_examples
    if os.environ.get("INKLING_FULL_DATA") != "1":
        pytest.skip("set INKLING_FULL_DATA=1 to encode the full selection")
    c = type(cfg)(**cfg.to_dict())
    c.max_examples, c.cache_dir = None, str(Path(__file__).resolve().parents[1] / ".verify" / "cache")
    data, _ = load_training_examples(c, bundle.tok)
    want = json.loads((Path(__file__).with_name("expected_data_stats.json")).read_text())
    assert data_stats(data) == want


def check_encoding_equals_inkling_chat_template(cfg, bundle):
    """Unsloth's guide formats data with Inkling's own chat template (reasoning_effort controls the thinking line).
    The hand-built encoding must be token-for-token the template with reasoning_effort=0 ("none"): system, the
    "Thinking effort level: 0" system line, user, then the model turn ending <|end_message|><|content_model_end_sampling|>."""
    tok = bundle.tok
    rows = lad_rows(cfg, "train", selection_index()[:64])
    n = 0
    for r in rows:
        p, a, ended = encode(tok, r, 10_000)
        user = r["instruction"] if not r["input"] else f"{r['instruction']}\n\n{r['input']}"
        conv = [{"role": "system", "content": (r["system"] or K.DEFAULT_SYSTEM).strip()},
                {"role": "user", "content": user}, {"role": "assistant", "content": r["output"]}]
        for effort in (0, "none"):
            ids = tok.apply_chat_template(conv, tokenize=True, reasoning_effort=effort)
            ids = ids["input_ids"] if isinstance(ids, dict) or hasattr(ids, "keys") else ids
            assert list(ids) == p + a, r["instruction"][:60]
        n += 1
    assert n == 64
