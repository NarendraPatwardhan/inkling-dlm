"""Read an offline W&B run file (.wandb, LevelDB log format) into protobuf records, for verification only."""
from __future__ import annotations

import struct
from pathlib import Path

BLOCK = 32 * 1024
HEADER = 7                      # crc32 (4) + length (2) + type (1)
FULL, FIRST, MIDDLE, LAST = 1, 2, 3, 4


def records(path: str | Path):
    from wandb.proto import wandb_internal_pb2 as pb
    data = Path(path).read_bytes()
    assert data[:3] == b":W&", "not a W&B run file"
    pos, buf = 7, b""                                   # file header ":W&B" + 2-byte magic + 1-byte version
    while pos + HEADER <= len(data):
        left = BLOCK - (pos % BLOCK)
        if left < HEADER:                               # block trailer padding
            pos += left
            continue
        _, length, kind = struct.unpack("<IHB", data[pos:pos + HEADER])
        if kind == 0 and length == 0:
            pos += left
            continue
        chunk = data[pos + HEADER:pos + HEADER + length]
        pos += HEADER + length
        if kind == FULL:
            buf = chunk
        elif kind == FIRST:
            buf = chunk
            continue
        elif kind == MIDDLE:
            buf += chunk
            continue
        elif kind == LAST:
            buf += chunk
        r = pb.Record()
        r.ParseFromString(buf)
        yield r


def summarize(path) -> dict:
    out = {"config": {}, "history": [], "summary": {}, "kinds": {}}
    for r in records(path):
        k = r.WhichOneof("record_type")
        out["kinds"][k] = out["kinds"].get(k, 0) + 1
        if k == "run":
            out["config"].update({u.key: u.value_json for u in r.run.config.update})
            out["run_id"] = r.run.run_id
        elif k == "config":
            out["config"].update({u.key: u.value_json for u in r.config.update})
        elif k == "history":
            out["history"].append({i.key or "/".join(i.nested_key): i.value_json for i in r.history.item})
        elif k == "summary":
            out["summary"].update({u.key or "/".join(u.nested_key): u.value_json for u in r.summary.update})
    return out
