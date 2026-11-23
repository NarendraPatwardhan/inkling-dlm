#!/usr/bin/env python3
"""Regenerate notebooks/inkling_dlm_a100.ipynb (the notebook only drives the package; no logic lives in it)."""
from pathlib import Path

import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = [
    md("""# Inkling 2 Mini → block diffusion on one A100 (40 GB is enough)

This notebook drives the `inkling_dlm` package (https://github.com/NarendraPatwardhan/inkling-dlm). One flow:

1. **Stage 1 — GPU gates** (≈ 15–25 min, a few CU): load (experts kept packed NVFP4/MXFP8), dispatch (Unsloth's
   packed-expert decode == transformers' dequantisation, bitwise; packed grouped GEMM vs the eager loop), LoRA-off
   logit equivalence, read-only cache, gradient flow, gradients through the cache under checkpointing, memory and
   speed with the **projected compute-unit cost**. Any failure — including a projection above 200 CU — stops here.
2. **Stage 2 — training** (25,000 updates). Runs **only** when `TRAIN_APPROVED = True` (the owner's approval).
   Detached; checkpoints every 500 updates to Drive; re-running the notebook resumes.
3. **Stage 3 — sanity checks** (held-out masked CE, copy test, code samples) and **stop the runtime**.

Everything is logged to W&B (`narendra-patwardhan-independent/inkling-dlm`). The key is read from a Colab secret
(`WANDB_API_KEY`) or `/content/wandb.key` and is never printed. Runtime: **any A100** (40 GB or 80 GB): the routed
experts stay in the checkpoint's own 4-bit bytes (15.7 GB) and Unsloth's kernels decode them per layer, so the model
holds ~19 GB instead of ~52 GB."""),
    code("""# Parameters
REPO = "https://github.com/NarendraPatwardhan/inkling-dlm.git"
COMMIT = "main"                  # pin a commit SHA for the real run
CONFIG = "configs/a100.yaml"
TRAIN_APPROVED = False           # set True only after the owner has approved the main training run
RUN_DIR = "/content/drive/MyDrive/inkling-dlm/runs/main"
J = "/content/inkling-dlm\""""),
    code("""# Hardware: any A100 (40 GB holds the packed model, ~19 GB, plus training state)
import subprocess
gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                     capture_output=True, text=True).stdout.strip()
print(gpu)
name, mem = gpu.rsplit(",", 1)
assert "A100" in name and int(mem) > 39_000, f"need an A100 runtime, got {gpu}"
"""),
    code("""# Install the package at the pinned commit (keeps Colab's torch; pins transformers/peft/accelerate), then
# Unsloth (packed-expert kernels) without its dependency pins (they would downgrade transformers below 5.24)
import os
if not os.path.exists(J):
    !git clone -q {REPO} {J}
!cd {J} && git fetch -q origin && git checkout -q {COMMIT} && git log -1 --format='%H %s'
!pip install -q -e {J}
import sys; sys.path.insert(0, f"{J}/src")
from inkling_dlm.constants import UNSLOTH
!pip install -q --no-deps {" ".join(f'"{p}"' for p in UNSLOTH)}
!python -c "import torch, transformers, peft, triton; print(torch.__version__, transformers.__version__, peft.__version__, triton.__version__)"
"""),
    code("""# Google Drive: checkpoints, data cache, gates.json, sanity.json
from google.colab import drive
drive.mount("/content/drive")
os.makedirs(RUN_DIR, exist_ok=True)
"""),
    code("""# W&B key -> /content/wandb.key (chmod 600). Never printed.
import pathlib, getpass
key_file = pathlib.Path("/content/wandb.key")
if not key_file.exists():
    key = None
    try:
        from google.colab import userdata
        key = userdata.get("WANDB_API_KEY")
    except Exception:
        pass
    key = key or getpass.getpass("W&B API key (hidden): ")
    key_file.write_text(key.strip()); key_file.chmod(0o600); del key
os.environ["WANDB_KEY_FILE"] = str(key_file)
print("W&B key: available" if key_file.stat().st_size > 0 else "W&B key: MISSING")
"""),
    md("## Stage 1 — GPU gates (must all pass)"),
    code("""!cd {J} && python -m inkling_dlm gates --config {CONFIG}
import json
g = json.load(open(f"{RUN_DIR}/gates.json"))
for r in g["gates"]:
    print(("PASS " if r["passed"] else "FAIL ") + r["gate"].ljust(24) + r["detail"])
print("setting:", g["setting"]); print("projection:", g["projection"])
assert g["passed"], "a gate failed: do not train"
"""),
    md("## Stage 2 — training (only with the owner's approval)"),
    code("""if not TRAIN_APPROVED:
    raise SystemExit("Gates passed. Training needs the owner's approval: set TRAIN_APPROVED = True and re-run.")
import subprocess
log = f"{RUN_DIR}/train.log"
cmd = f"cd {J} && nohup python -m inkling_dlm train --config {CONFIG} --approved >> {log} 2>&1 &"
subprocess.run(["bash", "-c", cmd], env=dict(os.environ, PYTHONUNBUFFERED="1"))
print("training launched (detached); it resumes from the newest Drive checkpoint if one exists")
"""),
    code("""# Monitor (re-run any time): last log lines and the newest checkpoint
import glob
print("".join(open(f"{RUN_DIR}/train.log").readlines()[-15:]))
print(sorted(p for p in glob.glob(f"{RUN_DIR}/state-*") if not p.endswith(".partial"))[-1:])
"""),
    md("## Stage 3 — sanity checks, then stop the runtime"),
    code("""!cd {J} && python -m inkling_dlm sanity --config {CONFIG}
s = json.load(open(f"{RUN_DIR}/sanity.json"))
print("held-out CE  parent:", round(s["init"]["heldout"]["ce"], 3), " trained:", round(s["final"]["heldout"]["ce"], 3))
print("copy test    parent:", s["init"]["copy"], " trained:", s["final"]["copy"])
print("parse rates:", s["final"]["parse_rate"])
"""),
    code("""# Never leave the runtime idle
from google.colab import runtime
runtime.unassign()
"""),
]
nb = nbf.v4.new_notebook(cells=cells, metadata={
    "accelerator": "GPU", "colab": {"gpuType": "A100", "provenance": []},
    "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}})
for i, c in enumerate(nb.cells):
    c.id = f"cell-{i:02d}"                          # deterministic ids: regeneration is byte-identical
out = Path(__file__).resolve().parents[1] / "notebooks" / "inkling_dlm_a100.ipynb"
nbf.write(nb, out)
print("wrote", out)
