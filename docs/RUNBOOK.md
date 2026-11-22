# A100 runbook

One notebook flow (`notebooks/inkling_dlm_a100.ipynb`) or the same steps from a shell (`scripts/colab_ctl.py`).
Stage 1 runs every GPU gate; stage 2 (training) needs the owner's approval; stage 3 runs the sanity checks; the
runtime is stopped as soon as the work ends.

## 0. Before any GPU

```bash
pip install -e ".[dev]"
INKLING_LAD_DIR=/path/to/LAD-snapshot INKLING_FULL_DATA=1 python -m pytest checks   # all CPU checks must pass
```

The suite builds a tiny random Inkling from the real `config.json` fields, modeling code (transformers 5.24.0) and
tokenizer; no weights are downloaded. See `checks/` for what each file verifies.

## 1. Runtime

* **A100 80 GB** (`colab new --gpu A100 --high-mem`). The BF16 text model is about 52 GB; a 40 GB A100 cannot hold it.
  `G-hardware` checks this before anything is downloaded and stops the run otherwise.
* High-RAM shape: loading dequantises the NVFP4/MXFP8 experts. If the high-RAM A100 is unavailable
  (`Service Unavailable` on assign), `colab_ctl.py gates --no-high-mem` tries the standard shape. A failed or
  timed-out assign creates no session and costs nothing; confirm with `colab sessions` / `colab usage`.
* Measured rate for budgeting: 8.88 CU/h (A100 80 GB); `colab usage` before and after every session.

## 2. Secrets

* W&B: Colab secret `WANDB_API_KEY`, or the key file uploaded file-to-file
  (`colab upload -s $S ~/.config/wandb/key /content/wandb.key`). The package reads it at runtime
  (`WANDB_API_KEY` env, else `wandb_key_file`) and never prints or logs it. W&B host: `https://api.forge.coreweave.com`,
  entity `narendra-patwardhan-independent`, project `inkling-dlm`.
* Never put a key in a notebook cell, exec'd code, config, or command line.

## 3. Stage 1: GPU gates (first A100 session, a few CU)

From a shell on the box (detached; the session stops itself when the job ends):

```bash
python scripts/colab_ctl.py gates --commit <SHA>            # up -> push -> launch gates -> wait -> pull -> stop
cat outputs/colab/gates-gates.json outputs/colab/ledger.jsonl
```

or notebook cells 1–7. Gates, in order (`src/inkling_dlm/gates.py`):

| gate | passes when |
|---|---|
| G-hardware | ≥ 70 GB GPU, sm80+ (checked before the 20 GB download) |
| G-load | no missing keys; unexpected keys only `visual.*`/MTP; nothing mismatched |
| G-dispatch | `experts_implementation == grouped_mm` (no silent Python-loop fallback); grouped_mm agrees with the eager loop on one MoE block |
| G-exact | LoRA off + causal canvas reproduces the parent's full-sequence logits, training path and sampler path: top-1 ≥ 0.995, max \|Δlogit\| ≤ 0.1; LoRA on at init (B = 0) equals LoRA off |
| G-readonly | tower states bit-identical after canvas forwards; repeat forward identical |
| G-grad | every LoRA-B and forward-bias bank gets a finite gradient; LoRA-A too once B ≠ 0; nothing frozen gets one |
| G-ckpt | gradients with checkpointing on equal those with it off (rel. diff < 1e-2): the read-only cache survives replay |
| G-mem/speed | 30 real-data updates: peak < 76 GB and projected training ≤ **200 CU**. If the no-checkpointing setting fails, it retries with checkpointing on and records the chosen setting in `gates.json` |

Any FAIL ends the session. Training refuses to start unless `run_dir/gates.json` says `passed` (the gate-chosen
`grad_ckpt` / `micro_blocks` are applied automatically).

The gates-only config (`configs/a100-gates.yaml`) keeps everything on the VM disk, so no Drive mount is needed for this
stage. For the main run, run the gates with `configs/a100.yaml` (Drive `run_dir`) so `gates.json` sits next to the
checkpoints.

## 4. Stage 2: training (owner approval required)

* Notebook: set `TRAIN_APPROVED = True`, run all cells. Shell:
  `python -m inkling_dlm train --config configs/a100.yaml --approved` (detached; logs to Drive).
* Drive layout: `MyDrive/inkling-dlm/runs/main/state-NNNNNN/` (last 2, atomic `.partial` → rename), `wandb_id`,
  `gates.json`, `train.log`; tokenised data cache in `MyDrive/inkling-dlm/cache/`.
* Preemption or disconnect: re-run the notebook (or the train command). It resumes from the newest complete state
  with the same data order, masks, optimizer, scheduler and RNG, and continues the same W&B run.
* W&B panels: `train/` (loss, answer/pad terms, grad norms, lr), `eval/heldout_*` every 1,000 updates, `sys/`
  (s/update, peak memory, blocks, projected CU, remaining hours).
* Guards: three consecutive non-finite losses abort the run (state is on Drive).

## 5. Stage 3: sanity checks, then stop

`python -m inkling_dlm sanity --config configs/a100.yaml` — held-out masked CE (parent vs trained) at t ∈ {.1,…,.9},
bidirectional copy test (clue vs no clue), 8 code prompts decoded by the parent (AR) and by the diffusion sampler at
32 and 16 steps per block, with parse rates. W&B: summary, tables, a CE-vs-t line plot and a copy-test bar chart;
`sanity.json` on Drive. Then `runtime.unassign()` / `colab stop`, and `colab sessions` must show none.
