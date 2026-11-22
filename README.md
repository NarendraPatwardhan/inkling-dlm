# inkling-dlm

Convert Thinking Machines' **Inkling 2 Mini** (`thinkingmachines/Inkling-2-Mini`, a looped MoE autoregressive LM)
into a **block-diffusion language model** on one Colab A100 80GB, with a *frozen-tower* design:

* **Tower** — the parent model itself with LoRA disabled, run causally over the prompt (and committed blocks). Its
  KV and short-convolution states are exported once per example and are **read-only**.
* **Denoiser** — the same weights with LoRA (r = α = 1024 on `self_attn.(q_proj|v_proj).linear`) and a learned
  forward relative-position bank (`proj_fwd`), run on a 33-slot canvas `[anchor, 32 block slots]` that attends to
  the tower states through `PrefixView`, a `Cache` that concatenates but never stores.
* **Objective** — BYOD masked diffusion per 32-token block: `0.9·ℓ_answer + 0.1·ℓ_EOS-pad`, 1/t weighted,
  normalised per region; truncated rows contribute full blocks only.
* **Data** — the exact 399,998-row BYOD selection of `Ruurd/LAD-training-1m-256` (index shipped in the package),
  25,000 updates × 16 examples.

The method, its derivations and the cost model are in the companion guide (*Inkling 2 Mini: AR → diffusion on a
Colab A100*); this repository is its implementation.

## Layout

```
configs/            a100.yaml (main run), a100-gates.yaml (GPU gates only, no Drive), cpu-tiny.yaml (verification)
src/inkling_dlm/
  constants.py      pinned repos/revisions, token ids, block size
  config.py         RunConfig dataclass, YAML + key=value overrides
  patches.py        PrefixView, mask/relative-distance geometry, forward-bias bank, runtime patches
  model.py          real model loader (BF16, grouped_mm forced, eager attention) and tiny random Inkling; LoRA
  tower.py          frozen tower fill, state compaction, canvas forward (softcapped logits)
  data.py           LAD selection, chat encoding, length-bucketed order, block collation
  loss.py           block loss
  train.py          training loop (W&B, resume, guards)
  checkpoint.py     atomic Drive checkpoints (last N), full RNG/optimizer/scheduler state
  evaluate.py       held-out masked CE on fixed blocks and masks
  sampler.py        block-wise low-confidence remasking sampler
  gates.py          pre-training gates (load, dispatch, exactness, read-only, gradients, checkpointing, speed/cost)
  sanity.py         post-training checks (held-out CE, copy test, code samples) with W&B tables and plots
  tracking.py       W&B wrapper (config + git commit, key from env/file, never printed)
  cli.py            python -m inkling_dlm {gates,train,sanity,pipeline,data-stats}
checks/             CPU verification suite (verify_*.py) on a tiny random Inkling
notebooks/          inkling_dlm_a100.ipynb: gates first, then (approved) training, then sanity
scripts/            colab_ctl.py: detached GPU jobs on Colab from a shell
docs/RUNBOOK.md     A100 runbook
```

## Quick start

```bash
pip install -e ".[dev]"
python -m pytest checks            # CPU verification on a tiny random Inkling (no weights downloaded)
python -m inkling_dlm gates --config configs/a100.yaml          # on the A100
python -m inkling_dlm train --config configs/a100.yaml --approved
python -m inkling_dlm sanity --config configs/a100.yaml
```

W&B is mandatory: set `WANDB_API_KEY` or put the key in a file (`wandb_key_file`, default `/content/wandb.key`).
The key is never printed or logged. Runs go to `narendra-patwardhan-independent/inkling-dlm`.

See [docs/RUNBOOK.md](docs/RUNBOOK.md) for the A100 procedure.

## Verification status

CPU (tiny random Inkling, real modeling code/config/tokenizer, real LAD data): `python -m pytest checks` — see
`checks/` and docs/RUNBOOK.md. The full 399,998-row selection reproduces the guide's data statistics exactly
(`checks/expected_data_stats.json`). GPU-only items (checkpoint load and dequantisation, grouped_mm on sm80, BF16
equivalence, memory, speed and cost) are the stage-1 gates on the A100.

## Pins

`transformers==5.24.0`, `peft==0.22.0`, `accelerate==1.16.0`; model `thinkingmachines/Inkling-2-Mini@b85c3cf`;
data `Ruurd/LAD-training-1m-256@6041513`. torch ≥ 2.9 (grouped_mm on sm80).

## License

Apache-2.0. Model and data are under their own licenses.
