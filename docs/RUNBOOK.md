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

## 1. Runtime: one A100, 40 GB is enough

* **Any Colab A100** (`colab new --gpu A100`; `colab_ctl.py up` tries the standard shape, then high-RAM, and accepts
  ≥ 39,000 MiB). `G-hardware` requires ≥ 39 GB and sm80 before anything is downloaded.
* A failed or timed-out assign creates no session and costs nothing; confirm with `colab sessions` / `colab usage`.
  The job uses the rate `colab usage` reports for the assigned card (8.88 CU/h was measured on the A100 80 GB).

### The 4-bit route (why the experts stay NVFP4/MXFP8)

The release is already 4-bit where it matters: the routed experts are **NVFP4** in layers 0–19 and **MXFP8** in
layers 20–23 (15.67 GB in the file); embeddings, attention, router and the rest are BF16/FP32 (3.22 GB).
transformers 5.24 can only load it by dequantising every expert to BF16 (~49 GB of experts, ~52 GB total), which is
what forced an 80 GB card. Options considered:

| route | resident weights | is the tower still the parent? | verdict |
|---|---|---|---|
| BF16 dequant (transformers) | ~52 GB | yes, up to its own amax rounding (below) | does not fit 40 GB |
| Re-quantise to bitsandbytes NF4 (Unsloth's stock Inkling loader, `load_in_4bit`) | ~14.4 GB | **no**: a second, different 4-bit error on every expert, so every context state moves | rejected |
| **Keep the checkpoint's packed bytes; Unsloth decodes per layer** (`unsloth_zoo.packed_experts`) | **18.9 GB** | **yes, exactly** | **chosen** |

How the chosen route works (`src/inkling_dlm/quant.py`):

* Loading: transformers loads the non-expert weights; its dequantising expert converters are removed and the expert
  stacks created empty, so no BF16 expert tensor is ever allocated. Each experts module then receives the
  checkpoint's own bytes as Unsloth `Nvfp4ExpertParam` (E2M1 codes, E4M3 block scales per 16, per-expert
  `amax / outer_scale_denominator` in FP32) or `Mxfp8ExpertParam` (E4M3 + E8M0 per 32). Gate/up rows are
  de-interleaved on the packed bytes (codes and scales are per row).
* Forward: transformers' `grouped_mm` experts function with each grouped linear replaced by Unsloth's
  `packed_expert_grouped_linear`: only experts that received tokens are decoded (Triton; E4M3 decoded by hand, so
  sm80 works), into a transient BF16 buffer (≤ 1.36 GB for one layer's gate/up), then the same `grouped_mm`.
* Backward: the experts are frozen; Unsloth's autograd saves only the packed tensors and decodes again for dX.
  Gradients reach LoRA through the tower-free canvas path and through checkpoint replays (G-grad, G-ckpt).
* The A100 has no FP4 tensor-core math, so this is weight-only 4-bit (W4A16): the matmuls run in BF16 on decoded
  weights. The fused-dequant GEMMs reported faster for ≤ 64 rows per expert were not needed to fit; the speed gate
  measures what the decode costs.

**What the reference is, honestly.**

* Unsloth's decode equals transformers' own decode formula applied to the checkpoint tensors in their stored dtypes,
  **bitwise in BF16**: 63.7M values on real slices of layers 0, 11, 19 (NVFP4) and 20, 23 (MXFP8) on the CPU
  (`checks/verify_experts.py` keeps a 5-slice subset), and whole layers on the GPU in G-dispatch.
* transformers' actual BF16 `from_pretrained` is **not** that: its loader casts every source tensor to the parameter
  dtype before converting (`core_model_loading.py:1918-1925`), so the FP32 per-expert `amax` is rounded to BF16 first.
  On the real slices that changes 0–57 % of a layer's expert weights, each by up to 0.78 % (the expert's whole scale, ≤ 0.34 %
  amax error). The packed model keeps the FP32 global scale, i.e. the checkpoint's own NVFP4 semantics; the former
  "BF16-loaded parent" carried that rounding. G-dispatch reports the BF16-load difference for the record.
* Not reproduced: the native Blackwell forward (FP4 tensor cores). Weight values are exact; activations stay BF16.
* LoRA-off equivalence (G-exact) compares the packed model's canvas path with the packed model's own full-sequence
  forward, so the tolerance is the existing one (top-1 ≥ 0.995, max |Δlogit| ≤ 0.1, BF16 attention-order rounding);
  on the CPU in FP32 the same comparison is < 1e-4.

### Memory budget on a 40 GB A100 (42.4e9 bytes; gate limits: 36 GB allocated, 95 % of the card reserved)

| item | GB | source |
|---|---|---|
| packed experts (NVFP4 20 layers + MXFP8 4 layers) | 15.67 | checkpoint header |
| non-expert weights (embed 0.93, unembed 0.92, attention/norms/router 1.37) | 3.22 | checkpoint header |
| LoRA r = α = 1024 on q/v `.linear`, 24 modules: 213.9M params FP32 + grads + AdamW m, v | 3.42 | 16 B/param |
| forward bias banks (393k) with grads and Adam | 0.01 | |
| tower states, 16 × 256 tokens × 48 cache layers × K,V (8 heads × 128, BF16) | 0.81 | |
| transient decode of one layer's experts (gate/up 1.36 or down 0.68, freed after the GEMM) | ≤ 1.36 | |
| **static + transient** | **≈ 24.5** | |
| left for canvas activations and the allocator | ≈ 11.5 to the 36 GB limit | measured by G-mem/speed |

### Unsloth's Inkling 2 Mini techniques (guide: unsloth.ai/docs/models/inkling-2-mini/fine-tuning)

Used through Unsloth's own code at the pinned commits (`src/inkling_dlm/unsloth_ops.py`), without `import unsloth`:

* **Message-residual recompute under checkpointing** (`unsloth/models/inkling2.py`,
  `checkpoint_inkling2_message_inputs`): the loop message residual (24 calls per forward) keeps only its inputs and is
  recomputed in backward. Engages only with `grad_ckpt`; G-load counts the wrapped residual, G-ckpt counts its
  recomputes and requires identical gradients.
* **Chunked cross entropy with the final logit softcap** (`unsloth_zoo/fused_losses/chunked_logprob.py`,
  `unsloth_chunked_logprobs`, `logit_softcapping = 30`): training and held-out CE never keep the fp32
  `[masked slots, 200,058]` logits, their softcap or softmax; the head GEMM accumulates in fp32.
* **16-bit message gates**: only the routed experts are packed; G-load checks every `message_gate` weight is BF16/FP32.
* **Bounded expert-decode scratch** (`UNSLOTH_MXFP4_EXPERT_CHUNK_MB`, `unsloth_zoo/mxfp4_stacked_experts.py`): off by
  default, as Unsloth leaves Mini's 1.27 GiB gate/up stack whole on 40 GB cards; the gates fall back to 192 MiB chunks
  only if no setting fits.
* **Chat template**: `checks/verify_data.py` proves the encoding is token-for-token Inkling's own template with
  `reasoning_effort=0` / `"none"`; the loss covers answer blocks only (Unsloth's `train_on_responses_only`).
* **expandable_segments** allocator (what `unsloth_zoo` sets on import), set explicitly by the CLI.

### Batch sizing on 40 GB

The recipe stays BYOD's: 16 examples per update, `maxlen` 256 (the dataset is a 256-token selection), 25,000 updates.
What the spare memory buys is fewer, larger micro-batches of the same update (identical gradients, `micro_blocks`
canvases per forward/backward). That is the opposite of Unsloth's "batch 1, raise gradient accumulation" advice, which
is about 1,024–2,048-token rows whose 200K-vocabulary logits dominate: here a canvas is 33 tokens, the logits are only
computed for masked slots and are now chunked, and each micro-batch pays a fixed cost, because every packed-expert call
decodes nearly every routed expert (≥ 500 tokens × 8 experts hit almost all 128), once in forward, once in backward,
once more in a checkpoint replay. An update has 42 canvases on average (2.63 blocks × 16) and at most 128.

* Unsloth's measured point: 2,048 tokens at batch 1 with checkpointing peaks at 27.9 GiB with 14.5 GiB of weights,
  ~13 GiB for activations, logits and optimizer, on rows attending 2,048 keys with LoRA on the experts. A canvas
  micro-batch of 64 is 2,112 tokens attending ≤ 289 keys with frozen experts, so ≤ ~13 GiB on top of our ≈ 22.8 GiB
  static: ≤ ~36 GiB, expected well under it.
* Ladder (`gate_settings`): `[ckpt, 128]` (every update in one micro-batch), `[ckpt, 64]`, `[ckpt, 32]`,
  `[no ckpt, 16]` (the former default). Each is probed on the worst-memory update (`peak_batch`: the 16 rows with the
  most blocks, 128 canvases, widest tower) plus 4 timed training-ordered updates; the fastest that stays under 36 GB
  allocated and 95 % reserved is written to `gates.json` and used by training; the full G-mem/speed run then times
  `gate_updates` updates in that setting for the CU projection.

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
| G-hardware | ≥ 39 GB GPU (any A100), sm80+ (checked before the 20 GB download) |
| G-load | no missing keys; unexpected keys only `visual.*`/MTP; nothing mismatched; all 24 experts modules packed (NVFP4 + MXFP8); every message-gate weight 16/32-bit; the message residual under Unsloth's checkpoint |
| G-dispatch | `experts_implementation == unsloth_packed`; on layers 0, 6, 19 (NVFP4), 20, 23 (MXFP8): Unsloth's decode == transformers' dequantisation of the raw checkpoint tensors, bitwise; packed forward and dX vs the eager loop on those weights (rel < 2e-2) |
| G-exact | LoRA off + causal canvas reproduces the parent's full-sequence logits, training path and sampler path: top-1 ≥ 0.995, max \|Δlogit\| ≤ 0.1; LoRA on at init (B = 0) equals LoRA off |
| G-readonly | tower states bit-identical after canvas forwards; repeat forward identical |
| G-grad | every LoRA-B and forward-bias bank gets a finite gradient; LoRA-A too once B ≠ 0; nothing frozen gets one |
| G-ckpt | gradients with checkpointing on equal those with it off (rel. diff < 1e-2): the read-only cache survives replay; Unsloth's message-residual recompute runs with checkpointing and only then |
| G-mem/speed | probes every `gate_settings` entry (worst-memory update + 4 timed), keeps the fastest that fits; then the worst update + `gate_updates` training-ordered updates incl. clip + AdamW step in that setting: peak < min(36 GB, 92 % of the card) allocated and < 95 % reserved, projected training ≤ **200 CU**; reports s/update, tower tokens/s, canvas slots/s, peak allocated/reserved, and records the setting |

Any FAIL ends the session. Training refuses to start unless `run_dir/gates.json` says `passed` (the gate-chosen
`grad_ckpt` / `micro_blocks` / `dequant_chunk_mb` are applied automatically).

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
