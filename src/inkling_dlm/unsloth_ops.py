"""Unsloth's Inkling 2 Mini training techniques, used through Unsloth's own code (pinned in `constants.UNSLOTH`).

`import unsloth` would patch transformers' trainer, attention and generation globally (and its stock Inkling loader
re-quantises the experts to bitsandbytes NF4, see quant.py), so the pieces this loop needs are taken one by one:

* `checkpoint_message_inputs` - unsloth/models/inkling2.py `checkpoint_inkling2_message_inputs`: under gradient
  checkpointing the looped layers' message residual (layers 6-17, passes 2 and 3: 24 calls per forward) keeps only its
  inputs and is recomputed in backward, op for op the module's math (guide: "Gradient checkpointing for the looped
  layers ... recomputing the message residual", ~1 GB at 1024 tokens). That file imports only `re` at module level, so
  it is loaded on its own, without running `unsloth/__init__.py`.
* `token_ce` - unsloth_zoo/fused_losses/chunked_logprob.py `unsloth_chunked_logprobs` with
  `logit_softcapping = final_logit_softcapping` (guide: "Fused cross entropy with the final logit softcap"): per-token
  log-probabilities of softcap(h @ W_head^T) computed chunk by chunk, keeping only the log-sum-exp for backward, so the
  fp32 [tokens, 200,058] logits, their softcap and softmax are never held; fp32-accumulated head GEMM.
* `DEQUANT_CHUNK_ENV` - unsloth_zoo/mxfp4_stacked_experts.py's chunk size for the packed NVFP4/MXFP8 expert decode
  (guide: "Bounded expert dequantization scratch"). Unsloth leaves Mini's 1.27 GiB gate/up stack whole on a 40 GB card
  (temporary_patches/moe_utils.py `_auto_dequant_decide_chunk`); here it is a measured fallback setting of the gates.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import torch

DEQUANT_CHUNK_ENV = "UNSLOTH_MXFP4_EXPERT_CHUNK_MB"
_INKLING2 = "inkling_dlm._unsloth_inkling2"


def _unsloth_dir() -> Path:
    spec = importlib.util.find_spec("unsloth")          # locates the package without importing it
    if spec is None or not spec.submodule_search_locations:
        raise ImportError("the pinned `unsloth` package is not installed (see constants.UNSLOTH)")
    return Path(list(spec.submodule_search_locations)[0])


def inkling2_module():
    """unsloth/models/inkling2.py as a standalone module (not `unsloth.models.inkling2`, which would import unsloth)."""
    if _INKLING2 in sys.modules:
        return sys.modules[_INKLING2]
    path = _unsloth_dir() / "models" / "inkling2.py"
    spec = importlib.util.spec_from_file_location(_INKLING2, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[_INKLING2] = mod
    spec.loader.exec_module(mod)
    return mod


def checkpoint_message_inputs(model) -> int:
    """Wrap every loop text model's message residual; it engages only under gradient checkpointing, in training, with
    grad enabled and eager attention (Unsloth's own conditions). Returns how many residuals were wrapped."""
    return inkling2_module().checkpoint_inkling2_message_inputs(model)


def message_inputs_checkpointed(model) -> int:
    """Residuals whose instance forward is Unsloth's _CheckpointedMessageInput."""
    return sum(1 for m in model.modules() if getattr(getattr(getattr(m, "message_input", None), "forward", None),
                                                    "_unsloth_checkpointed_message_input", False))


def token_ce(lm, hidden: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Per-token cross entropy (float32, [N]) of softcap(lm_head(hidden)) at `targets`, without materialising logits
    for backward. Same quantity as F.cross_entropy(lm.logit_softcap(lm.lm_head(hidden).float()), targets,
    reduction="none"), with the head GEMM accumulated in fp32 instead of rounded to the model dtype first."""
    os.environ.setdefault("UNSLOTH_IS_PRESENT", "1")
    from unsloth_zoo.fused_losses.chunked_logprob import unsloth_chunked_logprobs
    if hidden.shape[0] == 0:
        return hidden.new_zeros(0, dtype=torch.float32)
    head = lm.lm_head
    logp = unsloth_chunked_logprobs(hidden, head.weight, targets, logit_softcapping=float(lm.logit_softcap.softcap),
                                    lm_head_bias=getattr(head, "bias", None))
    return -logp


def set_dequant_chunk_mb(mb: int | None) -> None:
    """Bound the transient BF16 decode of one packed expert stack to `mb` MiB (None: Unsloth's default, 2 GiB, i.e. a
    whole Mini stack). Read on every call by unsloth_zoo, so it can change between gate settings."""
    if mb:
        os.environ[DEQUANT_CHUNK_ENV] = str(int(mb))
    else:
        os.environ.pop(DEQUANT_CHUNK_ENV, None)


class count_message_math:
    """Counts calls of Unsloth's checkpointed message-residual math (forward + backward recompute) while active."""

    def __enter__(self):
        mod = inkling2_module()
        self._mod, self._orig, self.calls = mod, mod._message_input_math, 0

        def counted(*a, **k):
            self.calls += 1
            return self._orig(*a, **k)
        mod._message_input_math = counted
        return self

    def __exit__(self, *exc):
        self._mod._message_input_math = self._orig
        return False
