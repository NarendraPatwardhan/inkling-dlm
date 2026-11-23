"""Pinned identifiers. Token ids were read from tokenizer.json at the pinned model revision."""
MODEL_REPO = "thinkingmachines/Inkling-2-Mini"
MODEL_REVISION = "b85c3cfa615fc413cc9af3379f491b963f3b586a"
# Unsloth (the zoo holds the packed NVFP4/MXFP8 expert kernels; it requires the unsloth package to be installed),
# pinned to commits of 2026-11-23; installed with --no-deps (their pins cap transformers at 5.18, this model needs
# 5.24). `unsloth` itself is not imported: it would patch transformers globally. AGPL-3.0, used at runtime only.
UNSLOTH = ("unsloth @ git+https://github.com/unslothai/unsloth.git@f5407c052a224792f1f1bdc549b3bdfc2cccf02c",
           "unsloth_zoo @ git+https://github.com/unslothai/unsloth-zoo.git@40b7262a0b1e81a2e767556d2bf72fd7f14ca9aa")
DATA_REPO = "Ruurd/LAD-training-1m-256"
DATA_REVISION = "6041513b684d6ca627879ffdb547b8629bb8b8de"

S = 32                      # block size: training block and the largest block the sampler may use
MASK_ID = 147742            # "MASK": an ordinary single token (BYOD-style reuse)
EOS_ID = 200006             # <|content_model_end_sampling|>
END_MSG_ID = 200010         # <|end_message|>
PAD_ID = 199999             # <|endoftext|>
MESSAGE_MODEL_ID = 200001   # <|message_model|>
CONTENT_TEXT_ID = 200004    # <|content_text|>
DEFAULT_SYSTEM = "You are a helpful assistant."
SELECTION_ROWS = 399_998
