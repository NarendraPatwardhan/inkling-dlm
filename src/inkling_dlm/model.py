"""Model construction: the real Inkling 2 Mini (text path; BF16 non-expert weights, routed experts kept in the
checkpoint's packed NVFP4/MXFP8 bytes and decoded by Unsloth's kernels) or a tiny random Inkling built from the real
config and modeling code for CPU verification, whose experts go through the same checkpoint layout and packed path.
Both get the same LoRA + forward-bias-bank wrapping."""
from __future__ import annotations

import json
from dataclasses import dataclass

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer
from transformers.models.inkling2_loop.configuration_inkling2_loop import Inkling2LoopTextConfig
from transformers.models.inkling2_loop.modeling_inkling2_loop import Inkling2LoopForCausalLM

from . import quant, unsloth_ops
from .patches import add_forward_bias_bank, apply_patches

LORA_TARGET = r".*\.self_attn\.(q_proj|v_proj)\.linear"   # the inner nn.Linear of the clipped projections


@dataclass
class Bundle:
    tok: object
    pm: object                 # PeftModel
    fwd: list                  # forward-bias-bank parameters
    lora: list                 # LoRA parameters
    load_info: dict | None
    expert_source: object = None   # raw checkpoint expert tensors: safetensors path (real) or dict (tiny)
    message_ckpt: int = 0          # loop message residuals under Unsloth's checkpointed recompute


def load_tokenizer(cfg):
    return AutoTokenizer.from_pretrained(cfg.model_repo, revision=cfg.model_revision)


def real_text_config(cfg) -> dict:
    from huggingface_hub import hf_hub_download
    p = hf_hub_download(cfg.model_repo, "config.json", revision=cfg.model_revision)
    c = json.load(open(p))
    return c.get("text_config", c)


def tiny_text_config(cfg, *, window: int | None = None) -> Inkling2LoopTextConfig:
    """Real vocabulary, softcap, norms and conv kernel; tiny width/depth. 6 layers with blocks of 2 and a looped core
    2..4 x3 passes give 10 cache layers and all four decoder-layer classes (sliding/full x plain/loop)."""
    real = real_text_config(cfg)
    keep = {k: real[k] for k in ("vocab_size", "unpadded_vocab_size", "final_logit_softcapping", "rms_norm_eps",
                                 "qk_l2_clip_threshold", "rel_l2_clip_threshold", "sconv_kernel_size",
                                 "log_scaling_n_floor", "log_scaling_alpha", "use_value_norm_no_gamma",
                                 "attn_res_normalize_values", "mlp_gate_softcap", "mlp_up_softcap", "route_scale",
                                 "use_gate_bias", "q_bias", "o_bias", "d_rel") if k in real}
    w = window or 64
    return Inkling2LoopTextConfig(
        **keep, hidden_size=64, num_hidden_layers=6, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        swa_num_attention_heads=4, swa_num_key_value_heads=2, swa_head_dim=16, sliding_window_size=w,
        rel_extent=w, n_routed_experts=8, num_experts_per_tok=2, n_shared_experts=0, intermediate_size=32,
        n_layers_per_attn_res_block=2,
        layer_types=["hybrid_sliding", "hybrid", "hybrid_sliding", "hybrid", "hybrid_sliding", "hybrid"],
        loop_config={"core_start": 2, "core_end": 4, "num_passes": 3, "gate_rank": 4},
    )


def _wrap(cfg, model, tok, info, source=None) -> Bundle:
    model.requires_grad_(False)
    apply_patches()
    lcfg = LoraConfig(r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=0.0, bias="none",
                      target_modules=LORA_TARGET)
    pm = get_peft_model(model, lcfg)
    n_msg = unsloth_ops.checkpoint_message_inputs(pm.base_model.model)
    fwd = add_forward_bias_bank(pm)
    for p in fwd:
        p.requires_grad_(True)
    lora = [p for n, p in pm.named_parameters() if p.requires_grad and "lora_" in n]
    if not lora or not fwd:
        raise RuntimeError(f"nothing to train: {len(lora)} LoRA tensors, {len(fwd)} bias banks")
    if cfg.grad_ckpt:
        pm.base_model.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return Bundle(tok=tok, pm=pm, fwd=fwd, lora=lora, load_info=info, expert_source=source, message_ckpt=n_msg)


TINY_MXFP8_LAST = 2      # the release keeps its last 4 of 24 layers in MXFP8; the tiny model its last 2 of 6


def build_tiny(cfg, *, window: int | None = None, seed: int = 0) -> Bundle:
    torch.manual_seed(seed)
    tc = tiny_text_config(cfg, window=window)
    tc._attn_implementation = "eager"
    model = Inkling2LoopForCausalLM(tc).to(cfg.torch_dtype)
    source = None
    if cfg.experts_impl == quant.IMPL:
        # random dense experts -> the checkpoint's own layout (NVFP4 + MXFP8, interleaved gate/up) -> packed stacks
        source = quant.quantize_like_checkpoint(model, TINY_MXFP8_LAST)
        quant.register()
        for name, mod in quant.experts_modules(model):
            layer = int(name.split("layers.")[1].split(".")[0])
            quant.install(mod, quant.raw_layer(source, layer, "cpu"), cfg.torch_dtype)
    model.config._experts_implementation = cfg.experts_impl
    _set_experts_impl(model, cfg.experts_impl)
    tok = load_tokenizer(cfg)
    return _wrap(cfg, model.to(cfg.device), tok, None, source)


def _set_experts_impl(model, impl):
    if hasattr(model, "set_experts_implementation"):
        model.set_experts_implementation(impl)


def load_real(cfg) -> Bundle:
    tok = load_tokenizer(cfg)
    if cfg.experts_impl == quant.IMPL:
        model, info = quant.load_packed(cfg.model_repo, cfg.model_revision, cfg.device, cfg.torch_dtype)
        return _wrap(cfg, model, tok, info, info["checkpoint"])
    model, info = Inkling2LoopForCausalLM.from_pretrained(
        cfg.model_repo, revision=cfg.model_revision, dtype=cfg.torch_dtype, device_map={"": 0},
        attn_implementation="eager", experts_implementation=cfg.experts_impl, output_loading_info=True)
    return _wrap(cfg, model, tok, info)


def build(cfg, **kw) -> Bundle:
    return build_tiny(cfg, **kw) if cfg.tiny else load_real(cfg)


def experts_implementation(pm) -> str | None:
    lm = pm.base_model.model
    return getattr(lm.config, "_experts_implementation", None)
