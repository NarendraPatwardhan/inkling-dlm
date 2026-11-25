"""Block-wise low-confidence remasking on top of the frozen tower (batch 1)."""
from __future__ import annotations

import torch

from .constants import EOS_ID, END_MSG_ID, MASK_ID, S
from .loopcd import guided_logits
from .tower import tower_fill, view_from_inference_cache


def per_step(n: int, steps: int) -> list[int]:
    base, extra = divmod(n, steps)
    return [base + (1 if i < extra else 0) for i in range(steps)]


@torch.no_grad()
def generate(peft_model, prompt_ids, max_blocks=8, steps_per_block=32, block=S, temperature=0.0, generator=None,
             trace=None, loopcd=None):
    """prompt_ids ends with <|message_model|><|content_text|>; its last token is the first anchor.
    block <= 32 (sampling blocks larger than the training block collapse). An end token may be committed only at the
    leftmost masked slot. If `trace` is a list, per-step (block, step, committed slots) records are appended.
    loopcd: a loopcd.Spec; its guided logits pick both the token and the confidence that orders unmasking (temperature
    applies after guidance, paper App. A.3). None or off at full depth is the plain path."""
    assert 1 <= block <= S
    dev = next(peft_model.parameters()).device
    ids = torch.tensor([list(prompt_ids)], device=dev)
    # record_past keeps every sliding-layer key: the anchor slot at position n needs key n-(window-1), which a
    # plain sliding cache (window-1 keys) has already dropped once the context exceeds the window.
    cache = tower_fill(peft_model, ids, record_past=True)
    n, out = ids.shape[1], []
    for bi in range(max_blocks):
        view = view_from_inference_cache(peft_model, cache, n)
        x = torch.full((1, S + 1), MASK_ID, device=dev)
        x[0, 0] = out[-1] if out else prompt_ids[-1]            # anchor = last committed token
        live = torch.zeros(S, dtype=torch.bool, device=dev)
        live[:block] = True                                      # slots beyond `block` stay masked and unused
        idx = torch.arange(S, device=dev)
        for si, k in enumerate(per_step(block, min(steps_per_block, block))):
            logits = guided_logits(peft_model, view, x, loopcd)[0]   # (S, V): row j predicts block slot j
            if temperature > 0:
                probs = torch.softmax(logits / temperature, -1)
                pred = torch.multinomial(probs, 1, generator=generator)[:, 0]
            else:
                probs = torch.softmax(logits, -1)
                pred = probs.argmax(-1)
            conf = probs.gather(-1, pred[:, None])[:, 0]
            masked = (x[0, 1:] == MASK_ID) & live
            if not masked.any():
                break
            conf = torch.where(masked, conf, torch.full_like(conf, -1.0))
            first = int(masked.nonzero()[0])
            is_end = (pred == EOS_ID) | (pred == END_MSG_ID)
            conf = torch.where(is_end & (idx != first) & masked, torch.full_like(conf, -0.5), conf)
            take = conf.topk(k).indices
            take = take[masked[take]]
            x[0, 1 + take] = pred[take]
            if trace is not None:
                trace.append((bi, si, sorted(take.tolist())))
        blk = x[0, 1:1 + block].tolist()
        stop = next((i for i, t in enumerate(blk) if t in (EOS_ID, END_MSG_ID)), None)
        if stop is not None:
            out += blk[:stop + 1]
            break
        out += blk
        # commit: extend the frozen tower with the block (adapters disabled), positions n..n+block-1
        pos = torch.arange(n, n + block, device=dev)[None]
        tower_fill(peft_model, torch.tensor([blk], device=dev), cache=cache, position_ids=pos)
        n += block
    return out
