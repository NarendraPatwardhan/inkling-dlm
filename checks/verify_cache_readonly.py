"""PrefixView never writes; the tower states are unchanged by canvas forwards (G-readonly)."""
import torch

from inkling_dlm.data import collate
from inkling_dlm.evaluate import tower_states
from inkling_dlm.gates import g_readonly
from inkling_dlm.patches import PrefixView
from inkling_dlm.tower import canvas_logits, geometry, view_for_batch


def check_gate_readonly(cfg, bundle, groups):
    r = g_readonly(bundle, cfg, groups)
    assert r["passed"], r["detail"]
    assert r["metrics"]["reads"] == 2 * geometry(bundle.pm)["n_layers"]   # 2 forwards x one read per cache layer (loop copies too)


def check_prefixview_update_concatenates_without_storing():
    k = [torch.randn(2, 1, 5, 4)]
    seeds = [[torch.randn(3, 3, 3) for _ in range(4)]]          # per canvas (Bc=3), C=3, kernel-1=3
    v = PrefixView(k, k, seeds, torch.tensor([1, 1, 0]), {}, {}, None, n_layers=1)
    new = torch.randn(3, 1, 2, 4)
    kk, vv = v.update(new, new, 0)
    assert kk.shape == (3, 1, 7, 4) and torch.equal(kk[:2, :, :5], k[0][[1, 1]]) and torch.equal(kk[:, :, 5:], new)
    c = v.update_conv_state(torch.randn(3, 3, 6), 0, state_idx=2)
    assert c.shape == (3, 3, 9)
    assert v.get_conv_state(0, 1) is None and len(v.layers) == 1
    assert k[0].shape == (2, 1, 5, 4)                         # nothing appended


def check_canvas_forward_under_grad_leaves_states_frozen(cfg, bundle, groups):
    b = collate(groups[0], 5, cfg.seed, cfg.device, cfg.block)
    k, v, s = tower_states(bundle.pm, b)
    assert not any(t.requires_grad for t in k + v)
    idx = torch.arange(4)
    view = view_for_batch(bundle.pm, k, v, s, b, idx)
    out = canvas_logits(bundle.pm, view, b["canvas"][idx])
    out.float().sum().backward()
    assert all(t.grad is None for t in k + v)
    bundle.pm.zero_grad(set_to_none=True)
