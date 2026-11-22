"""Mask and relative-distance geometry of the canvas view (patches._geometry / build_view)."""
import torch

from inkling_dlm.constants import S
from inkling_dlm.patches import _geometry, build_view

NEG = torch.finfo(torch.float32).min


def _allowed(mask):
    return mask[:, 0] == 0                                    # (Bc, Q, K) bool


def check_canvas_bidirectional_and_anchor_hidden():
    T = 40
    kv_valid = torch.ones(1, T, dtype=torch.bool)
    anchors = torch.tensor([T - 1])
    m, rel = _geometry(kv_valid, torch.arange(T), anchors, None, False, torch.float32)
    a = _allowed(m)[0]
    assert m.shape == (1, 1, S + 1, T + S + 1) and rel.shape == m.shape
    assert a[:, :T].all()                                     # every slot sees the whole valid prefix
    assert not a[:, T].any()                                  # the anchor's canvas key is hidden from everyone
    assert a[:, T + 1:].all()                                 # block slots are bidirectional


def check_causal_canvas_is_lower_triangular():
    T = 10
    m, _ = _geometry(torch.ones(1, T, dtype=torch.bool), torch.arange(T), torch.tensor([T - 1]), None, True,
                     torch.float32)
    c = _allowed(m)[0, :, T:]
    want = torch.ones(S + 1, S + 1, dtype=torch.bool).tril()
    want[:, 0] = False
    assert torch.equal(c, want)


def check_kv_valid_respected_and_relative_distance():
    T = 20
    kv_valid = torch.zeros(2, T, dtype=torch.bool)
    kv_valid[0, :8] = True
    kv_valid[1, :15] = True
    anchors = torch.tensor([7, 14])
    m, rel = _geometry(kv_valid, torch.arange(T), anchors, None, False, torch.float32)
    a = _allowed(m)
    assert torch.equal(a[0, :, :T], kv_valid[0].expand(S + 1, T))
    assert torch.equal(a[1, :, :T], kv_valid[1].expand(S + 1, T))
    for i, n in enumerate(anchors.tolist()):                  # rel = q - k with q = n + j, keys at 0..T-1 then canvas
        q = n + torch.arange(S + 1)
        k = torch.cat([torch.arange(T), q])
        assert torch.equal(rel[i, 0], q[:, None] - k[None])


def check_sliding_window_cut():
    T, W = 100, 16
    m, rel = _geometry(torch.ones(1, T, dtype=torch.bool), torch.arange(T), torch.tensor([T - 1]), W, False,
                       torch.float32)
    a = _allowed(m)[0]
    assert not (a & (rel[0, 0] >= W)).any()                   # nothing at distance >= window
    assert a[0, T - W:T].all() and not a[0, :T - W].any()     # the anchor slot sees exactly the last W positions


def check_build_view_positions_and_sliding_slice():
    T = 30
    keys = [torch.zeros(1, 1, T, 4)]
    view = build_view(keys, keys, [[torch.zeros(1, 2, 3)] * 4], torch.tensor([0]),
                      torch.ones(1, T, dtype=torch.bool), torch.tensor([T - 1]), sliding_len=12, window=8,
                      n_layers=1, dtype=torch.float32)
    assert torch.equal(view.position_ids, (T - 1 + torch.arange(S + 1))[None])
    assert view.masks["full_attention"].shape[-1] == T + S + 1
    assert view.masks["sliding_attention"].shape[-1] == 12 + S + 1
    assert view.rel["sliding"][0, 0, 0, 0] == (T - 1) - (T - 12)
