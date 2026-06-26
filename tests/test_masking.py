"""Tests for padding and causal (look-ahead) masks."""

from __future__ import annotations

import torch

from transformer.masking import (
    make_causal_mask,
    make_pad_mask,
    make_src_mask,
    make_tgt_mask,
)


def test_pad_mask_shape_and_values() -> None:
    seq = torch.tensor([[3, 4, 0, 0], [5, 6, 7, 0]])
    mask = make_pad_mask(seq, pad_idx=0)
    assert mask.shape == (2, 1, 1, 4)
    expected = torch.tensor([[True, True, False, False], [True, True, True, False]])
    assert torch.equal(mask[:, 0, 0], expected)


def test_causal_mask_is_lower_triangular() -> None:
    size = 5
    mask = make_causal_mask(size)
    assert mask.shape == (1, 1, size, size)
    m = mask[0, 0]
    # Strictly: position i may attend to j iff j <= i.
    for i in range(size):
        for j in range(size):
            assert bool(m[i, j]) == (j <= i)
    # No future leakage: everything strictly above the diagonal is masked.
    assert torch.equal(m, torch.tril(torch.ones(size, size, dtype=torch.bool)))
    assert m.triu(diagonal=1).sum() == 0


def test_tgt_mask_combines_pad_and_causal() -> None:
    tgt = torch.tensor([[1, 5, 6, 0]])  # last token is padding
    mask = make_tgt_mask(tgt, pad_idx=0)
    assert mask.shape == (1, 1, 4, 4)
    m = mask[0, 0]
    # Future positions are always blocked.
    assert m.triu(diagonal=1).sum() == 0
    # The padded key column (index 3) is never attended to.
    assert bool(m[:, 3].any()) is False
    # A valid past, non-pad key is attended to.
    assert bool(m[2, 0]) is True


def test_src_mask_matches_pad_mask() -> None:
    src = torch.tensor([[3, 0, 4]])
    assert torch.equal(make_src_mask(src), make_pad_mask(src))
