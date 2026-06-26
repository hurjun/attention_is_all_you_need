"""Tests for sinusoidal positional encoding."""

from __future__ import annotations

import math

import torch

from transformer.positional import PositionalEncoding


def test_pe_buffer_shape() -> None:
    pe = PositionalEncoding(d_model=16, dropout=0.0, max_len=50)
    assert pe.pe.shape == (1, 50, 16)


def test_pe_is_a_registered_buffer_not_a_parameter() -> None:
    pe = PositionalEncoding(d_model=16, dropout=0.0, max_len=10)
    assert "pe" in dict(pe.named_buffers())
    assert "pe" not in dict(pe.named_parameters())


def test_pe_values_match_formula() -> None:
    d_model = 8
    pe = PositionalEncoding(d_model=d_model, dropout=0.0, max_len=20)
    table = pe.pe[0]  # (max_len, d_model)
    # Position 0: sin(0)=0 on even indices, cos(0)=1 on odd indices.
    assert torch.allclose(table[0, 0::2], torch.zeros(d_model // 2), atol=1e-6)
    assert torch.allclose(table[0, 1::2], torch.ones(d_model // 2), atol=1e-6)
    # Spot-check a couple of explicit entries.
    pos = 3
    assert math.isclose(table[pos, 0].item(), math.sin(pos / 1.0), abs_tol=1e-5)
    div = 10000.0 ** (2 / d_model)
    assert math.isclose(table[pos, 2].item(), math.sin(pos / div), abs_tol=1e-5)
    assert math.isclose(table[pos, 3].item(), math.cos(pos / div), abs_tol=1e-5)


def test_pe_forward_preserves_shape_and_adds() -> None:
    pe = PositionalEncoding(d_model=16, dropout=0.0, max_len=50)
    x = torch.zeros(2, 5, 16)
    out = pe(x)
    assert out.shape == x.shape
    # With zero input the output equals the positional table slice.
    assert torch.allclose(out, pe.pe[:, :5].expand_as(out), atol=1e-6)


def test_pe_rejects_too_long_sequences() -> None:
    import pytest

    pe = PositionalEncoding(d_model=8, dropout=0.0, max_len=4)
    with pytest.raises(ValueError):
        pe(torch.zeros(1, 5, 8))
