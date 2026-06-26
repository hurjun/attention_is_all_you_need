"""Tests for scaled dot-product and multi-head attention."""

from __future__ import annotations

import torch

from transformer.attention import MultiHeadAttention, ScaledDotProductAttention


def test_scaled_dot_product_output_shapes() -> None:
    b, h, q_len, k_len, d_k, d_v = 2, 4, 5, 7, 16, 16
    attn = ScaledDotProductAttention()
    q = torch.randn(b, h, q_len, d_k)
    k = torch.randn(b, h, k_len, d_k)
    v = torch.randn(b, h, k_len, d_v)
    out, weights = attn(q, k, v)
    assert out.shape == (b, h, q_len, d_v)
    assert weights.shape == (b, h, q_len, k_len)


def test_attention_weights_sum_to_one() -> None:
    attn = ScaledDotProductAttention()
    q = torch.randn(2, 3, 4, 8)
    k = torch.randn(2, 3, 6, 8)
    v = torch.randn(2, 3, 6, 8)
    _, weights = attn(q, k, v)
    sums = weights.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_attention_mask_zeros_out_masked_positions() -> None:
    attn = ScaledDotProductAttention()
    q = torch.randn(1, 1, 3, 8)
    k = torch.randn(1, 1, 4, 8)
    v = torch.randn(1, 1, 4, 8)
    # Mask out the last key position for every query.
    mask = torch.ones(1, 1, 3, 4, dtype=torch.bool)
    mask[..., -1] = False
    _, weights = attn(q, k, v, mask=mask)
    assert torch.allclose(weights[..., -1], torch.zeros(1, 1, 3))
    sums = weights.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_multihead_output_shape() -> None:
    mha = MultiHeadAttention(d_model=32, num_heads=4, dropout=0.0)
    x = torch.randn(3, 9, 32)
    out = mha(x, x, x)
    assert out.shape == (3, 9, 32)


def test_split_merge_heads_roundtrip() -> None:
    mha = MultiHeadAttention(d_model=32, num_heads=4, dropout=0.0)
    x = torch.randn(2, 6, 32)
    split = mha._split_heads(x)
    assert split.shape == (2, 4, 6, 8)
    merged = mha._merge_heads(split)
    assert merged.shape == x.shape
    assert torch.allclose(merged, x, atol=1e-6)


def test_multihead_requires_divisible_dims() -> None:
    import pytest

    with pytest.raises(ValueError):
        MultiHeadAttention(d_model=30, num_heads=4)


def test_multihead_accepts_4d_mask() -> None:
    mha = MultiHeadAttention(d_model=16, num_heads=2, dropout=0.0)
    x = torch.randn(2, 5, 16)
    mask = torch.ones(2, 1, 1, 5, dtype=torch.bool)
    out = mha(x, x, x, mask=mask)
    assert out.shape == (2, 5, 16)
