"""Scaled dot-product and multi-head attention.

Implements section 3.2 of Vaswani et al. (2017):

    Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V                 (Eq. 1)

    MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
        where head_i = Attention(Q W^Q_i, K W^K_i, V W^V_i)
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class ScaledDotProductAttention(nn.Module):
    """Scaled dot-product attention (paper Eq. 1).

    The module is stateless apart from a dropout layer applied to the attention
    weights (as in the reference "Annotated Transformer" implementation).
    """

    def __init__(self, dropout: float = 0.0) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Compute attention.

        Args:
            query: Tensor of shape ``(..., q_len, d_k)``.
            key: Tensor of shape ``(..., k_len, d_k)``.
            value: Tensor of shape ``(..., k_len, d_v)``.
            mask: Optional boolean tensor broadcastable to
                ``(..., q_len, k_len)``. Positions that are ``False`` are
                masked out (set to ``-inf`` before the softmax).

        Returns:
            A tuple ``(output, attn)`` where ``output`` has shape
            ``(..., q_len, d_v)`` and ``attn`` (the attention weights) has shape
            ``(..., q_len, k_len)``.
        """
        d_k = query.size(-1)
        # (..., q_len, k_len)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        # Guard against rows that are fully masked (all -inf -> NaN softmax).
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = self.dropout(attn)
        output = torch.matmul(attn, value)
        return output, attn


class MultiHeadAttention(nn.Module):
    """Multi-head attention (paper section 3.2.2).

    The input is linearly projected into ``num_heads`` subspaces of dimension
    ``d_k = d_model / num_heads``, attention is applied independently per head,
    and the results are concatenated and projected back to ``d_model``.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by num_heads ({num_heads})."
            )
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # One projection per role; bias matches the paper's affine projections.
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention(dropout)
        self.dropout = nn.Dropout(dropout)
        self.last_attn: Tensor | None = None

    def _split_heads(self, x: Tensor) -> Tensor:
        """Reshape ``(B, T, d_model)`` into ``(B, num_heads, T, d_k)``."""
        batch, seq_len, _ = x.shape
        x = x.view(batch, seq_len, self.num_heads, self.d_k)
        return x.transpose(1, 2)

    def _merge_heads(self, x: Tensor) -> Tensor:
        """Reshape ``(B, num_heads, T, d_k)`` back into ``(B, T, d_model)``."""
        batch, _, seq_len, _ = x.shape
        x = x.transpose(1, 2).contiguous()
        return x.view(batch, seq_len, self.d_model)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Tensor | None = None,
    ) -> Tensor:
        """Apply multi-head attention.

        Args:
            query: Tensor of shape ``(B, q_len, d_model)``.
            key: Tensor of shape ``(B, k_len, d_model)``.
            value: Tensor of shape ``(B, k_len, d_model)``.
            mask: Optional boolean mask broadcastable to
                ``(B, num_heads, q_len, k_len)``. A common input shape is
                ``(B, 1, q_len, k_len)`` or ``(B, 1, 1, k_len)``.

        Returns:
            Tensor of shape ``(B, q_len, d_model)``.
        """
        q = self._split_heads(self.w_q(query))
        k = self._split_heads(self.w_k(key))
        v = self._split_heads(self.w_v(value))

        if mask is not None and mask.dim() == 3:
            # (B, q_len, k_len) -> (B, 1, q_len, k_len) so it broadcasts to heads.
            mask = mask.unsqueeze(1)

        out, attn = self.attention(q, k, v, mask=mask)
        self.last_attn = attn.detach()

        out = self._merge_heads(out)
        return self.dropout(self.w_o(out))
