"""Decoder stack (paper section 3.1, right half of Figure 1).

Each :class:`DecoderLayer` has three sub-layers: masked multi-head self-attention
over the (shifted) target, multi-head cross-attention over the encoder output,
and a position-wise feed-forward network. Each sub-layer is wrapped in a residual
connection and layer normalization.
"""

from __future__ import annotations

from torch import Tensor, nn

from .attention import MultiHeadAttention
from .feed_forward import PositionwiseFeedForward


class DecoderLayer(nn.Module):
    """A single Transformer decoder layer."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        norm_first: bool = False,
    ) -> None:
        super().__init__()
        self.norm_first = norm_first
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        memory: Tensor,
        tgt_mask: Tensor | None = None,
        memory_mask: Tensor | None = None,
    ) -> Tensor:
        """Run one decoder layer.

        Args:
            x: Target representations of shape ``(B, T, d_model)``.
            memory: Encoder output of shape ``(B, S, d_model)``.
            tgt_mask: Combined causal + padding mask, shape ``(B, 1, T, T)``.
            memory_mask: Source padding mask, shape ``(B, 1, 1, S)``.

        Returns:
            Tensor of shape ``(B, T, d_model)``.
        """
        if self.norm_first:
            nx = self.norm1(x)
            x = x + self.self_attn(nx, nx, nx, tgt_mask)
            nx = self.norm2(x)
            x = x + self.cross_attn(nx, memory, memory, memory_mask)
            x = x + self.feed_forward(self.norm3(x))
        else:
            x = self.norm1(x + self.self_attn(x, x, x, tgt_mask))
            x = self.norm2(x + self.cross_attn(x, memory, memory, memory_mask))
            x = self.norm3(x + self.feed_forward(x))
        return x


class Decoder(nn.Module):
    """Stack of ``num_layers`` identical decoder layers."""

    def __init__(
        self,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        norm_first: bool = False,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            DecoderLayer(d_model, num_heads, d_ff, dropout, norm_first)
            for _ in range(num_layers)
        )
        self.norm = nn.LayerNorm(d_model) if norm_first else None

    def forward(
        self,
        x: Tensor,
        memory: Tensor,
        tgt_mask: Tensor | None = None,
        memory_mask: Tensor | None = None,
    ) -> Tensor:
        """Decode ``(B, T, d_model)`` against ``memory`` -> ``(B, T, d_model)``."""
        for layer in self.layers:
            x = layer(x, memory, tgt_mask, memory_mask)
        if self.norm is not None:
            x = self.norm(x)
        return x
