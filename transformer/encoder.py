"""Encoder stack (paper section 3.1, left half of Figure 1).

Each :class:`EncoderLayer` has two sub-layers - multi-head self-attention and a
position-wise feed-forward network - each wrapped in a residual connection
followed (or preceded, for pre-norm) by layer normalization.
"""

from __future__ import annotations

from torch import Tensor, nn

from .attention import MultiHeadAttention
from .feed_forward import PositionwiseFeedForward


class EncoderLayer(nn.Module):
    """A single Transformer encoder layer.

    Args:
        d_model: Model dimension.
        num_heads: Number of attention heads.
        d_ff: Feed-forward inner dimension.
        dropout: Dropout probability.
        norm_first: Use pre-norm if True, post-norm (paper) if False.
    """

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
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, src_mask: Tensor | None = None) -> Tensor:
        """Run one encoder layer.

        Args:
            x: Input of shape ``(B, S, d_model)``.
            src_mask: Source padding mask broadcastable to ``(B, 1, S, S)``.

        Returns:
            Tensor of shape ``(B, S, d_model)``.
        """
        if self.norm_first:
            x = x + self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x), src_mask)
            x = x + self.feed_forward(self.norm2(x))
        else:
            x = self.norm1(x + self.self_attn(x, x, x, src_mask))
            x = self.norm2(x + self.feed_forward(x))
        return x


class Encoder(nn.Module):
    """Stack of ``num_layers`` identical encoder layers."""

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
            EncoderLayer(d_model, num_heads, d_ff, dropout, norm_first)
            for _ in range(num_layers)
        )
        # Final norm is needed for pre-norm; harmless (identity-initialized) for
        # post-norm where each sublayer already normalizes its output.
        self.norm = nn.LayerNorm(d_model) if norm_first else None

    def forward(self, x: Tensor, src_mask: Tensor | None = None) -> Tensor:
        """Encode ``(B, S, d_model)`` -> ``(B, S, d_model)``."""
        for layer in self.layers:
            x = layer(x, src_mask)
        if self.norm is not None:
            x = self.norm(x)
        return x
