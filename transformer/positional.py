"""Sinusoidal positional encoding (paper section 3.5).

    PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))

The table is precomputed once and stored with ``register_buffer`` so it moves
with the module (``.to(device)``) and is saved in the state dict, but is *not* a
learnable parameter.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class PositionalEncoding(nn.Module):
    """Add fixed sinusoidal positional encodings to token embeddings."""

    pe: Tensor  # populated by register_buffer; declared for type checkers.

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # position: (max_len, 1); div_term: (d_model / 2,)
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """Add positional encodings.

        Args:
            x: Token embeddings of shape ``(B, T, d_model)``.

        Returns:
            Tensor of the same shape with positional information added and
            dropout applied.
        """
        seq_len = x.size(1)
        if seq_len > self.pe.size(1):
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_len {self.pe.size(1)}."
            )
        x = x + self.pe[:, :seq_len]
        return self.dropout(x)
