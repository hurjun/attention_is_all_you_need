"""Position-wise feed-forward network (paper section 3.3).

    FFN(x) = max(0, x W_1 + b_1) W_2 + b_2
"""

from __future__ import annotations

from torch import Tensor, nn


class PositionwiseFeedForward(nn.Module):
    """Two linear transformations with a ReLU in between, applied per position."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        """Apply the feed-forward network to ``(B, T, d_model)`` -> ``(B, T, d_model)``."""
        return self.linear2(self.dropout(self.activation(self.linear1(x))))
