"""Label-smoothing loss via KL divergence (paper section 5.4).

Instead of a one-hot target, probability mass ``1 - epsilon`` is placed on the
correct class and ``epsilon`` is spread uniformly over the remaining classes.
The model's log-probabilities are then matched to this smoothed distribution
with the KL divergence, ignoring padding positions.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class LabelSmoothingLoss(nn.Module):
    """KL-divergence loss against a label-smoothed target distribution.

    Args:
        vocab_size: Number of target classes.
        pad_idx: Padding id; these target positions are excluded from the loss
            and never receive smoothed mass.
        smoothing: Smoothing epsilon (paper uses 0.1). ``0.0`` recovers the
            standard negative log-likelihood loss.
    """

    def __init__(self, vocab_size: int, pad_idx: int = 0, smoothing: float = 0.1) -> None:
        super().__init__()
        if not 0.0 <= smoothing < 1.0:
            raise ValueError("smoothing must be in [0, 1).")
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
        self.criterion = nn.KLDivLoss(reduction="sum")

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        """Compute the mean-per-token label-smoothed loss.

        Args:
            logits: Unnormalized scores of shape ``(B, T, vocab_size)``.
            target: Gold token ids of shape ``(B, T)``.

        Returns:
            Scalar loss averaged over non-pad target tokens.
        """
        logits = logits.reshape(-1, self.vocab_size)
        target = target.reshape(-1)
        log_probs = F.log_softmax(logits, dim=-1)

        # Build the smoothed target distribution.
        with torch.no_grad():
            true_dist = torch.full_like(log_probs, self.smoothing / (self.vocab_size - 2))
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
            true_dist[:, self.pad_idx] = 0.0
            pad_mask = target == self.pad_idx
            true_dist[pad_mask] = 0.0

        num_tokens = (~pad_mask).sum().clamp(min=1)
        return self.criterion(log_probs, true_dist) / num_tokens
