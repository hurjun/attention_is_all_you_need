"""Attention mask construction.

Two kinds of masks are used (paper sections 3.1 / 3.2.3):

* **Padding mask** - hides ``<pad>`` tokens so they never contribute to
  attention. Built from the token ids.
* **Causal / look-ahead mask** - prevents a position in the decoder from
  attending to subsequent positions, preserving the auto-regressive property.

Throughout this project a mask is a *boolean* tensor where ``True`` means
"attend to this position" and ``False`` means "mask out". This is the opposite
sign convention to the ``-1e9`` additive masks sometimes seen, and is applied in
:class:`~transformer.attention.ScaledDotProductAttention` via ``masked_fill``.
"""

from __future__ import annotations

import torch
from torch import Tensor


def make_pad_mask(seq: Tensor, pad_idx: int = 0) -> Tensor:
    """Build a key-padding mask.

    Args:
        seq: Token ids of shape ``(B, T)``.
        pad_idx: Padding token id.

    Returns:
        Boolean tensor of shape ``(B, 1, 1, T)`` where ``True`` marks real
        (non-pad) tokens. The singleton dims broadcast over heads and query
        positions.
    """
    return (seq != pad_idx).unsqueeze(1).unsqueeze(2)


def make_causal_mask(size: int, device: torch.device | None = None) -> Tensor:
    """Build a lower-triangular causal mask.

    Args:
        size: Sequence length ``T``.
        device: Device on which to allocate the mask.

    Returns:
        Boolean tensor of shape ``(1, 1, T, T)`` that is ``True`` on and below
        the main diagonal (positions a query may attend to) and ``False`` above
        it (future positions).
    """
    full = torch.ones(size, size, dtype=torch.bool, device=device)
    causal = torch.tril(full)
    return causal.unsqueeze(0).unsqueeze(0)


def make_src_mask(src: Tensor, pad_idx: int = 0) -> Tensor:
    """Encoder self-attention mask: padding only.

    Args:
        src: Source token ids of shape ``(B, S)``.
        pad_idx: Padding token id.

    Returns:
        Boolean tensor of shape ``(B, 1, 1, S)``.
    """
    return make_pad_mask(src, pad_idx)


def make_tgt_mask(tgt: Tensor, pad_idx: int = 0) -> Tensor:
    """Decoder self-attention mask: padding AND look-ahead combined.

    Args:
        tgt: Target token ids of shape ``(B, T)``.
        pad_idx: Padding token id.

    Returns:
        Boolean tensor of shape ``(B, 1, T, T)`` that is ``True`` only where a
        position is both non-pad and not in the future.
    """
    pad_mask = make_pad_mask(tgt, pad_idx)  # (B, 1, 1, T)
    causal_mask = make_causal_mask(tgt.size(1), tgt.device)  # (1, 1, T, T)
    return pad_mask & causal_mask
