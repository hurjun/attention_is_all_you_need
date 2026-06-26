"""Self-contained synthetic sequence-to-sequence tasks.

These tasks need no downloads and let a small Transformer demonstrably learn on
CPU in minutes. See :mod:`tasks.synthetic`.
"""

from __future__ import annotations

from .synthetic import (
    BOS_IDX,
    EOS_IDX,
    PAD_IDX,
    SyntheticTaskConfig,
    decode_tokens,
    make_batch,
    vocab_size,
)

__all__ = [
    "BOS_IDX",
    "EOS_IDX",
    "PAD_IDX",
    "SyntheticTaskConfig",
    "decode_tokens",
    "make_batch",
    "vocab_size",
]
