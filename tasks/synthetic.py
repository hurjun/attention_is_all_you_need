"""Synthetic copy / reverse / sort tasks for sanity-checking the Transformer.

The model receives a random sequence of *symbol* tokens and must produce a
transformed version of it (identity, reversal, or ascending sort). Sequences
have variable length and are right-padded, so the tasks exercise both padding
masks and the causal decoder mask.

Token layout::

    0 -> <pad>   1 -> <bos>   2 -> <eos>   3 .. 3+num_symbols-1 -> symbols

Target sequences are framed for teacher forcing: the decoder input is
``[<bos>, y_1, ..., y_L]`` and the gold output is ``[y_1, ..., y_L, <eos>]``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

PAD_IDX = 0
BOS_IDX = 1
EOS_IDX = 2
NUM_SPECIAL = 3  # pad, bos, eos

Task = str  # one of "copy", "reverse", "sort"


@dataclass(frozen=True)
class SyntheticTaskConfig:
    """Parameters defining a synthetic task instance.

    Attributes:
        task: One of ``"copy"``, ``"reverse"`` or ``"sort"``.
        num_symbols: Number of distinct content symbols.
        min_len: Minimum sequence length (in symbols).
        max_len: Maximum sequence length (in symbols).
    """

    task: Task = "reverse"
    num_symbols: int = 16
    min_len: int = 6
    max_len: int = 12

    def __post_init__(self) -> None:
        if self.task not in ("copy", "reverse", "sort"):
            raise ValueError(f"Unknown task: {self.task!r}")
        if not 1 <= self.min_len <= self.max_len:
            raise ValueError("Require 1 <= min_len <= max_len.")


def vocab_size(cfg: SyntheticTaskConfig) -> int:
    """Total vocabulary size including special tokens."""
    return cfg.num_symbols + NUM_SPECIAL


def _transform(symbols: list[int], task: Task) -> list[int]:
    """Apply the task transformation to a list of symbol ids."""
    if task == "copy":
        return list(symbols)
    if task == "reverse":
        return list(reversed(symbols))
    if task == "sort":
        return sorted(symbols)
    raise ValueError(f"Unknown task: {task!r}")  # pragma: no cover


def make_batch(
    cfg: SyntheticTaskConfig,
    batch_size: int,
    generator: torch.Generator | None = None,
    device: torch.device | str | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Generate one batch of (source, decoder-input, gold-output) tensors.

    Args:
        cfg: Task configuration.
        batch_size: Number of examples in the batch.
        generator: Optional ``torch.Generator`` for reproducible sampling.
        device: Device on which to place the returned tensors.

    Returns:
        Tuple ``(src, tgt_in, tgt_out)``:

        * ``src``: ``(B, max_len)`` source symbols, right-padded.
        * ``tgt_in``: ``(B, max_len + 1)`` decoder input ``[<bos>, ...]``.
        * ``tgt_out``: ``(B, max_len + 1)`` gold output ``[..., <eos>]``.
    """
    lo = NUM_SPECIAL
    hi = NUM_SPECIAL + cfg.num_symbols  # exclusive

    lengths = torch.randint(
        cfg.min_len, cfg.max_len + 1, (batch_size,), generator=generator
    )

    src = torch.full((batch_size, cfg.max_len), PAD_IDX, dtype=torch.long)
    tgt_in = torch.full((batch_size, cfg.max_len + 1), PAD_IDX, dtype=torch.long)
    tgt_out = torch.full((batch_size, cfg.max_len + 1), PAD_IDX, dtype=torch.long)

    for i in range(batch_size):
        length = int(lengths[i])
        symbols = torch.randint(lo, hi, (length,), generator=generator).tolist()
        target = _transform(symbols, cfg.task)

        src[i, :length] = torch.tensor(symbols, dtype=torch.long)
        tgt_in[i, 0] = BOS_IDX
        tgt_in[i, 1 : length + 1] = torch.tensor(target, dtype=torch.long)
        tgt_out[i, :length] = torch.tensor(target, dtype=torch.long)
        tgt_out[i, length] = EOS_IDX

    if device is not None:
        src = src.to(device)
        tgt_in = tgt_in.to(device)
        tgt_out = tgt_out.to(device)
    return src, tgt_in, tgt_out


def decode_tokens(tokens: list[int]) -> str:
    """Render a token-id sequence as a human-readable string.

    Special tokens are shown symbolically and content symbols as integers
    starting at 0. Padding is omitted.
    """
    out: list[str] = []
    for t in tokens:
        if t == PAD_IDX:
            continue
        if t == BOS_IDX:
            out.append("<bos>")
        elif t == EOS_IDX:
            out.append("<eos>")
        else:
            out.append(str(t - NUM_SPECIAL))
    return " ".join(out)
