"""Tests for the synthetic data generator."""

from __future__ import annotations

import torch

from tasks.synthetic import (
    BOS_IDX,
    EOS_IDX,
    PAD_IDX,
    SyntheticTaskConfig,
    decode_tokens,
    make_batch,
    vocab_size,
)


def test_batch_shapes() -> None:
    cfg = SyntheticTaskConfig(task="reverse", num_symbols=10, min_len=3, max_len=7)
    gen = torch.Generator().manual_seed(0)
    src, tgt_in, tgt_out = make_batch(cfg, batch_size=8, generator=gen)
    assert src.shape == (8, 7)
    assert tgt_in.shape == (8, 8)
    assert tgt_out.shape == (8, 8)


def test_reverse_relationship_and_special_tokens() -> None:
    cfg = SyntheticTaskConfig(task="reverse", num_symbols=10, min_len=4, max_len=6)
    gen = torch.Generator().manual_seed(1)
    src, tgt_in, tgt_out = make_batch(cfg, batch_size=16, generator=gen)
    for i in range(src.size(0)):
        symbols = [t for t in src[i].tolist() if t != PAD_IDX]
        # Decoder input starts with <bos>.
        assert tgt_in[i, 0].item() == BOS_IDX
        gold = [t for t in tgt_out[i].tolist() if t != PAD_IDX]
        # Gold ends with <eos> and equals reversed source plus <eos>.
        assert gold[-1] == EOS_IDX
        assert gold[:-1] == list(reversed(symbols))


def test_sort_task() -> None:
    cfg = SyntheticTaskConfig(task="sort", num_symbols=10, min_len=4, max_len=6)
    gen = torch.Generator().manual_seed(2)
    src, _, tgt_out = make_batch(cfg, batch_size=8, generator=gen)
    for i in range(src.size(0)):
        symbols = [t for t in src[i].tolist() if t != PAD_IDX]
        gold = [t for t in tgt_out[i].tolist() if t not in (PAD_IDX, EOS_IDX)]
        assert gold == sorted(symbols)


def test_vocab_size_and_decode() -> None:
    cfg = SyntheticTaskConfig(num_symbols=16)
    assert vocab_size(cfg) == 16 + 3
    assert decode_tokens([BOS_IDX, 5, 6, EOS_IDX, PAD_IDX]) == "<bos> 2 3 <eos>"


def test_reproducible_with_generator() -> None:
    cfg = SyntheticTaskConfig()
    a = make_batch(cfg, 4, generator=torch.Generator().manual_seed(42))
    b = make_batch(cfg, 4, generator=torch.Generator().manual_seed(42))
    assert torch.equal(a[0], b[0])
    assert torch.equal(a[2], b[2])
