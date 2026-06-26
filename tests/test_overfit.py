"""Overfit-one-batch test: proves the model can drive loss toward zero.

This is the strongest cheap correctness signal - if the gradients, masking,
embeddings and loss are wired correctly, a tiny model memorizes a single fixed
batch perfectly within a couple hundred steps on CPU.
"""

from __future__ import annotations

import torch

from tasks.synthetic import PAD_IDX, SyntheticTaskConfig, make_batch
from transformer import LabelSmoothingLoss, Transformer, TransformerConfig


def test_overfit_single_batch() -> None:
    torch.manual_seed(0)
    task = SyntheticTaskConfig(task="reverse", num_symbols=12, min_len=5, max_len=8)
    vocab = task.num_symbols + 3

    cfg = TransformerConfig(
        src_vocab_size=vocab,
        tgt_vocab_size=vocab,
        d_model=64,
        num_heads=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        d_ff=128,
        dropout=0.0,  # determinism for overfitting
        max_seq_len=32,
        warmup_steps=50,
    )
    model = Transformer(cfg)
    # No label smoothing so the loss floor is 0 and we can assert it drops hard.
    criterion = LabelSmoothingLoss(vocab, pad_idx=PAD_IDX, smoothing=0.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9)

    gen = torch.Generator().manual_seed(123)
    src, tgt_in, tgt_out = make_batch(task, batch_size=8, generator=gen)

    model.train()
    first_loss = None
    last_loss = None
    for _ in range(300):
        logits = model(src, tgt_in)
        loss = criterion(logits, tgt_out)
        if first_loss is None:
            first_loss = loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        last_loss = loss.item()

    assert first_loss is not None and last_loss is not None
    # Loss collapses by well over an order of magnitude.
    assert last_loss < 0.05, f"loss did not collapse: {first_loss:.3f} -> {last_loss:.3f}"

    # Teacher-forced predictions are perfect on the memorized batch.
    model.eval()
    with torch.no_grad():
        preds = model(src, tgt_in).argmax(dim=-1)
    mask = tgt_out != PAD_IDX
    accuracy = (preds.eq(tgt_out) & mask).sum().item() / mask.sum().item()
    assert accuracy == 1.0, f"token accuracy {accuracy} != 1.0"
