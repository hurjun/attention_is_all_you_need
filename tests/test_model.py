"""Tests for the full Transformer, schedule and loss."""

from __future__ import annotations

import torch

from transformer import (
    LabelSmoothingLoss,
    Transformer,
    TransformerConfig,
    make_noam_scheduler,
    noam_lambda,
)


def _small_config(**kw: object) -> TransformerConfig:
    base = {
        "src_vocab_size": 50,
        "tgt_vocab_size": 50,
        "d_model": 64,
        "num_heads": 4,
        "num_encoder_layers": 2,
        "num_decoder_layers": 2,
        "d_ff": 128,
        "dropout": 0.0,
        "max_seq_len": 64,
    }
    base.update(kw)
    return TransformerConfig(**base)  # type: ignore[arg-type]


def test_forward_returns_logits_shape() -> None:
    cfg = _small_config()
    model = Transformer(cfg)
    src = torch.randint(3, 50, (3, 9))
    tgt = torch.randint(3, 50, (3, 7))
    out = model(src, tgt)
    assert out.shape == (3, 7, cfg.tgt_vocab_size)


def test_paper_size_parameter_count() -> None:
    # Base model with a shared BPE vocabulary of 37000 (paper section 5.1).
    cfg = TransformerConfig(
        src_vocab_size=37000, tgt_vocab_size=37000, tie_embeddings=True
    )
    model = Transformer(cfg)
    n = model.count_parameters()
    # Paper reports ~65M; with tied embeddings this implementation is ~63.1M.
    assert 60_000_000 < n < 66_000_000, n


def test_tied_embeddings_share_storage() -> None:
    cfg = _small_config(tie_embeddings=True)
    model = Transformer(cfg)
    assert model.tgt_embed.weight is model.src_embed.weight
    assert model.generator.weight is model.tgt_embed.weight


def test_untied_embeddings_are_separate() -> None:
    cfg = _small_config(tie_embeddings=False)
    model = Transformer(cfg)
    assert model.tgt_embed.weight is not model.src_embed.weight
    assert model.generator.weight is not model.tgt_embed.weight


def test_mask_builders_shapes() -> None:
    cfg = _small_config()
    model = Transformer(cfg)
    src = torch.randint(3, 50, (2, 6))
    tgt = torch.randint(3, 50, (2, 5))
    assert model.make_src_mask(src).shape == (2, 1, 1, 6)
    assert model.make_tgt_mask(tgt).shape == (2, 1, 5, 5)


def test_greedy_decode_shape_and_bos() -> None:
    cfg = _small_config()
    model = Transformer(cfg)
    src = torch.randint(3, 50, (4, 8))
    out = model.greedy_decode(src, max_len=10, bos_idx=1, eos_idx=2)
    assert out.size(0) == 4
    assert out.size(1) <= 11
    assert torch.all(out[:, 0] == 1)


def test_padding_does_not_change_unpadded_outputs() -> None:
    # Appending padding to the source must not alter logits at real positions.
    cfg = _small_config()
    model = Transformer(cfg).eval()
    src = torch.tensor([[3, 4, 5, 6]])
    tgt = torch.tensor([[1, 7, 8]])
    out = model(src, tgt)
    src_pad = torch.tensor([[3, 4, 5, 6, 0, 0]])
    out_pad = model(src_pad, tgt)
    assert torch.allclose(out, out_pad, atol=1e-5)


def test_noam_schedule_peaks_near_warmup() -> None:
    d_model, warmup = 512, 4000
    fn = noam_lambda(d_model, warmup)
    values = [fn(s) for s in range(0, 20000)]
    peak_step = max(range(len(values)), key=lambda s: values[s])
    # The schedule peaks at the warmup step (0-based index warmup - 1).
    assert abs(peak_step - (warmup - 1)) <= 1
    assert values[10] < values[warmup]  # rising during warmup
    assert values[8000] < values[warmup]  # decaying afterwards


def test_noam_scheduler_drives_optimizer_lr() -> None:
    cfg = _small_config()
    model = Transformer(cfg)
    opt = torch.optim.Adam(model.parameters(), lr=1.0)
    sched = make_noam_scheduler(opt, cfg.d_model, warmup_steps=100)
    lrs = []
    for _ in range(5):
        opt.step()
        sched.step()
        lrs.append(opt.param_groups[0]["lr"])
    # Learning rate increases during warmup.
    assert lrs == sorted(lrs)
    assert lrs[0] > 0


def test_label_smoothing_ignores_padding() -> None:
    vocab = 10
    loss_fn = LabelSmoothingLoss(vocab, pad_idx=0, smoothing=0.1)
    logits = torch.randn(2, 4, vocab)
    target = torch.tensor([[3, 4, 5, 0], [6, 7, 0, 0]])
    base = loss_fn(logits, target)
    # Changing logits only at padded target positions must not change the loss.
    logits2 = logits.clone()
    logits2[0, 3] += 5.0
    logits2[1, 2:] += 5.0
    assert torch.allclose(base, loss_fn(logits2, target), atol=1e-6)


def test_label_smoothing_small_when_confident() -> None:
    vocab = 10
    loss_fn = LabelSmoothingLoss(vocab, pad_idx=0, smoothing=0.0)
    target = torch.tensor([[3, 4]])
    logits = torch.full((1, 2, vocab), -10.0)
    logits[0, 0, 3] = 10.0
    logits[0, 1, 4] = 10.0
    assert float(loss_fn(logits, target)) < 1e-3
