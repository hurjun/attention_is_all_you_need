"""Real, reproducible CPU experiments for the from-scratch Transformer.

Two experiments, both seeded and fast enough to run on a laptop CPU in a couple
of minutes:

(a) **Attention visualization.** Train the small model on the synthetic
    ``reverse`` task, run a single held-out example through it, and pull the
    *real* attention weights cached on every
    :class:`~transformer.attention.MultiHeadAttention` (``last_attn``). The
    encoder self-attention and the decoder->encoder cross-attention are saved as
    a heatmap (``assets/attention_maps.png``). For ``reverse`` the cross-
    attention is expected to concentrate on an anti-diagonal (output position
    ``t`` reads source position ``L-t``).

(b) **Positional-encoding ablation.** Train the *same* model twice on
    ``reverse`` with identical seeds/data -- once with sinusoidal positional
    encoding and once with it removed (the ``pos_encoding`` module is swapped for
    a no-op that only applies dropout, so no library code is changed). Without
    positional information the encoder is permutation-invariant, so ``reverse``
    (which is purely positional) should collapse. The measured accuracy gap is
    written to ``assets/pe_ablation.png`` and printed as a table.

Nothing here is mocked: every number comes from the run you launch, and every
figure is drawn from tensors produced by that run.

Example::

    python scripts/experiments.py --steps 1200 --device cpu
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch
from torch import Tensor, nn

# Make the repository root importable when run as a script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.train_synthetic import (  # noqa: E402
    evaluate,
    resolve_device,
)
from tasks.synthetic import (  # noqa: E402
    BOS_IDX,
    EOS_IDX,
    PAD_IDX,
    SyntheticTaskConfig,
    make_batch,
    vocab_size,
)
from transformer import (  # noqa: E402
    LabelSmoothingLoss,
    Transformer,
    TransformerConfig,
    make_noam_scheduler,
)


class _NoPositionalEncoding(nn.Module):
    """Drop-in replacement for the ablation: adds **no** positional signal.

    Matches the :class:`~transformer.positional.PositionalEncoding` interface
    (it still applies dropout) but never adds the sinusoidal table, so the model
    sees a permutation-invariant bag of token embeddings.
    """

    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(x)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", default="reverse", choices=["copy", "reverse", "sort"])
    p.add_argument("--num-symbols", type=int, default=16)
    p.add_argument("--min-len", type=int, default=6)
    p.add_argument("--max-len", type=int, default=12)
    # Small model so each run is fast on CPU.
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--d-ff", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--steps", type=int, default=1200)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--eval-batches", type=int, default=16)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--device", default="cpu", choices=["auto", "cpu", "mps", "cuda"])
    p.add_argument("--attn-figure", default="assets/attention_maps.png")
    p.add_argument("--ablation-figure", default="assets/pe_ablation.png")
    return p.parse_args()


def build_model(
    args: argparse.Namespace, vocab: int, use_positional_encoding: bool
) -> Transformer:
    """Build the small model; optionally strip positional encoding (ablation)."""
    cfg = TransformerConfig(
        src_vocab_size=vocab,
        tgt_vocab_size=vocab,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_encoder_layers=args.num_layers,
        num_decoder_layers=args.num_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        max_seq_len=args.max_len + 8,
        pad_idx=PAD_IDX,
        tie_embeddings=True,
        label_smoothing=args.label_smoothing,
        warmup_steps=args.warmup,
    )
    model = Transformer(cfg)
    if not use_positional_encoding:
        # Swap in a no-op so the encoder/decoder receive no order information.
        model.pos_encoding = _NoPositionalEncoding(args.dropout)
    return model


def train_model(
    args: argparse.Namespace,
    task_cfg: SyntheticTaskConfig,
    vocab: int,
    device: torch.device,
    use_positional_encoding: bool,
) -> tuple[Transformer, float, float, float]:
    """Train one model from a fixed seed and return (model, tok_acc, seq_acc, secs).

    Both ablation arms call this with the same ``args.seed``, so weight
    initialization and the training-data stream are identical; the only
    difference is whether positional encoding is present.
    """
    torch.manual_seed(args.seed)  # identical init across arms
    model = build_model(args, vocab, use_positional_encoding).to(device)

    criterion = LabelSmoothingLoss(vocab, pad_idx=PAD_IDX, smoothing=args.label_smoothing)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1.0, betas=model.config.adam_betas, eps=model.config.adam_eps
    )
    scheduler = make_noam_scheduler(optimizer, model.config.d_model, args.warmup)
    train_gen = torch.Generator().manual_seed(args.seed)  # identical data stream

    start = time.time()
    for _ in range(1, args.steps + 1):
        model.train()
        src, tgt_in, tgt_out = make_batch(task_cfg, args.batch_size, generator=train_gen)
        src, tgt_in, tgt_out = src.to(device), tgt_in.to(device), tgt_out.to(device)
        logits = model(src, tgt_in)
        loss = criterion(logits, tgt_out)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
    elapsed = time.time() - start

    eval_gen = torch.Generator().manual_seed(args.seed + 777)
    tok_acc, seq_acc = evaluate(
        model, task_cfg, args.batch_size, args.eval_batches, eval_gen, device
    )
    return model, tok_acc, seq_acc, elapsed


def _token_labels(ids: list[int]) -> list[str]:
    """Compact per-token labels for heatmap ticks."""
    labels: list[str] = []
    for t in ids:
        if t == PAD_IDX:
            labels.append("·")
        elif t == BOS_IDX:
            labels.append("<s>")
        elif t == EOS_IDX:
            labels.append("</s>")
        else:
            labels.append(str(t - 3))
    return labels


def save_attention_maps(
    model: Transformer,
    task_cfg: SyntheticTaskConfig,
    device: torch.device,
    seed: int,
    path: str,
) -> dict[str, float]:
    """Run one held-out example, extract real attention, and save a heatmap.

    Returns a small dict of diagnostics (e.g. the fraction of decoder cross-
    attention mass that lands on the reverse anti-diagonal) so the caller can
    report a real, measured number.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gen = torch.Generator().manual_seed(seed + 9999)
    # Draw examples until we get one with a clean, full-ish length for display.
    src, tgt_in, tgt_out = make_batch(task_cfg, 1, generator=gen)
    src, tgt_in, tgt_out = src.to(device), tgt_in.to(device), tgt_out.to(device)

    model.eval()
    with torch.no_grad():
        model(src, tgt_in)  # populates last_attn on every attention module

    src_ids = src[0].tolist()
    src_len = sum(1 for t in src_ids if t != PAD_IDX)
    tgt_in_ids = tgt_in[0].tolist()
    tgt_len = sum(1 for t in tgt_in_ids if t != PAD_IDX)

    # Real cached attention. Shapes: (B, heads, q, k). Average over heads, take
    # example 0, and trim to the non-pad region.
    enc_self = model.encoder.layers[-1].self_attn.last_attn[0].mean(0)
    enc_self = enc_self[:src_len, :src_len].cpu().numpy()
    cross = model.decoder.layers[-1].cross_attn.last_attn[0].mean(0)
    cross = cross[:tgt_len, :src_len].cpu().numpy()

    # Diagnostic: for reverse, decoder query t should read source position
    # src_len-1-t. Measure how much cross-attention mass sits on that anti-
    # diagonal (queries 0..src_len-1, i.e. excluding the trailing </s> step).
    anti = 0.0
    n = min(src_len, tgt_len)
    for t in range(n):
        anti += float(cross[t, src_len - 1 - t])
    anti_diag_mass = anti / max(n, 1)

    src_labels = _token_labels(src_ids[:src_len])
    tgt_labels = _token_labels(tgt_in_ids[:tgt_len])

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))

    im0 = axes[0].imshow(enc_self, cmap="viridis", vmin=0.0, aspect="auto")
    axes[0].set_title("Encoder self-attention\n(last layer, head-averaged)", fontsize=10)
    axes[0].set_xlabel("source key position")
    axes[0].set_ylabel("source query position")
    axes[0].set_xticks(range(src_len))
    axes[0].set_xticklabels(src_labels, fontsize=8)
    axes[0].set_yticks(range(src_len))
    axes[0].set_yticklabels(src_labels, fontsize=8)
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(cross, cmap="magma", vmin=0.0, aspect="auto")
    axes[1].set_title(
        f"Decoder->encoder cross-attention\n(last layer, head-averaged; "
        f"anti-diag mass={anti_diag_mass:.2f})",
        fontsize=10,
    )
    axes[1].set_xlabel("source key position")
    axes[1].set_ylabel("decoder query (output) position")
    axes[1].set_xticks(range(src_len))
    axes[1].set_xticklabels(src_labels, fontsize=8)
    axes[1].set_yticks(range(tgt_len))
    axes[1].set_yticklabels(tgt_labels, fontsize=8)
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    fig.suptitle(
        f"Real attention on the synthetic '{task_cfg.task}' task  "
        f"(src = {' '.join(src_labels)})",
        fontsize=11,
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return {"anti_diag_mass": anti_diag_mass, "src_len": float(src_len)}


def save_ablation_figure(
    results: dict[str, tuple[float, float]], task: str, path: str
) -> None:
    """Bar chart comparing seq/token accuracy with vs. without positional encoding."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arms = ["with PE", "without PE"]
    seq = [results["with"][1] * 100, results["without"][1] * 100]
    tok = [results["with"][0] * 100, results["without"][0] * 100]

    x = range(len(arms))
    width = 0.36
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    b1 = ax.bar([i - width / 2 for i in x], seq, width, label="seq. exact-match", color="#d62728")
    b2 = ax.bar([i + width / 2 for i in x], tok, width, label="token accuracy", color="#1f77b4")
    ax.set_ylabel("accuracy (%)")
    ax.set_ylim(0, 105)
    ax.set_xticks(list(x))
    ax.set_xticklabels(arms)
    ax.set_title(f"Positional-encoding ablation on '{task}'")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    for bars in (b1, b2):
        for rect in bars:
            ax.annotate(
                f"{rect.get_height():.1f}",
                (rect.get_x() + rect.get_width() / 2, rect.get_height()),
                xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8,
            )
    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> None:
    """Entry point: run both experiments and report real numbers + figures."""
    args = parse_args()
    device = resolve_device(args.device)
    print(f"Device: {device}")

    task_cfg = SyntheticTaskConfig(
        task=args.task,
        num_symbols=args.num_symbols,
        min_len=args.min_len,
        max_len=args.max_len,
    )
    vocab = vocab_size(task_cfg)

    # --- Experiment (b): positional-encoding ablation -----------------------
    print("\n=== Experiment (b): positional-encoding ablation ===")
    model_pe, tok_pe, seq_pe, t_pe = train_model(
        args, task_cfg, vocab, device, use_positional_encoding=True
    )
    print(f"with PE    : token_acc {tok_pe:.4f} | seq_acc {seq_pe:.4f} | {t_pe:.1f}s")
    _, tok_no, seq_no, t_no = train_model(
        args, task_cfg, vocab, device, use_positional_encoding=False
    )
    print(f"without PE : token_acc {tok_no:.4f} | seq_acc {seq_no:.4f} | {t_no:.1f}s")
    print(
        f"GAP (seq exact-match): {seq_pe:.4f} - {seq_no:.4f} = "
        f"{seq_pe - seq_no:+.4f}"
    )

    results = {"with": (tok_pe, seq_pe), "without": (tok_no, seq_no)}
    save_ablation_figure(results, args.task, args.ablation_figure)
    print(f"Saved ablation figure to {args.ablation_figure}")

    # --- Experiment (a): attention visualization ----------------------------
    print("\n=== Experiment (a): attention visualization ===")
    diag = save_attention_maps(model_pe, task_cfg, device, args.seed, args.attn_figure)
    print(
        f"Saved attention maps to {args.attn_figure} "
        f"(decoder cross-attention anti-diagonal mass = {diag['anti_diag_mass']:.2f})"
    )

    # --- Markdown-ready summary --------------------------------------------
    print("\n=== Summary (real measured numbers) ===")
    print(f"{'arm':<12}{'token acc':>12}{'seq exact':>12}")
    print(f"{'with PE':<12}{tok_pe:>12.4f}{seq_pe:>12.4f}")
    print(f"{'without PE':<12}{tok_no:>12.4f}{seq_no:>12.4f}")


if __name__ == "__main__":
    main()
