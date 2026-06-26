"""Train a small Transformer on a synthetic copy/reverse/sort task.

This script needs no downloads and trains to high accuracy on CPU/MPS in a few
minutes, producing a real loss curve (saved as a PNG) and real sample
predictions. It is the reproducible demonstration referenced by the README.

Example:
    python scripts/train_synthetic.py --task reverse --steps 2000
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch

# Make the repository root importable when run as a script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tasks.synthetic import (  # noqa: E402
    BOS_IDX,
    EOS_IDX,
    PAD_IDX,
    SyntheticTaskConfig,
    decode_tokens,
    make_batch,
    vocab_size,
)
from transformer import (  # noqa: E402
    LabelSmoothingLoss,
    Transformer,
    TransformerConfig,
    make_noam_scheduler,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__)
    # Task
    p.add_argument("--task", default="reverse", choices=["copy", "reverse", "sort"])
    p.add_argument("--num-symbols", type=int, default=16)
    p.add_argument("--min-len", type=int, default=6)
    p.add_argument("--max-len", type=int, default=12)
    # Model (small by default so it trains fast on CPU)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--d-ff", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.1)
    # Optimization
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--warmup", type=int, default=400)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--eval-every", type=int, default=200)
    p.add_argument("--eval-batches", type=int, default=8)
    # Misc
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    p.add_argument("--figure", default="assets/loss_curve.png")
    p.add_argument("--save", default="", help="Optional checkpoint path (.pt).")
    return p.parse_args()


def resolve_device(choice: str) -> torch.device:
    """Resolve the requested device, falling back to CPU when unavailable."""
    if choice == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(choice)


@torch.no_grad()
def teacher_forced_token_accuracy(
    model: Transformer, src: torch.Tensor, tgt_in: torch.Tensor, tgt_out: torch.Tensor
) -> tuple[int, int]:
    """Return (correct, total) non-pad tokens under teacher forcing."""
    model.eval()
    logits = model(src, tgt_in)
    preds = logits.argmax(dim=-1)
    mask = tgt_out != PAD_IDX
    correct = int((preds.eq(tgt_out) & mask).sum())
    total = int(mask.sum())
    return correct, total


def _strip_at_eos(tokens: list[int]) -> list[int]:
    """Truncate a token list at (and including) the first EOS, dropping pads."""
    out: list[int] = []
    for t in tokens:
        if t == PAD_IDX:
            continue
        out.append(t)
        if t == EOS_IDX:
            break
    return out


@torch.no_grad()
def greedy_sequence_accuracy(
    model: Transformer, src: torch.Tensor, tgt_out: torch.Tensor, max_len: int
) -> tuple[int, int]:
    """Return (exact_matches, batch_size) under greedy free-running decoding."""
    model.eval()
    generated = model.greedy_decode(src, max_len=max_len, bos_idx=BOS_IDX, eos_idx=EOS_IDX)
    pred = generated[:, 1:]  # drop the leading <bos>
    matches = 0
    for i in range(src.size(0)):
        gold = _strip_at_eos(tgt_out[i].tolist())
        hyp = _strip_at_eos(pred[i].tolist())
        matches += int(gold == hyp)
    return matches, src.size(0)


def evaluate(
    model: Transformer,
    task_cfg: SyntheticTaskConfig,
    batch_size: int,
    num_batches: int,
    gen: torch.Generator,
    device: torch.device,
) -> tuple[float, float]:
    """Compute token-level and sequence-level accuracy over fresh batches."""
    tok_correct = tok_total = seq_correct = seq_total = 0
    for _ in range(num_batches):
        src, tgt_in, tgt_out = make_batch(task_cfg, batch_size, generator=gen)
        src, tgt_in, tgt_out = src.to(device), tgt_in.to(device), tgt_out.to(device)
        c, t = teacher_forced_token_accuracy(model, src, tgt_in, tgt_out)
        tok_correct += c
        tok_total += t
        sc, st = greedy_sequence_accuracy(model, src, tgt_out, max_len=task_cfg.max_len + 1)
        seq_correct += sc
        seq_total += st
    return tok_correct / max(tok_total, 1), seq_correct / max(seq_total, 1)


def save_loss_curve(
    steps: list[int],
    losses: list[float],
    eval_steps: list[int],
    eval_acc: list[float],
    path: str,
    task: str,
) -> None:
    """Save a loss + accuracy curve to ``path`` (small PNG)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(6.0, 3.8))
    ax1.plot(steps, losses, color="#1f77b4", linewidth=1.5, label="train loss")
    ax1.set_xlabel("training step")
    ax1.set_ylabel("label-smoothed loss", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(eval_steps, [a * 100 for a in eval_acc], color="#d62728",
             marker="o", markersize=3, linewidth=1.3, label="seq. accuracy")
    ax2.set_ylabel("greedy seq. accuracy (%)", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")
    ax2.set_ylim(0, 105)

    fig.suptitle(f"Transformer on synthetic '{task}' task", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> None:
    """Entry point: build the model, train, evaluate, and report."""
    args = parse_args()
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    print(f"Device: {device}")

    task_cfg = SyntheticTaskConfig(
        task=args.task,
        num_symbols=args.num_symbols,
        min_len=args.min_len,
        max_len=args.max_len,
    )
    vocab = vocab_size(task_cfg)

    model_cfg = TransformerConfig(
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
    model = Transformer(model_cfg).to(device)
    print(f"Task: {args.task} | vocab={vocab} | params={model.count_parameters():,}")

    criterion = LabelSmoothingLoss(vocab, pad_idx=PAD_IDX, smoothing=args.label_smoothing)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1.0, betas=model_cfg.adam_betas, eps=model_cfg.adam_eps
    )
    scheduler = make_noam_scheduler(optimizer, model_cfg.d_model, args.warmup)

    train_gen = torch.Generator().manual_seed(args.seed)
    eval_gen = torch.Generator().manual_seed(args.seed + 1)

    steps_hist: list[int] = []
    loss_hist: list[float] = []
    eval_steps: list[int] = []
    eval_acc_hist: list[float] = []

    start = time.time()
    running = 0.0
    for step in range(1, args.steps + 1):
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

        loss_value = loss.item()
        running += loss_value
        steps_hist.append(step)
        loss_hist.append(loss_value)

        if step % args.eval_every == 0 or step == args.steps:
            tok_acc, seq_acc = evaluate(
                model, task_cfg, args.batch_size, args.eval_batches, eval_gen, device
            )
            eval_steps.append(step)
            eval_acc_hist.append(seq_acc)
            avg = running / args.eval_every
            running = 0.0
            lr = scheduler.get_last_lr()[0]
            print(
                f"step {step:5d} | loss {avg:.4f} | lr {lr:.2e} | "
                f"token_acc {tok_acc:.4f} | seq_acc {seq_acc:.4f}"
            )

    elapsed = time.time() - start
    print(f"\nTrained {args.steps} steps in {elapsed:.1f}s on {device}.")

    # Final evaluation on a larger held-out sample.
    final_gen = torch.Generator().manual_seed(args.seed + 777)
    tok_acc, seq_acc = evaluate(model, task_cfg, args.batch_size, 16, final_gen, device)
    print(f"FINAL  token_acc {tok_acc:.4f} | seq_acc {seq_acc:.4f}")

    save_loss_curve(steps_hist, loss_hist, eval_steps, eval_acc_hist, args.figure, args.task)
    print(f"Saved loss curve to {args.figure}")

    # Qualitative samples.
    print("\nSample greedy decodes (held-out):")
    sample_gen = torch.Generator().manual_seed(args.seed + 9999)
    src, _, tgt_out = make_batch(task_cfg, 5, generator=sample_gen)
    src, tgt_out = src.to(device), tgt_out.to(device)
    generated = model.greedy_decode(
        src, max_len=task_cfg.max_len + 1, bos_idx=BOS_IDX, eos_idx=EOS_IDX
    )
    for i in range(src.size(0)):
        gold = _strip_at_eos(tgt_out[i].tolist())
        hyp = _strip_at_eos(generated[i, 1:].tolist())
        ok = "OK " if gold == hyp else "XX "
        print(f"  {ok}src={decode_tokens(src[i].tolist())}")
        print(f"      gold={decode_tokens(gold)}")
        print(f"      pred={decode_tokens(hyp)}")

    if args.save:
        os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
        torch.save({"model": model.state_dict(), "config": model_cfg}, args.save)
        print(f"Saved checkpoint to {args.save}")


if __name__ == "__main__":
    main()
