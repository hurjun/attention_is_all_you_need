"""Small-scale, real EN->DE translation run on Multi30k (laptop-sized).

Unlike :mod:`scripts.train_translation` (which targets the paper-size base model
and was never run to convergence here), this script trains a deliberately *small*
Transformer that fits in ~8 GB and finishes in well under an hour on an Apple
Silicon ``mps`` device, then measures a **real** BLEU with sacrebleu on the
held-out Multi30k test split. It exists so the README can report an honestly
measured translation number — clearly labelled as a small-scale run, not the
paper-scale WMT'14 result.

It reuses the exact same from-scratch model, Noam schedule and label-smoothing
loss as the synthetic demo; only the model size and data source change.

Setup:
    pip install -r requirements.txt -r requirements-translation.txt

Run (defaults are the small config used for the README numbers):
    python scripts/train_translation_small.py --device mps

``datasets``/``sacrebleu`` are imported lazily so the rest of the repo and the
test suite never depend on them.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter

import torch
from torch import Tensor
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformer import (  # noqa: E402
    LabelSmoothingLoss,
    Transformer,
    TransformerConfig,
    make_noam_scheduler,
)

PAD, BOS, EOS, UNK = "<pad>", "<bos>", "<eos>", "<unk>"
PAD_IDX, BOS_IDX, EOS_IDX, UNK_IDX = 0, 1, 2, 3
SPECIALS = [PAD, BOS, EOS, UNK]

_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments (small defaults that fit on a laptop)."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="bentrevett/multi30k",
                   help="HuggingFace dataset id with 'en' and 'de' columns.")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--min-freq", type=int, default=2)
    p.add_argument("--max-len", type=int, default=40)
    # Deliberately small model (~paper base is d_model=512/N=6/h=8/d_ff=2048).
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--num-layers", type=int, default=3)
    p.add_argument("--d-ff", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    p.add_argument("--eval-batch-size", type=int, default=64)
    p.add_argument("--out", default="results/multi30k_small.json",
                   help="Where to write the JSON results record.")
    return p.parse_args()


def resolve_device(choice: str) -> torch.device:
    """Resolve the requested device, falling back to CPU."""
    if choice == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(choice)


def tokenize(text: str) -> list[str]:
    """Lowercase word/punctuation tokenizer.

    Splits punctuation off words (e.g. ``"bushes."`` -> ``["bushes", "."]``) so
    the word vocabulary is not polluted by glued punctuation. The paper uses
    subword (BPE) tokenization; this stays dependency-light and self-contained.
    """
    return _TOKEN_RE.findall(text.lower().strip())


def build_vocab(sentences: list[str], min_freq: int) -> dict[str, int]:
    """Build a word->id vocabulary, reserving ids for special tokens."""
    counter: Counter[str] = Counter()
    for s in sentences:
        counter.update(tokenize(s))
    vocab = {tok: i for i, tok in enumerate(SPECIALS)}
    for token, freq in counter.most_common():
        if freq >= min_freq:
            vocab[token] = len(vocab)
    return vocab


def encode(text: str, vocab: dict[str, int], max_len: int) -> list[int]:
    """Encode a sentence into ids with <bos>/<eos>, truncated to ``max_len``."""
    ids = [vocab.get(tok, UNK_IDX) for tok in tokenize(text)][: max_len - 2]
    return [BOS_IDX, *ids, EOS_IDX]


def collate(
    batch: list[tuple[list[int], list[int]]], device: torch.device, pad_to: int
) -> tuple[Tensor, Tensor]:
    """Pad a batch of (src_ids, tgt_ids) to a *fixed* width ``pad_to``.

    Padding to a constant length (rather than per-batch max) keeps every batch
    the same shape. On Apple ``mps`` this is a large win: variable shapes force
    the Metal graph to recompile every step, which we measured at ~10x slower.
    Pad positions are masked in attention and ignored by the loss, so fixed
    padding changes speed only, not correctness.
    """
    src = torch.full((len(batch), pad_to), PAD_IDX, dtype=torch.long)
    tgt = torch.full((len(batch), pad_to), PAD_IDX, dtype=torch.long)
    for i, (s, t) in enumerate(batch):
        src[i, : len(s)] = torch.tensor(s)
        tgt[i, : len(t)] = torch.tensor(t)
    return src.to(device), tgt.to(device)


@torch.no_grad()
def compute_bleu(
    model: Transformer,
    sources: list[list[int]],
    refs: list[str],
    inv_tgt: dict[int, str],
    device: torch.device,
    max_len: int,
    eval_batch_size: int,
) -> tuple[float, list[str]]:
    """Batched greedy-decode and score corpus BLEU with sacrebleu.

    Sentences are length-sorted into batches to cut padding, decoded greedily,
    then restored to the original order. Returns ``(bleu, hypotheses)``. BLEU is
    computed case-insensitively because the model is trained on lowercased text.
    """
    import sacrebleu

    model.eval()
    src_width = max(len(s) for s in sources)
    order = sorted(range(len(sources)), key=lambda i: len(sources[i]))
    hyps_by_idx: dict[int, str] = {}
    for start in range(0, len(order), eval_batch_size):
        idxs = order[start : start + eval_batch_size]
        src = torch.full((len(idxs), src_width), PAD_IDX, dtype=torch.long)
        for row, i in enumerate(idxs):
            src[row, : len(sources[i])] = torch.tensor(sources[i])
        src = src.to(device)
        out = model.greedy_decode(src, max_len=max_len, bos_idx=BOS_IDX, eos_idx=EOS_IDX)
        for row, i in enumerate(idxs):
            toks = [inv_tgt.get(int(t), UNK) for t in out[row, 1:].tolist()]
            cut = [t for t in _until_eos(toks) if t not in (PAD, BOS)]
            hyps_by_idx[i] = " ".join(cut)
    hyps = [hyps_by_idx[i] for i in range(len(sources))]
    bleu = float(sacrebleu.corpus_bleu(hyps, [refs], lowercase=True).score)
    return bleu, hyps


def _until_eos(toks: list[str]) -> list[str]:
    """Truncate a token list at the first ``<eos>`` (exclusive)."""
    out: list[str] = []
    for t in toks:
        if t == EOS:
            break
        out.append(t)
    return out


def main() -> None:
    """Entry point for the small-scale translation training + BLEU run."""
    from datasets import load_dataset

    args = parse_args()
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    print(f"Device: {device}")

    ds = load_dataset(args.dataset)
    train, valid, test = ds["train"], ds["validation"], ds["test"]

    src_vocab = build_vocab([ex["en"] for ex in train], args.min_freq)
    tgt_vocab = build_vocab([ex["de"] for ex in train], args.min_freq)
    inv_tgt = {i: t for t, i in tgt_vocab.items()}
    print(f"src vocab={len(src_vocab)} | tgt vocab={len(tgt_vocab)}")

    def to_pair(ex: dict[str, str]) -> tuple[list[int], list[int]]:
        return (
            encode(ex["en"], src_vocab, args.max_len),
            encode(ex["de"], tgt_vocab, args.max_len),
        )

    train_pairs = [to_pair(ex) for ex in train]
    test_src = [encode(ex["en"], src_vocab, args.max_len) for ex in test]
    test_ref = [ex["de"].strip() for ex in test]

    loader = DataLoader(
        train_pairs,  # type: ignore[arg-type]
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate(b, device, args.max_len),
    )

    cfg = TransformerConfig(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_encoder_layers=args.num_layers,
        num_decoder_layers=args.num_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        max_seq_len=args.max_len + 8,
        pad_idx=PAD_IDX,
        tie_embeddings=False,  # separate src/tgt vocabularies
        label_smoothing=args.label_smoothing,
        warmup_steps=args.warmup,
    )
    model = Transformer(cfg).to(device)
    n_params = model.count_parameters()
    print(f"params={n_params:,}")

    criterion = LabelSmoothingLoss(len(tgt_vocab), PAD_IDX, args.label_smoothing)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1.0, betas=cfg.adam_betas, eps=cfg.adam_eps
    )
    scheduler = make_noam_scheduler(optimizer, cfg.d_model, args.warmup)

    t0 = time.time()
    step = 0
    history: list[dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for src, tgt in loader:
            tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]
            logits = model(src, tgt_in)
            loss = criterion(logits, tgt_out)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total += loss.item()
            step += 1
        train_loss = total / len(loader)
        history.append({"epoch": epoch, "train_loss": round(train_loss, 4)})
        print(f"epoch {epoch:3d} | step {step:5d} | train_loss {train_loss:.4f} "
              f"| {time.time() - t0:.0f}s")

    train_secs = time.time() - t0
    print(f"training done in {train_secs:.0f}s; computing test BLEU...")

    bleu, hyps = compute_bleu(
        model, test_src, test_ref, inv_tgt, device, args.max_len, args.eval_batch_size
    )
    print(f"TEST BLEU (sacrebleu, case-insensitive) = {bleu:.2f}")

    samples = []
    for i in range(min(5, len(test_ref))):
        src_text = " ".join(tokenize(test[i]["en"]))
        samples.append({"src": src_text, "ref": test_ref[i], "hyp": hyps[i]})

    record = {
        "dataset": args.dataset,
        "split_sizes": {"train": len(train), "valid": len(valid), "test": len(test)},
        "device": str(device),
        "torch_version": torch.__version__,
        "model": {
            "d_model": args.d_model,
            "num_heads": args.num_heads,
            "num_layers": args.num_layers,
            "d_ff": args.d_ff,
            "params": n_params,
            "src_vocab": len(src_vocab),
            "tgt_vocab": len(tgt_vocab),
            "tie_embeddings": False,
        },
        "training": {
            "epochs": args.epochs,
            "steps": step,
            "batch_size": args.batch_size,
            "warmup_steps": args.warmup,
            "label_smoothing": args.label_smoothing,
            "max_len": args.max_len,
            "seed": args.seed,
            "wall_clock_seconds": round(train_secs, 1),
            "final_train_loss": history[-1]["train_loss"],
        },
        "eval": {
            "metric": "sacrebleu corpus BLEU (case-insensitive, 13a tokenization)",
            "test_bleu": round(bleu, 2),
            "samples": samples,
        },
        "note": (
            "Small-scale laptop run, NOT the paper-scale WMT'14 EN->DE result. "
            "Paper base model reports BLEU 27.3 on WMT'14 newstest2014."
        ),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
