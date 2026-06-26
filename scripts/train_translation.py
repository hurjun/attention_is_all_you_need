"""Optional EN->DE translation training on Multi30k (HuggingFace ``datasets``).

This is the "scale to a real task" path. It reuses the exact same from-scratch
model, Noam schedule and label-smoothing loss as the synthetic demo, swapping the
data source for the Multi30k EN->DE corpus loaded via HuggingFace ``datasets``
(``torchtext`` is deprecated and intentionally avoided).

The reported BLEU in the README for this path is the paper's *target*; it has NOT
been reproduced on the CPU-only machine used for the synthetic demo. Running this
to a competitive BLEU needs a GPU and longer training.

Setup:
    pip install -r requirements.txt -r requirements-translation.txt

Run:
    python scripts/train_translation.py --epochs 20 --batch-size 128

This module imports ``datasets``/``sacrebleu`` lazily so the rest of the repo and
the test suite do not depend on them.
"""

from __future__ import annotations

import argparse
import os
import sys
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


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="bentrevett/multi30k",
                   help="HuggingFace dataset id with 'en' and 'de' columns.")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--min-freq", type=int, default=2)
    p.add_argument("--max-len", type=int, default=64)
    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--num-heads", type=int, default=8)
    p.add_argument("--num-layers", type=int, default=6)
    p.add_argument("--d-ff", type=int, default=2048)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--warmup", type=int, default=4000)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
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
    """Very small whitespace + lowercase tokenizer.

    A production system would use a subword tokenizer (e.g. SentencePiece BPE,
    as in the paper). Whitespace tokenization keeps this script dependency-light
    and self-contained.
    """
    return text.lower().strip().split()


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
    batch: list[tuple[list[int], list[int]]], device: torch.device
) -> tuple[Tensor, Tensor]:
    """Pad a batch of (src_ids, tgt_ids) into tensors."""
    src_max = max(len(s) for s, _ in batch)
    tgt_max = max(len(t) for _, t in batch)
    src = torch.full((len(batch), src_max), PAD_IDX, dtype=torch.long)
    tgt = torch.full((len(batch), tgt_max), PAD_IDX, dtype=torch.long)
    for i, (s, t) in enumerate(batch):
        src[i, : len(s)] = torch.tensor(s)
        tgt[i, : len(t)] = torch.tensor(t)
    return src.to(device), tgt.to(device)


@torch.no_grad()
def compute_bleu(
    model: Transformer,
    examples: list[tuple[list[int], str]],
    inv_tgt: dict[int, str],
    device: torch.device,
    max_len: int,
) -> float:
    """Greedy-decode and score corpus BLEU with sacrebleu."""
    import sacrebleu

    model.eval()
    hyps: list[str] = []
    refs: list[str] = []
    for src_ids, ref_text in examples:
        src = torch.tensor([src_ids], device=device)
        out = model.greedy_decode(src, max_len=max_len, bos_idx=BOS_IDX, eos_idx=EOS_IDX)
        toks = [inv_tgt.get(int(t), UNK) for t in out[0, 1:].tolist()]
        toks = [t for t in toks if t not in (EOS, PAD, BOS)]
        hyps.append(" ".join(toks))
        refs.append(ref_text)
    return float(sacrebleu.corpus_bleu(hyps, [refs]).score)


def main() -> None:
    """Entry point for the optional translation training run."""
    from datasets import load_dataset

    args = parse_args()
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    print(f"Device: {device}")

    ds = load_dataset(args.dataset)
    train, valid = ds["train"], ds["validation"]

    src_vocab = build_vocab([ex["en"] for ex in train], args.min_freq)
    tgt_vocab = build_vocab([ex["de"] for ex in train], args.min_freq)
    inv_tgt = {i: t for t, i in tgt_vocab.items()}
    print(f"src vocab={len(src_vocab)} | tgt vocab={len(tgt_vocab)}")

    def to_pair(ex: dict[str, str]) -> tuple[list[int], list[int]]:
        return encode(ex["en"], src_vocab, args.max_len), encode(
            ex["de"], tgt_vocab, args.max_len
        )

    train_pairs = [to_pair(ex) for ex in train]
    valid_eval = [
        (encode(ex["en"], src_vocab, args.max_len), ex["de"].strip()) for ex in valid
    ]

    loader = DataLoader(
        train_pairs,  # type: ignore[arg-type]
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate(b, device),
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
    print(f"params={model.count_parameters():,}")

    criterion = LabelSmoothingLoss(len(tgt_vocab), PAD_IDX, args.label_smoothing)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1.0, betas=cfg.adam_betas, eps=cfg.adam_eps
    )
    scheduler = make_noam_scheduler(optimizer, cfg.d_model, args.warmup)

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
        bleu = compute_bleu(model, valid_eval, inv_tgt, device, args.max_len)
        print(f"epoch {epoch:3d} | train_loss {total / len(loader):.4f} | val BLEU {bleu:.2f}")


if __name__ == "__main__":
    main()
