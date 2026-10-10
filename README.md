# Transformer from Scratch — the 2017 Transformer, built from PyTorch primitives and checked on a laptop

[![CI](https://github.com/hurjun/attention_is_all_you_need/actions/workflows/ci.yml/badge.svg)](https://github.com/hurjun/attention_is_all_you_need/actions/workflows/ci.yml)

An educational re-implementation of the encoder–decoder Transformer from
*Attention Is All You Need* (Vaswani et al., 2017). Not novel research, and
not a wrapper around `torch.nn.Transformer`. Every part is built from
`nn.Linear`/`nn.LayerNorm`: attention, masks, positional encoding (token
position signal), the Noam schedule (learning-rate warmup) and label-smoothing
loss (softened targets). A synthetic task makes the result checkable on a
laptop CPU.

![Attention maps: encoder self-attention (left) and decoder cross-attention on the anti-diagonal (right)](assets/attention_maps.png)

## Result

- Synthetic `reverse` task, 2000 steps, ≈0.66 M params: **0.9824** exact-match on held-out sequences, ~75–90 s on CPU (Apple Silicon).
- Decoder cross-attention puts **0.97** of its mass on the anti-diagonal: the model learned to read the source backwards.
- Ablation (separate 1200-step run, same seed and data): removing positional encoding drops exact-match **91.2% → 0.0%**.
- Real translation: BLEU 27.3 is the paper's target, **not reproduced** here (no GPU). Only the synthetic-task numbers were measured.

Why it is here: this project is practice in making a model checkable. It uses
invariant tests, seeded reruns, a one-variable ablation, and reads internal state
to explain a result. That is the same discipline I want to bring to
simulation-based validation and fault diagnosis of cyber-physical systems.

## How it works

- Encoder and decoder stacks: multi-head attention, feed-forward, residual + LayerNorm (post-norm, as in the paper).
- Masks are boolean. Masked logits get `-inf` before the softmax, so the decoder never sees the future.
- Training follows the paper: Adam β=(0.9, 0.98), ε=1e-9, warmup=4000, label smoothing 0.1.
- 33 tests (~4 s on CPU) check shapes, the causal mask, head split/merge and overfit-one-batch.
- `scripts/experiments.py` reads the real attention weights cached on each head and runs the ablation.

## Run it

```bash
git clone https://github.com/hurjun/attention_is_all_you_need.git && cd attention_is_all_you_need
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/train_synthetic.py --task reverse --steps 2000 --device cpu   # main run, ~90 s
python scripts/experiments.py --steps 1200 --warmup 200 --device cpu          # attention map + ablation
pip install pytest ruff && ruff check . && pytest -q                           # 33 tests, ~4 s on CPU
```

## Limitations

- The headline result is a synthetic task. No translation BLEU was measured; the Multi30k scripts are provided but not run to a result (no GPU).
- Results were produced with Python 3.13 (torch 2.12.1). CI tests Python 3.11 and 3.12.
- The translation scripts use whitespace or regex word tokenizers, not the paper's subword (BPE) tokenizer.
- The paper-size base config (d_model=512, N=6, h=8) has ≈63.1 M params, not the paper's ≈65 M (embedding tying, vocab size).

## More

- [Details](docs/DETAILS.md): full metric tables, sample predictions, loss curve, ablation interpretation, architecture diagram, tests, project tree, pinned versions.
- [paper_notes.md](paper_notes.md): module → paper section/equation mapping and design choices.
- Paper: [arXiv:1706.03762](https://arxiv.org/abs/1706.03762). Also: [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/).

## License

MIT.

— Jun Hur · [portfolio](https://hurjun.github.io) · [KCC 2024 paper](https://hurjun.github.io/asset/kcc2024-kiise-nursing-record-paper.pdf) · hurjun96@gmail.com
