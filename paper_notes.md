# Paper notes: mapping code to "Attention Is All You Need"

Reference: Vaswani et al., 2017, *Attention Is All You Need*,
[arXiv:1706.03762](https://arxiv.org/abs/1706.03762). Section/equation numbers
below refer to that paper. The goal of this document is to show that every piece
of the implementation is grounded in a specific part of the paper rather than
copied from a tutorial.

## Module-to-paper map

| Code | Paper | Notes |
|------|-------|-------|
| `transformer/attention.py` · `ScaledDotProductAttention` | §3.2.1, **Eq. 1** | `softmax(QKᵀ / √d_k) V`. The `1/√d_k` scaling counteracts the growth of dot products in large dimension (paper footnote 4). Masked positions are set to `-inf` before softmax. |
| `transformer/attention.py` · `MultiHeadAttention` | §3.2.2, **Eq. 2** | `h` parallel heads of size `d_k = d_v = d_model/h`. Linear projections `W^Q, W^K, W^V, W^O`; split → attend → concat → project. |
| `transformer/positional.py` · `PositionalEncoding` | §3.5 | `PE(pos,2i)=sin(pos/10000^{2i/d})`, `PE(pos,2i+1)=cos(...)`. Stored via `register_buffer` (fixed, not learned). |
| `transformer/feed_forward.py` · `PositionwiseFeedForward` | §3.3, **Eq. 2** (FFN) | `max(0, xW₁+b₁)W₂+b₂` with inner size `d_ff=2048`. Applied identically at every position. |
| `transformer/encoder.py` · `EncoderLayer` / `Encoder` | §3.1 (left of Fig. 1) | Two sub-layers: self-attention + FFN, each wrapped by residual + LayerNorm (`LayerNorm(x + Sublayer(x))`, §5.4 residual dropout). `N = 6`. |
| `transformer/decoder.py` · `DecoderLayer` / `Decoder` | §3.1 (right of Fig. 1) | Three sub-layers: masked self-attention, encoder–decoder (cross) attention, FFN. `N = 6`. |
| `transformer/masking.py` · `make_tgt_mask` | §3.2.3 | Causal (look-ahead) mask preserving auto-regression, combined (AND) with the padding mask. |
| `transformer/masking.py` · `make_pad_mask` | §5.1 | Excludes `<pad>` tokens from attention. |
| `transformer/model.py` · `Transformer` | Fig. 1, §3.4 | Full assembly. Embeddings scaled by `√d_model` (§3.4); optional weight tying between the two embeddings and the pre-softmax projection (§3.4). |
| `transformer/schedule.py` · `noam_lambda` | §5.3, **Eq. 3** | `lr = d_model^{-0.5} · min(step^{-0.5}, step · warmup^{-1.5})`, `warmup=4000`. |
| `transformer/loss.py` · `LabelSmoothingLoss` | §5.4 | Label smoothing `ε_ls = 0.1` via KL divergence against a smoothed target distribution; hurts perplexity but improves accuracy/BLEU. |

## Hyperparameters (base model, paper Table 3)

| Symbol | Value | Where |
|--------|-------|-------|
| `d_model` | 512 | `TransformerConfig.d_model` |
| `N` (layers) | 6 | `num_encoder_layers`, `num_decoder_layers` |
| `h` (heads) | 8 | `num_heads` |
| `d_ff` | 2048 | `d_ff` |
| `P_drop` | 0.1 | `dropout` |
| Adam `β₁, β₂` | 0.9, 0.98 | `adam_betas` |
| Adam `ε` | 1e-9 | `adam_eps` |
| warmup | 4000 | `warmup_steps` |
| `ε_ls` | 0.1 | `label_smoothing` |

## Deliberate deviations / choices

1. **Tokenizer.** The paper uses byte-pair / word-piece encoding (~37 000 shared
   subwords). The synthetic demo uses an integer symbol vocabulary (no tokenizer
   needed); the optional translation script uses simple whitespace tokenization
   to stay dependency-light. Neither changes the architecture.
2. **Pre-norm option.** The paper uses **post-norm** (`LayerNorm(x + Sublayer(x))`),
   which is the default here (`norm_first=False`). A pre-norm variant
   (`norm_first=True`), which is more stable for deep stacks, is also provided.
3. **Parameter count.** With tied embeddings and a shared 37 000-token vocab this
   implementation has ≈63.1 M parameters; the paper quotes ≈65 M for the base
   model. The difference comes from embedding tying and the exact vocabulary
   size, not from the layer architecture.
4. **Masking sign convention.** Masks here are boolean tensors where `True` means
   "attend"; masked entries are filled with `-inf` before the softmax. This is
   equivalent to the additive `-1e9` masks sometimes used, but avoids the small
   leakage that a finite `-1e9` allows.
5. **Synthetic demo, not WMT.** The reproducible result in this repo is a
   sequence-reversal task that a small model learns on CPU in minutes. The WMT/
   Multi30k path is provided but its BLEU is the paper's *target*, not a number
   reproduced on this hardware.

## References

- Vaswani et al., *Attention Is All You Need*, NeurIPS 2017. https://arxiv.org/abs/1706.03762
- Rush, *The Annotated Transformer*. https://nlp.seas.harvard.edu/annotated-transformer/
- Press & Wolf, *Using the Output Embedding to Improve Language Models*, 2017
  (weight tying). https://arxiv.org/abs/1608.05859
