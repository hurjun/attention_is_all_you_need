"""Configuration for the Transformer model.

All hyperparameters are gathered in a single immutable dataclass so that a
model, optimizer schedule and loss can be reconstructed exactly from one object.
Defaults follow the *base* model of Vaswani et al. (2017), "Attention Is All You
Need" (https://arxiv.org/abs/1706.03762), Table 3.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TransformerConfig:
    """Hyperparameters for :class:`transformer.model.Transformer`.

    Attributes:
        src_vocab_size: Size of the source vocabulary.
        tgt_vocab_size: Size of the target vocabulary.
        d_model: Embedding / model dimension (``d_model`` in the paper).
        num_heads: Number of attention heads (``h``). Must divide ``d_model``.
        num_encoder_layers: Number of stacked encoder layers (``N``).
        num_decoder_layers: Number of stacked decoder layers (``N``).
        d_ff: Inner dimension of the position-wise feed-forward network.
        dropout: Dropout probability applied throughout the network.
        max_seq_len: Maximum sequence length supported by positional encoding.
        pad_idx: Token id used for padding (excluded from attention and loss).
        tie_embeddings: If True, share weights between the source embedding,
            target embedding and the output projection (paper, section 3.4).
        norm_first: If True use pre-norm (LayerNorm before each sublayer); if
            False use the original post-norm formulation of the paper.
        label_smoothing: Label-smoothing epsilon (paper section 5.4).
        warmup_steps: Warmup steps for the Noam learning-rate schedule.
        adam_betas: Adam ``(beta1, beta2)`` coefficients.
        adam_eps: Adam epsilon.
    """

    src_vocab_size: int
    tgt_vocab_size: int
    d_model: int = 512
    num_heads: int = 8
    num_encoder_layers: int = 6
    num_decoder_layers: int = 6
    d_ff: int = 2048
    dropout: float = 0.1
    max_seq_len: int = 5000
    pad_idx: int = 0
    tie_embeddings: bool = True
    norm_first: bool = False
    label_smoothing: float = 0.1
    warmup_steps: int = 4000
    adam_betas: tuple[float, float] = (0.9, 0.98)
    adam_eps: float = 1e-9

    def __post_init__(self) -> None:
        if self.d_model % self.num_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by "
                f"num_heads ({self.num_heads})."
            )
        if self.tie_embeddings and self.src_vocab_size != self.tgt_vocab_size:
            raise ValueError(
                "tie_embeddings=True requires src_vocab_size == tgt_vocab_size "
                f"(got {self.src_vocab_size} and {self.tgt_vocab_size})."
            )

    @property
    def d_k(self) -> int:
        """Dimension of each attention head (``d_model / num_heads``)."""
        return self.d_model // self.num_heads
