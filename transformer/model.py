"""Full Transformer model (paper Figure 1, section 3).

Assembles token embeddings, sinusoidal positional encoding, the encoder and
decoder stacks, and the final linear projection to vocabulary logits. Supports
weight tying between embeddings and the output projection (section 3.4).
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from .config import TransformerConfig
from .decoder import Decoder
from .encoder import Encoder
from .masking import make_src_mask, make_tgt_mask
from .positional import PositionalEncoding


class Transformer(nn.Module):
    """Encoder-decoder Transformer for sequence-to-sequence tasks.

    Args:
        config: A :class:`TransformerConfig` with all hyperparameters.
    """

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config

        self.src_embed = nn.Embedding(
            config.src_vocab_size, config.d_model, padding_idx=config.pad_idx
        )
        self.tgt_embed = nn.Embedding(
            config.tgt_vocab_size, config.d_model, padding_idx=config.pad_idx
        )
        self.pos_encoding = PositionalEncoding(
            config.d_model, config.dropout, config.max_seq_len
        )

        self.encoder = Encoder(
            config.num_encoder_layers,
            config.d_model,
            config.num_heads,
            config.d_ff,
            config.dropout,
            config.norm_first,
        )
        self.decoder = Decoder(
            config.num_decoder_layers,
            config.d_model,
            config.num_heads,
            config.d_ff,
            config.dropout,
            config.norm_first,
        )

        self.generator = nn.Linear(config.d_model, config.tgt_vocab_size)

        if config.tie_embeddings:
            # Share representations between source/target embeddings and the
            # output projection (paper section 3.4 / Press & Wolf 2017).
            self.tgt_embed.weight = self.src_embed.weight
            self.generator.weight = self.tgt_embed.weight

        self._reset_parameters()
        self._embed_scale = math.sqrt(config.d_model)

    def _reset_parameters(self) -> None:
        """Xavier-uniform initialization for all multi-dimensional parameters."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        # Keep padding embeddings at zero after the global init above.
        with torch.no_grad():
            self.src_embed.weight[self.config.pad_idx].zero_()
            self.tgt_embed.weight[self.config.pad_idx].zero_()

    def make_src_mask(self, src: Tensor) -> Tensor:
        """Build the source padding mask, shape ``(B, 1, 1, S)``."""
        return make_src_mask(src, self.config.pad_idx)

    def make_tgt_mask(self, tgt: Tensor) -> Tensor:
        """Build the combined causal + padding target mask, shape ``(B, 1, T, T)``."""
        return make_tgt_mask(tgt, self.config.pad_idx)

    def encode(self, src: Tensor, src_mask: Tensor) -> Tensor:
        """Embed and encode the source sequence ``(B, S)`` -> ``(B, S, d_model)``."""
        x = self.src_embed(src) * self._embed_scale
        x = self.pos_encoding(x)
        return self.encoder(x, src_mask)

    def decode(
        self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor, memory_mask: Tensor
    ) -> Tensor:
        """Embed and decode the target sequence -> ``(B, T, d_model)``."""
        x = self.tgt_embed(tgt) * self._embed_scale
        x = self.pos_encoding(x)
        return self.decoder(x, memory, tgt_mask, memory_mask)

    def forward(self, src: Tensor, tgt: Tensor) -> Tensor:
        """Run the full model.

        Args:
            src: Source token ids of shape ``(B, S)``.
            tgt: Target token ids of shape ``(B, T)`` (already shifted right, i.e.
                the decoder input that starts with ``<bos>``).

        Returns:
            Logits of shape ``(B, T, tgt_vocab_size)``.
        """
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)
        memory = self.encode(src, src_mask)
        out = self.decode(tgt, memory, tgt_mask, src_mask)
        return self.generator(out)

    def count_parameters(self, trainable_only: bool = True) -> int:
        """Return the number of (trainable) parameters."""
        return sum(
            p.numel()
            for p in self.parameters()
            if p.requires_grad or not trainable_only
        )

    @torch.no_grad()
    def greedy_decode(
        self,
        src: Tensor,
        max_len: int,
        bos_idx: int,
        eos_idx: int,
    ) -> Tensor:
        """Greedily decode target sequences for a batch of sources.

        Args:
            src: Source token ids of shape ``(B, S)``.
            max_len: Maximum number of target tokens to generate.
            bos_idx: Beginning-of-sequence token id (decoder start symbol).
            eos_idx: End-of-sequence token id; decoding stops once every sequence
                in the batch has emitted it.

        Returns:
            Generated token ids of shape ``(B, L)`` including the leading
            ``<bos>``; ``L`` is at most ``max_len + 1``.
        """
        self.eval()
        device = src.device
        batch = src.size(0)
        src_mask = self.make_src_mask(src)
        memory = self.encode(src, src_mask)

        ys = torch.full((batch, 1), bos_idx, dtype=torch.long, device=device)
        finished = torch.zeros(batch, dtype=torch.bool, device=device)
        for _ in range(max_len):
            tgt_mask = self.make_tgt_mask(ys)
            out = self.decode(ys, memory, tgt_mask, src_mask)
            logits = self.generator(out[:, -1])  # (B, vocab)
            next_token = logits.argmax(dim=-1, keepdim=True)  # (B, 1)
            ys = torch.cat([ys, next_token], dim=1)
            finished = finished | (next_token.squeeze(1) == eos_idx)
            if bool(finished.all()):
                break
        return ys
