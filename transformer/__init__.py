"""A from-scratch PyTorch implementation of the Transformer.

Reimplements "Attention Is All You Need" (Vaswani et al., 2017,
https://arxiv.org/abs/1706.03762) without using ``torch.nn.Transformer``.
"""

from __future__ import annotations

from .attention import MultiHeadAttention, ScaledDotProductAttention
from .config import TransformerConfig
from .decoder import Decoder, DecoderLayer
from .encoder import Encoder, EncoderLayer
from .feed_forward import PositionwiseFeedForward
from .loss import LabelSmoothingLoss
from .masking import make_causal_mask, make_pad_mask, make_src_mask, make_tgt_mask
from .model import Transformer
from .positional import PositionalEncoding
from .schedule import make_noam_scheduler, noam_lambda

__all__ = [
    "Decoder",
    "DecoderLayer",
    "Encoder",
    "EncoderLayer",
    "LabelSmoothingLoss",
    "MultiHeadAttention",
    "PositionalEncoding",
    "PositionwiseFeedForward",
    "ScaledDotProductAttention",
    "Transformer",
    "TransformerConfig",
    "make_causal_mask",
    "make_noam_scheduler",
    "make_pad_mask",
    "make_src_mask",
    "make_tgt_mask",
    "noam_lambda",
]

__version__ = "0.1.0"
