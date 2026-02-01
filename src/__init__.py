"""
Transformer implementation from scratch based on "Attention Is All You Need" paper.

This package provides a pure PyTorch implementation of the Transformer architecture
without using torch.nn.Transformer or torch.nn.MultiheadAttention.
"""

__version__ = "0.1.0"

from src.attention import (
    scaled_dot_product_attention,
    ScaledDotProductAttention,
    MultiHeadAttention,
    create_causal_mask,
    create_padding_mask,
)
from src.feedforward import PositionwiseFeedForward
from src.positional_encoding import PositionalEncoding
