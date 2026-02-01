"""
Encoder components for the Transformer model.

This module implements the Encoder Layer and Encoder Stack as described in
"Attention Is All You Need" (Vaswani et al., 2017).
"""

from typing import Optional

import torch
import torch.nn as nn

from src.attention import MultiHeadAttention
from src.feedforward import PositionwiseFeedForward


class EncoderLayer(nn.Module):
    """
    A single Encoder Layer as described in "Attention Is All You Need".

    Each encoder layer consists of two sub-layers:
    1. Multi-head self-attention mechanism
    2. Position-wise fully connected feed-forward network

    Each sub-layer has a residual connection followed by layer normalization:
    LayerNorm(x + Sublayer(x))

    Args:
        d_model: Model dimension (size of input/output embeddings). Default: 512
        n_heads: Number of attention heads. Default: 8
        d_ff: Inner layer dimension of the feed-forward network. Default: 2048
        dropout: Dropout probability. Default: 0.1

    Example:
        >>> layer = EncoderLayer(d_model=512, n_heads=8, d_ff=2048, dropout=0.1)
        >>> x = torch.randn(2, 10, 512)  # (batch, seq_len, d_model)
        >>> output = layer(x)
        >>> output.shape
        torch.Size([2, 10, 512])
    """

    def __init__(
        self,
        d_model: int = 512,
        n_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff

        # Multi-head self-attention
        self.self_attention = MultiHeadAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
        )

        # Position-wise feed-forward network
        self.feed_forward = PositionwiseFeedForward(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Dropout for residual connections
        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)

    def forward(
        self,
        x: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Process input through the encoder layer.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            src_mask: Optional mask for source sequence padding.
                      Shape should be broadcastable to (batch, n_heads, seq_len, seq_len)

        Returns:
            Output tensor of shape (batch, seq_len, d_model)
        """
        # Sub-layer 1: Multi-head self-attention with residual connection
        # Attention output
        attn_output, _ = self.self_attention(x, x, x, mask=src_mask)
        # Residual connection and layer norm
        x = self.norm1(x + self.dropout1(attn_output))

        # Sub-layer 2: Feed-forward network with residual connection
        # FFN output
        ff_output = self.feed_forward(x)
        # Residual connection and layer norm
        x = self.norm2(x + self.dropout2(ff_output))

        return x

    def extra_repr(self) -> str:
        """Return extra representation for print."""
        return f"d_model={self.d_model}, n_heads={self.n_heads}, d_ff={self.d_ff}"
