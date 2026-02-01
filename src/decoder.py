"""
Decoder components for the Transformer model.

This module implements the Decoder Layer and Decoder Stack as described in
"Attention Is All You Need" (Vaswani et al., 2017).
"""

from typing import Optional

import torch
import torch.nn as nn

from src.attention import MultiHeadAttention
from src.feedforward import PositionwiseFeedForward


class DecoderLayer(nn.Module):
    """
    A single Decoder Layer as described in "Attention Is All You Need".

    Each decoder layer consists of three sub-layers:
    1. Masked multi-head self-attention mechanism (prevents attending to future positions)
    2. Multi-head cross-attention mechanism (attends to encoder output)
    3. Position-wise fully connected feed-forward network

    Each sub-layer has a residual connection followed by layer normalization:
    LayerNorm(x + Sublayer(x))

    Args:
        d_model: Model dimension (size of input/output embeddings). Default: 512
        n_heads: Number of attention heads. Default: 8
        d_ff: Inner layer dimension of the feed-forward network. Default: 2048
        dropout: Dropout probability. Default: 0.1

    Example:
        >>> layer = DecoderLayer(d_model=512, n_heads=8, d_ff=2048, dropout=0.1)
        >>> tgt = torch.randn(2, 10, 512)  # (batch, tgt_seq_len, d_model)
        >>> memory = torch.randn(2, 15, 512)  # (batch, src_seq_len, d_model)
        >>> output = layer(tgt, memory)
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

        # Sub-layer 1: Masked multi-head self-attention
        self.self_attention = MultiHeadAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
        )

        # Sub-layer 2: Multi-head cross-attention (encoder-decoder attention)
        self.cross_attention = MultiHeadAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
        )

        # Sub-layer 3: Position-wise feed-forward network
        self.feed_forward = PositionwiseFeedForward(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
        )

        # Layer normalization for each sub-layer
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        # Dropout for residual connections
        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)
        self.dropout3 = nn.Dropout(p=dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Process input through the decoder layer.

        Args:
            x: Target tensor of shape (batch, tgt_seq_len, d_model)
            memory: Encoder output tensor of shape (batch, src_seq_len, d_model)
            tgt_mask: Optional mask for target sequence. Should combine causal mask
                      and padding mask. Shape broadcastable to
                      (batch, n_heads, tgt_seq_len, tgt_seq_len)
            memory_mask: Optional mask for source sequence (encoder output) padding.
                         Shape broadcastable to
                         (batch, n_heads, tgt_seq_len, src_seq_len)

        Returns:
            Output tensor of shape (batch, tgt_seq_len, d_model)
        """
        # Sub-layer 1: Masked multi-head self-attention
        # Query, Key, Value all come from decoder input (self-attention)
        self_attn_output, _ = self.self_attention(x, x, x, mask=tgt_mask)
        x = self.norm1(x + self.dropout1(self_attn_output))

        # Sub-layer 2: Multi-head cross-attention (encoder-decoder attention)
        # Query comes from decoder, Key and Value come from encoder output
        cross_attn_output, _ = self.cross_attention(x, memory, memory, mask=memory_mask)
        x = self.norm2(x + self.dropout2(cross_attn_output))

        # Sub-layer 3: Feed-forward network
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout3(ff_output))

        return x

    def extra_repr(self) -> str:
        """Return extra representation for print."""
        return f"d_model={self.d_model}, n_heads={self.n_heads}, d_ff={self.d_ff}"


class Decoder(nn.Module):
    """
    Transformer Decoder Stack as described in "Attention Is All You Need".

    The decoder consists of a stack of N identical decoder layers.
    Each layer has masked self-attention, encoder-decoder cross-attention,
    and a position-wise feed-forward network, with residual connections
    and layer normalization.

    Args:
        n_layers: Number of decoder layers. Default: 6 (base model)
        d_model: Model dimension (size of input/output embeddings). Default: 512
        n_heads: Number of attention heads. Default: 8
        d_ff: Inner layer dimension of the feed-forward network. Default: 2048
        dropout: Dropout probability. Default: 0.1

    Example:
        >>> decoder = Decoder(n_layers=6, d_model=512, n_heads=8, d_ff=2048)
        >>> tgt = torch.randn(2, 10, 512)  # (batch, tgt_seq_len, d_model)
        >>> memory = torch.randn(2, 15, 512)  # (batch, src_seq_len, d_model)
        >>> output = decoder(tgt, memory)
        >>> output.shape
        torch.Size([2, 10, 512])
    """

    def __init__(
        self,
        n_layers: int = 6,
        d_model: int = 512,
        n_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.n_layers = n_layers
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff

        # Stack of N decoder layers
        self.layers = nn.ModuleList([
            DecoderLayer(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])

        # Final layer normalization
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Process input through all decoder layers.

        Args:
            x: Target tensor of shape (batch, tgt_seq_len, d_model)
               This should already have positional encoding added.
            memory: Encoder output tensor of shape (batch, src_seq_len, d_model)
            tgt_mask: Optional mask for target sequence. Should combine causal mask
                      and padding mask for autoregressive decoding.
                      Shape broadcastable to (batch, n_heads, tgt_seq_len, tgt_seq_len)
            memory_mask: Optional mask for source sequence (encoder output) padding.
                         Shape broadcastable to (batch, n_heads, tgt_seq_len, src_seq_len)

        Returns:
            Output tensor of shape (batch, tgt_seq_len, d_model)
        """
        # Pass through each decoder layer sequentially
        for layer in self.layers:
            x = layer(x, memory, tgt_mask=tgt_mask, memory_mask=memory_mask)

        # Apply final layer normalization
        x = self.norm(x)

        return x

    def extra_repr(self) -> str:
        """Return extra representation for print."""
        return (
            f"n_layers={self.n_layers}, d_model={self.d_model}, "
            f"n_heads={self.n_heads}, d_ff={self.d_ff}"
        )
