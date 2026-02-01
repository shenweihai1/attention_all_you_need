"""
Positional Encoding for the Transformer model.

This module implements the sinusoidal positional encoding as described in
"Attention Is All You Need" (Vaswani et al., 2017).
"""

import math
from typing import Optional

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding as described in "Attention Is All You Need".

    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    The positional encodings have the same dimension as the embeddings so they
    can be summed. This module adds positional encoding to the input embeddings.

    Args:
        d_model: Model dimension (size of embeddings). Default: 512
        max_seq_len: Maximum sequence length to pre-compute encodings for. Default: 5000
        dropout: Dropout probability applied after adding positional encoding. Default: 0.0

    Example:
        >>> pe = PositionalEncoding(d_model=512, max_seq_len=1000, dropout=0.1)
        >>> x = torch.randn(2, 100, 512)  # (batch, seq_len, d_model)
        >>> output = pe(x)
        >>> output.shape
        torch.Size([2, 100, 512])
    """

    def __init__(
        self,
        d_model: int = 512,
        max_seq_len: int = 5000,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # Dropout applied after adding positional encoding
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else None

        # Pre-compute positional encodings for all positions up to max_seq_len
        pe = self._create_positional_encoding(max_seq_len, d_model)

        # Register as buffer (not a parameter, but should be saved/loaded with model)
        # Shape: (1, max_seq_len, d_model) for easy broadcasting
        self.register_buffer("pe", pe.unsqueeze(0))

    def _create_positional_encoding(
        self, max_seq_len: int, d_model: int
    ) -> torch.Tensor:
        """
        Create the positional encoding matrix.

        Args:
            max_seq_len: Maximum sequence length
            d_model: Model dimension

        Returns:
            Positional encoding tensor of shape (max_seq_len, d_model)
        """
        # Create position indices: (max_seq_len, 1)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)

        # Create dimension indices for the division term
        # div_term = 10000^(2i/d_model) = exp(2i * -log(10000) / d_model)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )

        # Create the positional encoding matrix
        pe = torch.zeros(max_seq_len, d_model)

        # Apply sin to even indices (0, 2, 4, ...)
        pe[:, 0::2] = torch.sin(position * div_term)

        # Apply cos to odd indices (1, 3, 5, ...)
        # Handle case where d_model is odd
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])

        return pe

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input embeddings.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model)

        Returns:
            Output tensor of shape (batch, seq_len, d_model) with positional
            encoding added.

        Raises:
            ValueError: If sequence length exceeds max_seq_len
        """
        seq_len = x.size(1)

        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence length ({seq_len}) exceeds maximum "
                f"({self.max_seq_len}). Increase max_seq_len."
            )

        # Add positional encoding to input
        # pe is (1, max_seq_len, d_model), slice to (1, seq_len, d_model)
        x = x + self.pe[:, :seq_len, :]

        # Apply dropout if configured
        if self.dropout is not None:
            x = self.dropout(x)

        return x

    def get_encoding(self, seq_len: int) -> torch.Tensor:
        """
        Get the positional encoding for a given sequence length.

        Useful for visualization or debugging.

        Args:
            seq_len: Desired sequence length

        Returns:
            Positional encoding tensor of shape (seq_len, d_model)
        """
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence length ({seq_len}) exceeds maximum ({self.max_seq_len})"
            )
        return self.pe[0, :seq_len, :]

    def extra_repr(self) -> str:
        """Return extra representation for print."""
        return f"d_model={self.d_model}, max_seq_len={self.max_seq_len}"
