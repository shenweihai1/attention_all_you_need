"""
Embedding components for the Transformer model.

This module implements the Input Embedding with scaling as described in
"Attention Is All You Need" (Vaswani et al., 2017).
"""

import math

import torch
import torch.nn as nn


class TransformerEmbedding(nn.Module):
    """
    Transformer Input Embedding with scaling as described in "Attention Is All You Need".

    In the embedding layers, we multiply the learned embeddings by sqrt(d_model).
    This scaling is applied to make the embeddings have similar magnitude to the
    positional encodings that will be added to them.

    Args:
        vocab_size: Size of the vocabulary.
        d_model: Dimension of the model embeddings. Default: 512
        padding_idx: If specified, the entries at this index do not contribute
                     to the gradient; the embedding vector at this index is
                     initialized to zeros. Default: None

    Example:
        >>> embedding = TransformerEmbedding(vocab_size=10000, d_model=512)
        >>> tokens = torch.randint(0, 10000, (2, 10))  # (batch, seq_len)
        >>> embedded = embedding(tokens)
        >>> embedded.shape
        torch.Size([2, 10, 512])
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        padding_idx: int = None,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.padding_idx = padding_idx

        # Standard embedding layer
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            padding_idx=padding_idx,
        )

        # Scaling factor: sqrt(d_model)
        self.scale = math.sqrt(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert token indices to scaled embeddings.

        Args:
            x: Tensor of token indices of shape (batch, seq_len)

        Returns:
            Scaled embedding tensor of shape (batch, seq_len, d_model)
        """
        # Multiply embeddings by sqrt(d_model) as per paper
        return self.embedding(x) * self.scale

    def extra_repr(self) -> str:
        """Return extra representation for print."""
        return (
            f"vocab_size={self.vocab_size}, d_model={self.d_model}, "
            f"padding_idx={self.padding_idx}"
        )
