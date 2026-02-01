"""
Attention mechanisms for the Transformer model.

This module implements the Scaled Dot-Product Attention as described in
"Attention Is All You Need" (Vaswani et al., 2017).
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    dropout: Optional[nn.Dropout] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Scaled Dot-Product Attention.

    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V

    Args:
        query: Query tensor of shape (batch, ..., seq_len_q, d_k)
        key: Key tensor of shape (batch, ..., seq_len_k, d_k)
        value: Value tensor of shape (batch, ..., seq_len_k, d_v)
        mask: Optional mask tensor. Positions with True/1 are masked (not attended to).
              Shape should be broadcastable to (batch, ..., seq_len_q, seq_len_k)
        dropout: Optional dropout module to apply to attention weights.

    Returns:
        tuple of:
            - Output tensor of shape (batch, ..., seq_len_q, d_v)
            - Attention weights of shape (batch, ..., seq_len_q, seq_len_k)

    Example:
        >>> q = torch.randn(2, 8, 10, 64)  # (batch, heads, seq_len, d_k)
        >>> k = torch.randn(2, 8, 10, 64)
        >>> v = torch.randn(2, 8, 10, 64)
        >>> output, attn_weights = scaled_dot_product_attention(q, k, v)
        >>> output.shape
        torch.Size([2, 8, 10, 64])
    """
    d_k = query.size(-1)

    # Compute attention scores: QK^T / sqrt(d_k)
    # query: (..., seq_len_q, d_k), key: (..., seq_len_k, d_k)
    # scores: (..., seq_len_q, seq_len_k)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    # Apply mask if provided
    if mask is not None:
        # Mask positions should be filled with -inf so softmax gives 0
        scores = scores.masked_fill(mask, float("-inf"))

    # Apply softmax to get attention weights
    attention_weights = F.softmax(scores, dim=-1)

    # Apply dropout if provided
    if dropout is not None:
        attention_weights = dropout(attention_weights)

    # Compute weighted sum of values
    # attention_weights: (..., seq_len_q, seq_len_k), value: (..., seq_len_k, d_v)
    # output: (..., seq_len_q, d_v)
    output = torch.matmul(attention_weights, value)

    return output, attention_weights


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention as a PyTorch module.

    This implements the attention mechanism described in "Attention Is All You Need":
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V

    The module supports:
    - Optional masking for preventing attention to certain positions
    - Optional dropout on attention weights

    Args:
        dropout: Dropout probability for attention weights. Default: 0.0

    Example:
        >>> attention = ScaledDotProductAttention(dropout=0.1)
        >>> q = torch.randn(2, 8, 10, 64)  # (batch, heads, seq_len, d_k)
        >>> k = torch.randn(2, 8, 10, 64)
        >>> v = torch.randn(2, 8, 10, 64)
        >>> output, attn_weights = attention(q, k, v)
    """

    def __init__(self, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else None

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply scaled dot-product attention.

        Args:
            query: Query tensor of shape (batch, ..., seq_len_q, d_k)
            key: Key tensor of shape (batch, ..., seq_len_k, d_k)
            value: Value tensor of shape (batch, ..., seq_len_k, d_v)
            mask: Optional boolean mask. True positions are masked (not attended).
                  Shape should be broadcastable to (batch, ..., seq_len_q, seq_len_k)

        Returns:
            tuple of:
                - Output tensor of shape (batch, ..., seq_len_q, d_v)
                - Attention weights of shape (batch, ..., seq_len_q, seq_len_k)
        """
        return scaled_dot_product_attention(
            query, key, value, mask=mask, dropout=self.dropout
        )


def create_causal_mask(seq_len: int, device: torch.device = None) -> torch.Tensor:
    """
    Create a causal (look-ahead) mask for decoder self-attention.

    This mask prevents positions from attending to subsequent positions,
    ensuring the auto-regressive property during generation.

    Args:
        seq_len: Sequence length
        device: Device to create the mask on

    Returns:
        Boolean mask of shape (seq_len, seq_len) where True means "do not attend".
        The mask is upper triangular (excluding diagonal).

    Example:
        >>> mask = create_causal_mask(4)
        >>> mask
        tensor([[False,  True,  True,  True],
                [False, False,  True,  True],
                [False, False, False,  True],
                [False, False, False, False]])
    """
    # Create upper triangular matrix (True above diagonal)
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
    return mask


def create_padding_mask(
    seq: torch.Tensor, pad_idx: int = 0
) -> torch.Tensor:
    """
    Create a padding mask from a sequence tensor.

    Args:
        seq: Input sequence tensor of shape (batch, seq_len)
        pad_idx: Index of the padding token

    Returns:
        Boolean mask of shape (batch, 1, 1, seq_len) where True means "padding position".
        The extra dimensions allow broadcasting with attention scores.

    Example:
        >>> seq = torch.tensor([[1, 2, 3, 0, 0], [1, 2, 0, 0, 0]])
        >>> mask = create_padding_mask(seq, pad_idx=0)
        >>> mask.shape
        torch.Size([2, 1, 1, 5])
    """
    # (batch, seq_len) -> (batch, 1, 1, seq_len)
    mask = (seq == pad_idx).unsqueeze(1).unsqueeze(2)
    return mask


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism as described in "Attention Is All You Need".

    MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
    where head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)

    This implementation does NOT use torch.nn.MultiheadAttention, as per project requirements.
    Instead, it manually implements the multi-head attention using linear projections
    and the scaled dot-product attention from this module.

    Args:
        d_model: Model dimension (size of input/output embeddings)
        n_heads: Number of attention heads
        dropout: Dropout probability for attention weights. Default: 0.0

    Raises:
        ValueError: If d_model is not divisible by n_heads

    Example:
        >>> mha = MultiHeadAttention(d_model=512, n_heads=8, dropout=0.1)
        >>> x = torch.randn(2, 10, 512)  # (batch, seq_len, d_model)
        >>> output, attn_weights = mha(x, x, x)
        >>> output.shape
        torch.Size([2, 10, 512])
        >>> attn_weights.shape
        torch.Size([2, 8, 10, 10])
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()

        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
            )

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # Dimension per head

        # Linear projections for Q, K, V
        # Each projects from d_model to d_model (all heads combined)
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)

        # Output projection
        self.w_o = nn.Linear(d_model, d_model, bias=False)

        # Dropout for attention weights
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else None

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply multi-head attention.

        Args:
            query: Query tensor of shape (batch, seq_len_q, d_model)
            key: Key tensor of shape (batch, seq_len_k, d_model)
            value: Value tensor of shape (batch, seq_len_k, d_model)
            mask: Optional boolean mask. True positions are masked (not attended).
                  Shape should be broadcastable to (batch, n_heads, seq_len_q, seq_len_k)

        Returns:
            tuple of:
                - Output tensor of shape (batch, seq_len_q, d_model)
                - Attention weights of shape (batch, n_heads, seq_len_q, seq_len_k)
        """
        batch_size = query.size(0)
        seq_len_q = query.size(1)
        seq_len_k = key.size(1)

        # Step 1: Linear projections
        # (batch, seq_len, d_model) -> (batch, seq_len, d_model)
        q = self.w_q(query)
        k = self.w_k(key)
        v = self.w_v(value)

        # Step 2: Reshape to separate heads
        # (batch, seq_len, d_model) -> (batch, seq_len, n_heads, d_k)
        # -> (batch, n_heads, seq_len, d_k)
        q = q.view(batch_size, seq_len_q, self.n_heads, self.d_k).transpose(1, 2)
        k = k.view(batch_size, seq_len_k, self.n_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, seq_len_k, self.n_heads, self.d_k).transpose(1, 2)

        # Step 3: Apply scaled dot-product attention to all heads in parallel
        # q, k, v: (batch, n_heads, seq_len, d_k)
        attn_output, attn_weights = scaled_dot_product_attention(
            q, k, v, mask=mask, dropout=self.dropout
        )

        # Step 4: Concatenate heads
        # (batch, n_heads, seq_len_q, d_k) -> (batch, seq_len_q, n_heads, d_k)
        # -> (batch, seq_len_q, d_model)
        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len_q, self.d_model)
        )

        # Step 5: Final linear projection
        output = self.w_o(attn_output)

        return output, attn_weights

    def extra_repr(self) -> str:
        """Return extra representation for print."""
        return f"d_model={self.d_model}, n_heads={self.n_heads}, d_k={self.d_k}"
