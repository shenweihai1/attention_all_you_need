"""
Position-wise Feed-Forward Network for the Transformer model.

This module implements the feed-forward network as described in
"Attention Is All You Need" (Vaswani et al., 2017).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionwiseFeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network as described in "Attention Is All You Need".

    FFN(x) = max(0, xW1 + b1)W2 + b2

    This consists of two linear transformations with a ReLU activation in between.
    The same network is applied to each position separately and identically.

    Args:
        d_model: Model dimension (size of input/output embeddings). Default: 512
        d_ff: Inner layer dimension (hidden size). Default: 2048
        dropout: Dropout probability applied after the ReLU. Default: 0.0

    Example:
        >>> ffn = PositionwiseFeedForward(d_model=512, d_ff=2048, dropout=0.1)
        >>> x = torch.randn(2, 10, 512)  # (batch, seq_len, d_model)
        >>> output = ffn(x)
        >>> output.shape
        torch.Size([2, 10, 512])
    """

    def __init__(
        self,
        d_model: int = 512,
        d_ff: int = 2048,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.d_model = d_model
        self.d_ff = d_ff

        # First linear transformation: d_model -> d_ff
        self.linear1 = nn.Linear(d_model, d_ff)

        # Second linear transformation: d_ff -> d_model
        self.linear2 = nn.Linear(d_ff, d_model)

        # Dropout after ReLU activation
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply position-wise feed-forward network.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model)

        Returns:
            Output tensor of shape (batch, seq_len, d_model)
        """
        # FFN(x) = max(0, xW1 + b1)W2 + b2
        # Step 1: First linear + ReLU
        hidden = F.relu(self.linear1(x))

        # Step 2: Dropout (if enabled)
        if self.dropout is not None:
            hidden = self.dropout(hidden)

        # Step 3: Second linear
        output = self.linear2(hidden)

        return output

    def extra_repr(self) -> str:
        """Return extra representation for print."""
        return f"d_model={self.d_model}, d_ff={self.d_ff}"
