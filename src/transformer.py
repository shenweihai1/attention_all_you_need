"""
Full Transformer model implementation.

This module implements the complete Transformer architecture as described in
"Attention Is All You Need" (Vaswani et al., 2017).
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.embedding import TransformerEmbedding
from src.positional_encoding import PositionalEncoding
from src.encoder import Encoder
from src.decoder import Decoder
from src.attention import create_causal_mask, create_padding_mask
from src.init import init_transformer_weights, init_bert_weights


class Transformer(nn.Module):
    """
    Full Transformer model as described in "Attention Is All You Need".

    The Transformer follows an encoder-decoder architecture:
    - Source tokens are embedded, positional encoding is added, and passed through encoder
    - Target tokens are embedded, positional encoding is added, and passed through decoder
    - Decoder output is projected to vocabulary size for prediction

    Args:
        src_vocab_size: Size of the source vocabulary
        tgt_vocab_size: Size of the target vocabulary
        d_model: Model dimension (embedding dimension). Default: 512
        n_heads: Number of attention heads. Default: 8
        n_encoder_layers: Number of encoder layers. Default: 6
        n_decoder_layers: Number of decoder layers. Default: 6
        d_ff: Feed-forward network inner dimension. Default: 2048
        dropout: Dropout probability. Default: 0.1
        max_seq_len: Maximum sequence length for positional encoding. Default: 5000
        pad_idx: Padding token index. Default: 0
        share_embeddings: Whether to share embeddings between encoder and decoder. Default: False

    Example:
        >>> model = Transformer(src_vocab_size=10000, tgt_vocab_size=10000)
        >>> src = torch.randint(0, 10000, (2, 20))  # (batch, src_len)
        >>> tgt = torch.randint(0, 10000, (2, 15))  # (batch, tgt_len)
        >>> output = model(src, tgt)
        >>> output.shape
        torch.Size([2, 15, 10000])
    """

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 512,
        n_heads: int = 8,
        n_encoder_layers: int = 6,
        n_decoder_layers: int = 6,
        d_ff: int = 2048,
        dropout: float = 0.1,
        max_seq_len: int = 5000,
        pad_idx: int = 0,
        share_embeddings: bool = False,
    ):
        super().__init__()

        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_encoder_layers = n_encoder_layers
        self.n_decoder_layers = n_decoder_layers
        self.d_ff = d_ff
        self.pad_idx = pad_idx

        # Source embedding with scaling
        self.src_embedding = TransformerEmbedding(
            vocab_size=src_vocab_size,
            d_model=d_model,
            padding_idx=pad_idx,
        )

        # Target embedding with scaling
        if share_embeddings and src_vocab_size == tgt_vocab_size:
            self.tgt_embedding = self.src_embedding
        else:
            self.tgt_embedding = TransformerEmbedding(
                vocab_size=tgt_vocab_size,
                d_model=d_model,
                padding_idx=pad_idx,
            )

        # Positional encoding (shared between encoder and decoder)
        self.positional_encoding = PositionalEncoding(
            d_model=d_model,
            max_seq_len=max_seq_len,
            dropout=dropout,
        )

        # Encoder stack
        self.encoder = Encoder(
            n_layers=n_encoder_layers,
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            dropout=dropout,
        )

        # Decoder stack
        self.decoder = Decoder(
            n_layers=n_decoder_layers,
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            dropout=dropout,
        )

        # Final linear projection to vocabulary
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize all weights using Xavier uniform initialization."""
        self.apply(lambda m: init_transformer_weights(m, d_model=self.d_model))

    def init_weights(self, method: str = "xavier") -> None:
        """
        Reinitialize all weights in the model.

        Args:
            method: Initialization method. Options:
                - "xavier": Xavier uniform (default, from original paper)
                - "bert": Normal with std=0.02 (BERT-style)
        """
        if method == "xavier":
            self.apply(lambda m: init_transformer_weights(m, d_model=self.d_model))
        elif method == "bert":
            self.apply(init_bert_weights)
        else:
            raise ValueError(f"Unknown initialization method: {method}")

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through the Transformer.

        Args:
            src: Source token indices of shape (batch, src_len)
            tgt: Target token indices of shape (batch, tgt_len)
            src_mask: Optional mask for source padding.
                      If None, will be computed from src using pad_idx.
            tgt_mask: Optional mask for target (causal + padding).
                      If None, will be computed as causal mask.
            memory_mask: Optional mask for encoder-decoder attention.
                         If None, will use src_mask.

        Returns:
            Logits of shape (batch, tgt_len, tgt_vocab_size)
        """
        # Create masks if not provided
        if src_mask is None:
            src_mask = create_padding_mask(src, pad_idx=self.pad_idx)

        if tgt_mask is None:
            tgt_mask = self._create_tgt_mask(tgt)

        if memory_mask is None:
            memory_mask = src_mask

        # Encode source
        memory = self.encode(src, src_mask)

        # Decode target
        decoder_output = self.decode(tgt, memory, tgt_mask, memory_mask)

        # Project to vocabulary
        logits = self.output_projection(decoder_output)

        return logits

    def encode(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode source sequence.

        Args:
            src: Source token indices of shape (batch, src_len)
            src_mask: Optional mask for source padding

        Returns:
            Encoder output of shape (batch, src_len, d_model)
        """
        # Embed and add positional encoding
        src_embedded = self.src_embedding(src)
        src_embedded = self.positional_encoding(src_embedded)

        # Pass through encoder
        memory = self.encoder(src_embedded, src_mask=src_mask)

        return memory

    def decode(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Decode target sequence given encoder memory.

        Args:
            tgt: Target token indices of shape (batch, tgt_len)
            memory: Encoder output of shape (batch, src_len, d_model)
            tgt_mask: Optional mask for target (causal + padding)
            memory_mask: Optional mask for encoder-decoder attention

        Returns:
            Decoder output of shape (batch, tgt_len, d_model)
        """
        # Embed and add positional encoding
        tgt_embedded = self.tgt_embedding(tgt)
        tgt_embedded = self.positional_encoding(tgt_embedded)

        # Pass through decoder
        decoder_output = self.decoder(
            tgt_embedded,
            memory,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
        )

        return decoder_output

    def _create_tgt_mask(self, tgt: torch.Tensor) -> torch.Tensor:
        """
        Create target mask combining causal and padding masks.

        Args:
            tgt: Target tensor of shape (batch, tgt_len)

        Returns:
            Combined mask for decoder self-attention
        """
        tgt_len = tgt.size(1)

        # Causal mask to prevent attending to future positions
        causal_mask = create_causal_mask(tgt_len, device=tgt.device)

        # Padding mask for target
        padding_mask = create_padding_mask(tgt, pad_idx=self.pad_idx)

        # Combine masks (both are True where attention should be blocked)
        # causal_mask: (1, 1, tgt_len, tgt_len)
        # padding_mask: (batch, 1, 1, tgt_len)
        combined_mask = causal_mask | padding_mask

        return combined_mask

    def generate(
        self,
        src: torch.Tensor,
        max_len: int,
        start_token: int,
        end_token: int,
        src_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Generate target sequence autoregressively.

        Args:
            src: Source token indices of shape (batch, src_len)
            max_len: Maximum length to generate
            start_token: Start of sequence token index
            end_token: End of sequence token index
            src_mask: Optional mask for source padding

        Returns:
            Generated token indices of shape (batch, generated_len)
        """
        batch_size = src.size(0)
        device = src.device

        # Create source mask if not provided
        if src_mask is None:
            src_mask = create_padding_mask(src, pad_idx=self.pad_idx)

        # Encode source
        memory = self.encode(src, src_mask)

        # Initialize target with start token
        tgt = torch.full((batch_size, 1), start_token, dtype=torch.long, device=device)

        # Track which sequences have finished
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(max_len - 1):
            # Create target mask
            tgt_mask = self._create_tgt_mask(tgt)

            # Decode
            decoder_output = self.decode(tgt, memory, tgt_mask, src_mask)

            # Get logits for last position
            logits = self.output_projection(decoder_output[:, -1, :])

            # Get predicted token (greedy decoding)
            next_token = logits.argmax(dim=-1, keepdim=True)

            # Append to target
            tgt = torch.cat([tgt, next_token], dim=1)

            # Update finished status
            finished = finished | (next_token.squeeze(-1) == end_token)

            # Stop if all sequences have finished
            if finished.all():
                break

        return tgt

    def extra_repr(self) -> str:
        """Return extra representation for print."""
        return (
            f"src_vocab_size={self.src_vocab_size}, "
            f"tgt_vocab_size={self.tgt_vocab_size}, "
            f"d_model={self.d_model}, n_heads={self.n_heads}, "
            f"n_encoder_layers={self.n_encoder_layers}, "
            f"n_decoder_layers={self.n_decoder_layers}, "
            f"d_ff={self.d_ff}"
        )
