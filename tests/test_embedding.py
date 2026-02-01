"""
Tests for the Embedding module.

Tests the TransformerEmbedding implementation ensuring it matches the architecture
described in "Attention Is All You Need".
"""

import math

import pytest
import torch
import torch.nn as nn

from src.embedding import TransformerEmbedding


class TestTransformerEmbedding:
    """Tests for the TransformerEmbedding module."""

    def test_initialization_default(self):
        """Test TransformerEmbedding initialization with default d_model."""
        embedding = TransformerEmbedding(vocab_size=10000)
        assert embedding.vocab_size == 10000
        assert embedding.d_model == 512
        assert embedding.padding_idx is None
        assert embedding.scale == math.sqrt(512)

    def test_initialization_custom(self):
        """Test TransformerEmbedding initialization with custom parameters."""
        embedding = TransformerEmbedding(vocab_size=5000, d_model=256, padding_idx=0)
        assert embedding.vocab_size == 5000
        assert embedding.d_model == 256
        assert embedding.padding_idx == 0
        assert embedding.scale == math.sqrt(256)

    def test_output_shape(self):
        """Test that output has correct shape."""
        vocab_size = 10000
        d_model = 512
        embedding = TransformerEmbedding(vocab_size=vocab_size, d_model=d_model)

        batch, seq_len = 2, 10
        tokens = torch.randint(0, vocab_size, (batch, seq_len))
        output = embedding(tokens)

        assert output.shape == (batch, seq_len, d_model)

    def test_output_shape_various_sizes(self):
        """Test output shape with various batch and sequence sizes."""
        embedding = TransformerEmbedding(vocab_size=1000, d_model=256)

        test_cases = [
            (1, 1),
            (1, 100),
            (32, 50),
            (4, 512),
        ]

        for batch, seq_len in test_cases:
            tokens = torch.randint(0, 1000, (batch, seq_len))
            output = embedding(tokens)
            assert output.shape == (batch, seq_len, 256)

    def test_scaling_applied(self):
        """Test that scaling by sqrt(d_model) is applied."""
        d_model = 256
        embedding = TransformerEmbedding(vocab_size=100, d_model=d_model)

        tokens = torch.tensor([[1, 2, 3]])

        # Get raw embedding without scaling
        raw_embedding = embedding.embedding(tokens)

        # Get scaled embedding
        scaled_embedding = embedding(tokens)

        # Verify scaling is applied
        expected = raw_embedding * math.sqrt(d_model)
        assert torch.allclose(scaled_embedding, expected)

    def test_scaling_factor_value(self):
        """Test the scaling factor value for different d_model."""
        for d_model in [64, 128, 256, 512, 1024]:
            embedding = TransformerEmbedding(vocab_size=100, d_model=d_model)
            expected_scale = math.sqrt(d_model)
            assert embedding.scale == expected_scale

    def test_padding_idx_zero_gradient(self):
        """Test that padding_idx position has zero embedding."""
        pad_idx = 0
        embedding = TransformerEmbedding(vocab_size=100, d_model=64, padding_idx=pad_idx)

        # Get the embedding for padding index
        pad_token = torch.tensor([[pad_idx]])
        pad_embedding = embedding(pad_token)

        # Padding embedding should be zero (before and after scaling)
        assert torch.allclose(pad_embedding, torch.zeros_like(pad_embedding))

    def test_gradient_flow(self):
        """Test that gradients flow through the embedding layer."""
        embedding = TransformerEmbedding(vocab_size=100, d_model=64)
        tokens = torch.randint(0, 100, (2, 10))

        output = embedding(tokens)
        loss = output.sum()
        loss.backward()

        # Check gradients on embedding weights
        assert embedding.embedding.weight.grad is not None
        assert not torch.isnan(embedding.embedding.weight.grad).any()

    def test_gradient_not_on_padding(self):
        """Test that gradients don't flow to padding index."""
        pad_idx = 0
        embedding = TransformerEmbedding(vocab_size=100, d_model=64, padding_idx=pad_idx)

        # Use only padding tokens
        tokens = torch.zeros(2, 10, dtype=torch.long)

        output = embedding(tokens)
        loss = output.sum()
        loss.backward()

        # Padding index should have zero gradient
        assert embedding.embedding.weight.grad[pad_idx].abs().sum() == 0

    def test_different_tokens_different_embeddings(self):
        """Test that different tokens produce different embeddings."""
        embedding = TransformerEmbedding(vocab_size=100, d_model=64)

        token1 = torch.tensor([[1]])
        token2 = torch.tensor([[2]])

        emb1 = embedding(token1)
        emb2 = embedding(token2)

        assert not torch.allclose(emb1, emb2)

    def test_same_token_same_embedding(self):
        """Test that same tokens produce same embeddings."""
        embedding = TransformerEmbedding(vocab_size=100, d_model=64)

        tokens = torch.tensor([[5, 5, 5]])
        output = embedding(tokens)

        # All positions with same token should have same embedding
        assert torch.allclose(output[0, 0], output[0, 1])
        assert torch.allclose(output[0, 1], output[0, 2])

    def test_batch_independence(self):
        """Test that batches are processed independently."""
        embedding = TransformerEmbedding(vocab_size=100, d_model=64)

        tokens = torch.randint(0, 100, (2, 10))

        output_batched = embedding(tokens)
        output0 = embedding(tokens[0:1])
        output1 = embedding(tokens[1:2])

        assert torch.allclose(output_batched[0:1], output0)
        assert torch.allclose(output_batched[1:2], output1)

    def test_deterministic_output(self):
        """Test that output is deterministic."""
        embedding = TransformerEmbedding(vocab_size=100, d_model=64)

        tokens = torch.randint(0, 100, (2, 10))

        output1 = embedding(tokens)
        output2 = embedding(tokens)

        assert torch.allclose(output1, output2)

    def test_no_nan_output(self):
        """Test that output contains no NaN values."""
        embedding = TransformerEmbedding(vocab_size=100, d_model=64)

        # Test various inputs
        test_cases = [
            torch.randint(0, 100, (2, 10)),
            torch.zeros(2, 10, dtype=torch.long),
            torch.ones(2, 10, dtype=torch.long) * 99,
        ]

        for tokens in test_cases:
            output = embedding(tokens)
            assert not torch.isnan(output).any()
            assert not torch.isinf(output).any()

    def test_embedding_weight_shape(self):
        """Test that embedding weight has correct shape."""
        vocab_size = 1000
        d_model = 256
        embedding = TransformerEmbedding(vocab_size=vocab_size, d_model=d_model)

        assert embedding.embedding.weight.shape == (vocab_size, d_model)

    def test_parameter_count(self):
        """Test that the model has expected number of parameters."""
        vocab_size = 1000
        d_model = 256
        embedding = TransformerEmbedding(vocab_size=vocab_size, d_model=d_model)

        expected_params = vocab_size * d_model
        total_params = sum(p.numel() for p in embedding.parameters())
        assert total_params == expected_params

    def test_extra_repr(self):
        """Test the extra_repr method."""
        embedding = TransformerEmbedding(vocab_size=10000, d_model=512, padding_idx=0)
        repr_str = embedding.extra_repr()

        assert "vocab_size=10000" in repr_str
        assert "d_model=512" in repr_str
        assert "padding_idx=0" in repr_str

    def test_embedding_magnitude(self):
        """Test that scaled embeddings have appropriate magnitude."""
        d_model = 512
        embedding = TransformerEmbedding(vocab_size=1000, d_model=d_model)

        tokens = torch.randint(0, 1000, (10, 50))
        output = embedding(tokens)

        # The scaling should make embeddings have similar magnitude to positional encodings
        # Positional encodings are bounded by [-1, 1], so scaled embeddings should be
        # roughly on the same scale
        mean_magnitude = output.abs().mean()

        # After scaling by sqrt(d_model), embeddings should be reasonably sized
        # This is a sanity check, not an exact bound
        assert mean_magnitude > 0.1
        assert mean_magnitude < 100

    def test_with_various_d_model(self):
        """Test embedding works with various d_model values."""
        for d_model in [64, 128, 256, 512, 768, 1024]:
            embedding = TransformerEmbedding(vocab_size=100, d_model=d_model)
            tokens = torch.randint(0, 100, (2, 10))
            output = embedding(tokens)

            assert output.shape == (2, 10, d_model)
            assert not torch.isnan(output).any()


class TestTransformerEmbeddingIntegration:
    """Integration tests for the TransformerEmbedding."""

    def test_with_positional_encoding(self):
        """Test TransformerEmbedding combined with positional encoding."""
        from src.positional_encoding import PositionalEncoding

        vocab_size = 1000
        d_model = 256
        seq_len = 50

        embedding = TransformerEmbedding(vocab_size=vocab_size, d_model=d_model)
        pe = PositionalEncoding(d_model=d_model, max_seq_len=100, dropout=0.0)

        tokens = torch.randint(0, vocab_size, (2, seq_len))

        # Embed tokens
        embedded = embedding(tokens)

        # Add positional encoding
        output = pe(embedded)

        assert output.shape == (2, seq_len, d_model)
        assert not torch.isnan(output).any()

    def test_with_encoder(self):
        """Test TransformerEmbedding with Encoder."""
        from src.positional_encoding import PositionalEncoding
        from src.encoder import Encoder

        vocab_size = 1000
        d_model = 256
        seq_len = 20

        embedding = TransformerEmbedding(vocab_size=vocab_size, d_model=d_model)
        pe = PositionalEncoding(d_model=d_model, max_seq_len=100, dropout=0.0)
        encoder = Encoder(n_layers=2, d_model=d_model, n_heads=4, d_ff=1024, dropout=0.0)

        tokens = torch.randint(0, vocab_size, (2, seq_len))

        # Full pipeline
        x = embedding(tokens)
        x = pe(x)
        output = encoder(x)

        assert output.shape == (2, seq_len, d_model)
        assert not torch.isnan(output).any()

    def test_with_decoder(self):
        """Test TransformerEmbedding with Decoder."""
        from src.positional_encoding import PositionalEncoding
        from src.decoder import Decoder

        vocab_size = 1000
        d_model = 256

        embedding = TransformerEmbedding(vocab_size=vocab_size, d_model=d_model)
        pe = PositionalEncoding(d_model=d_model, max_seq_len=100, dropout=0.0)
        decoder = Decoder(n_layers=2, d_model=d_model, n_heads=4, d_ff=1024, dropout=0.0)

        tgt_tokens = torch.randint(0, vocab_size, (2, 15))
        memory = torch.randn(2, 20, d_model)

        # Embed target tokens
        tgt = embedding(tgt_tokens)
        tgt = pe(tgt)

        # Pass through decoder
        output = decoder(tgt, memory)

        assert output.shape == (2, 15, d_model)
        assert not torch.isnan(output).any()

    def test_full_transformer_pipeline(self):
        """Test embedding in full transformer pipeline."""
        from src.positional_encoding import PositionalEncoding
        from src.encoder import Encoder
        from src.decoder import Decoder

        vocab_size = 1000
        d_model = 256
        src_len, tgt_len = 20, 15

        src_embedding = TransformerEmbedding(vocab_size=vocab_size, d_model=d_model)
        tgt_embedding = TransformerEmbedding(vocab_size=vocab_size, d_model=d_model)
        pe = PositionalEncoding(d_model=d_model, max_seq_len=100, dropout=0.0)
        encoder = Encoder(n_layers=2, d_model=d_model, n_heads=4, d_ff=1024, dropout=0.0)
        decoder = Decoder(n_layers=2, d_model=d_model, n_heads=4, d_ff=1024, dropout=0.0)

        src_tokens = torch.randint(0, vocab_size, (2, src_len))
        tgt_tokens = torch.randint(0, vocab_size, (2, tgt_len))

        # Encode source
        src = src_embedding(src_tokens)
        src = pe(src)
        memory = encoder(src)

        # Decode target
        tgt = tgt_embedding(tgt_tokens)
        tgt = pe(tgt)
        output = decoder(tgt, memory)

        assert output.shape == (2, tgt_len, d_model)
        assert not torch.isnan(output).any()

    def test_gradient_flow_full_pipeline(self):
        """Test gradient flow through full pipeline with embeddings."""
        from src.positional_encoding import PositionalEncoding
        from src.encoder import Encoder

        vocab_size = 100
        d_model = 64

        embedding = TransformerEmbedding(vocab_size=vocab_size, d_model=d_model)
        pe = PositionalEncoding(d_model=d_model, max_seq_len=50, dropout=0.0)
        encoder = Encoder(n_layers=2, d_model=d_model, n_heads=4, d_ff=256, dropout=0.0)

        tokens = torch.randint(0, vocab_size, (2, 10))

        x = embedding(tokens)
        x = pe(x)
        output = encoder(x)

        loss = output.sum()
        loss.backward()

        # Check gradients on embedding
        assert embedding.embedding.weight.grad is not None
        assert not torch.isnan(embedding.embedding.weight.grad).any()

    def test_shared_embedding(self):
        """Test that embedding can be shared between encoder and decoder."""
        from src.positional_encoding import PositionalEncoding
        from src.encoder import Encoder
        from src.decoder import Decoder

        vocab_size = 1000
        d_model = 256

        # Shared embedding for source and target
        shared_embedding = TransformerEmbedding(vocab_size=vocab_size, d_model=d_model)
        pe = PositionalEncoding(d_model=d_model, max_seq_len=100, dropout=0.0)
        encoder = Encoder(n_layers=2, d_model=d_model, n_heads=4, d_ff=1024, dropout=0.0)
        decoder = Decoder(n_layers=2, d_model=d_model, n_heads=4, d_ff=1024, dropout=0.0)

        src_tokens = torch.randint(0, vocab_size, (2, 20))
        tgt_tokens = torch.randint(0, vocab_size, (2, 15))

        # Use same embedding for source and target
        src = shared_embedding(src_tokens)
        src = pe(src)
        memory = encoder(src)

        tgt = shared_embedding(tgt_tokens)
        tgt = pe(tgt)
        output = decoder(tgt, memory)

        assert output.shape == (2, 15, d_model)
        assert not torch.isnan(output).any()
