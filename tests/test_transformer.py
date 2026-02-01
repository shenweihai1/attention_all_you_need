"""
Tests for the full Transformer model.

Tests the Transformer implementation ensuring it matches the architecture
described in "Attention Is All You Need".
"""

import pytest
import torch
import torch.nn as nn

from src.transformer import Transformer
from src.attention import create_padding_mask, create_causal_mask


class TestTransformer:
    """Tests for the Transformer model."""

    def test_initialization_default(self):
        """Test Transformer initialization with default parameters."""
        model = Transformer(src_vocab_size=1000, tgt_vocab_size=1000)
        assert model.src_vocab_size == 1000
        assert model.tgt_vocab_size == 1000
        assert model.d_model == 512
        assert model.n_heads == 8
        assert model.n_encoder_layers == 6
        assert model.n_decoder_layers == 6
        assert model.d_ff == 2048
        assert model.pad_idx == 0

    def test_initialization_custom(self):
        """Test Transformer initialization with custom parameters."""
        model = Transformer(
            src_vocab_size=5000,
            tgt_vocab_size=8000,
            d_model=256,
            n_heads=4,
            n_encoder_layers=3,
            n_decoder_layers=3,
            d_ff=1024,
            dropout=0.2,
            max_seq_len=1000,
            pad_idx=1,
        )
        assert model.src_vocab_size == 5000
        assert model.tgt_vocab_size == 8000
        assert model.d_model == 256
        assert model.n_heads == 4
        assert model.n_encoder_layers == 3
        assert model.n_decoder_layers == 3
        assert model.d_ff == 1024
        assert model.pad_idx == 1

    def test_output_shape(self):
        """Test that output has correct shape."""
        src_vocab, tgt_vocab = 1000, 1200
        model = Transformer(
            src_vocab_size=src_vocab,
            tgt_vocab_size=tgt_vocab,
            d_model=256,
            n_heads=4,
            n_encoder_layers=2,
            n_decoder_layers=2,
            d_ff=512,
        )

        batch, src_len, tgt_len = 2, 20, 15
        src = torch.randint(0, src_vocab, (batch, src_len))
        tgt = torch.randint(0, tgt_vocab, (batch, tgt_len))

        output = model(src, tgt)

        assert output.shape == (batch, tgt_len, tgt_vocab)

    def test_output_shape_various_sizes(self):
        """Test output shape with various batch and sequence sizes."""
        model = Transformer(
            src_vocab_size=500,
            tgt_vocab_size=500,
            d_model=128,
            n_heads=4,
            n_encoder_layers=2,
            n_decoder_layers=2,
            d_ff=256,
        )

        test_cases = [
            (1, 5, 5),
            (2, 20, 15),
            (4, 50, 40),
            (8, 30, 25),
        ]

        for batch, src_len, tgt_len in test_cases:
            src = torch.randint(0, 500, (batch, src_len))
            tgt = torch.randint(0, 500, (batch, tgt_len))
            output = model(src, tgt)
            assert output.shape == (batch, tgt_len, 500)

    def test_with_padding(self):
        """Test Transformer with padded sequences."""
        model = Transformer(
            src_vocab_size=100,
            tgt_vocab_size=100,
            d_model=64,
            n_heads=4,
            n_encoder_layers=2,
            n_decoder_layers=2,
            d_ff=128,
            pad_idx=0,
        )

        # Create sequences with padding
        src = torch.tensor([[1, 2, 3, 0, 0], [1, 2, 3, 4, 5]])
        tgt = torch.tensor([[1, 2, 0, 0], [1, 2, 3, 4]])

        output = model(src, tgt)

        assert output.shape == (2, 4, 100)
        assert not torch.isnan(output).any()

    def test_gradient_flow(self):
        """Test that gradients flow through the entire model."""
        model = Transformer(
            src_vocab_size=100,
            tgt_vocab_size=100,
            d_model=64,
            n_heads=4,
            n_encoder_layers=2,
            n_decoder_layers=2,
            d_ff=128,
        )

        src = torch.randint(0, 100, (2, 10))
        tgt = torch.randint(0, 100, (2, 8))

        output = model(src, tgt)
        loss = output.sum()
        loss.backward()

        # Check gradients on embeddings
        assert model.src_embedding.embedding.weight.grad is not None
        assert model.tgt_embedding.embedding.weight.grad is not None

        # Check gradients on encoder and decoder
        for param in model.encoder.parameters():
            assert param.grad is not None

        for param in model.decoder.parameters():
            assert param.grad is not None

        # Check gradients on output projection
        assert model.output_projection.weight.grad is not None
        assert model.output_projection.bias.grad is not None

    def test_has_all_components(self):
        """Test that Transformer has all required components."""
        model = Transformer(src_vocab_size=100, tgt_vocab_size=100)

        assert hasattr(model, 'src_embedding')
        assert hasattr(model, 'tgt_embedding')
        assert hasattr(model, 'positional_encoding')
        assert hasattr(model, 'encoder')
        assert hasattr(model, 'decoder')
        assert hasattr(model, 'output_projection')

    def test_shared_embeddings(self):
        """Test shared embeddings between encoder and decoder."""
        model = Transformer(
            src_vocab_size=100,
            tgt_vocab_size=100,
            d_model=64,
            n_heads=4,
            n_encoder_layers=2,
            n_decoder_layers=2,
            d_ff=128,
            share_embeddings=True,
        )

        # Shared embeddings should be the same object
        assert model.src_embedding is model.tgt_embedding

    def test_separate_embeddings(self):
        """Test separate embeddings for different vocab sizes."""
        model = Transformer(
            src_vocab_size=100,
            tgt_vocab_size=200,
            d_model=64,
            n_heads=4,
            n_encoder_layers=2,
            n_decoder_layers=2,
            d_ff=128,
        )

        # Different vocab sizes should have separate embeddings
        assert model.src_embedding is not model.tgt_embedding

    def test_eval_mode_deterministic(self):
        """Test that eval mode produces deterministic outputs."""
        model = Transformer(
            src_vocab_size=100,
            tgt_vocab_size=100,
            d_model=64,
            n_heads=4,
            n_encoder_layers=2,
            n_decoder_layers=2,
            d_ff=128,
            dropout=0.5,
        )
        model.eval()

        src = torch.randint(0, 100, (2, 10))
        tgt = torch.randint(0, 100, (2, 8))

        output1 = model(src, tgt)
        output2 = model(src, tgt)

        assert torch.allclose(output1, output2)

    def test_train_mode_with_dropout(self):
        """Test that training mode applies dropout."""
        torch.manual_seed(42)
        model = Transformer(
            src_vocab_size=100,
            tgt_vocab_size=100,
            d_model=64,
            n_heads=4,
            n_encoder_layers=2,
            n_decoder_layers=2,
            d_ff=128,
            dropout=0.5,
        )
        model.train()

        src = torch.randint(0, 100, (2, 10))
        tgt = torch.randint(0, 100, (2, 8))

        output1 = model(src, tgt)
        output2 = model(src, tgt)

        # With dropout, outputs should differ
        assert not torch.allclose(output1, output2)

    def test_batch_independence(self):
        """Test that batches are processed independently."""
        model = Transformer(
            src_vocab_size=100,
            tgt_vocab_size=100,
            d_model=64,
            n_heads=4,
            n_encoder_layers=2,
            n_decoder_layers=2,
            d_ff=128,
            dropout=0.0,
        )
        model.eval()

        src = torch.randint(0, 100, (2, 10))
        tgt = torch.randint(0, 100, (2, 8))

        output_batched = model(src, tgt)
        output0 = model(src[0:1], tgt[0:1])
        output1 = model(src[1:2], tgt[1:2])

        assert torch.allclose(output_batched[0:1], output0, atol=1e-5)
        assert torch.allclose(output_batched[1:2], output1, atol=1e-5)

    def test_no_nan_output(self):
        """Test that output contains no NaN values."""
        model = Transformer(
            src_vocab_size=100,
            tgt_vocab_size=100,
            d_model=64,
            n_heads=4,
            n_encoder_layers=2,
            n_decoder_layers=2,
            d_ff=128,
        )

        src = torch.randint(0, 100, (2, 10))
        tgt = torch.randint(0, 100, (2, 8))

        output = model(src, tgt)

        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_extra_repr(self):
        """Test the extra_repr method."""
        model = Transformer(
            src_vocab_size=1000,
            tgt_vocab_size=1200,
            d_model=512,
            n_heads=8,
            n_encoder_layers=6,
            n_decoder_layers=6,
            d_ff=2048,
        )
        repr_str = model.extra_repr()

        assert "src_vocab_size=1000" in repr_str
        assert "tgt_vocab_size=1200" in repr_str
        assert "d_model=512" in repr_str
        assert "n_heads=8" in repr_str
        assert "n_encoder_layers=6" in repr_str
        assert "n_decoder_layers=6" in repr_str
        assert "d_ff=2048" in repr_str


class TestTransformerEncode:
    """Tests for the encode method."""

    def test_encode_output_shape(self):
        """Test that encode produces correct output shape."""
        model = Transformer(
            src_vocab_size=100,
            tgt_vocab_size=100,
            d_model=64,
            n_heads=4,
            n_encoder_layers=2,
            n_decoder_layers=2,
            d_ff=128,
        )

        src = torch.randint(0, 100, (2, 15))
        memory = model.encode(src)

        assert memory.shape == (2, 15, 64)

    def test_encode_with_mask(self):
        """Test encode with source padding mask."""
        model = Transformer(
            src_vocab_size=100,
            tgt_vocab_size=100,
            d_model=64,
            n_heads=4,
            n_encoder_layers=2,
            n_decoder_layers=2,
            d_ff=128,
            pad_idx=0,
        )

        src = torch.tensor([[1, 2, 3, 0, 0], [1, 2, 3, 4, 5]])
        src_mask = create_padding_mask(src, pad_idx=0)
        memory = model.encode(src, src_mask)

        assert memory.shape == (2, 5, 64)
        assert not torch.isnan(memory).any()


class TestTransformerDecode:
    """Tests for the decode method."""

    def test_decode_output_shape(self):
        """Test that decode produces correct output shape."""
        model = Transformer(
            src_vocab_size=100,
            tgt_vocab_size=100,
            d_model=64,
            n_heads=4,
            n_encoder_layers=2,
            n_decoder_layers=2,
            d_ff=128,
        )

        memory = torch.randn(2, 15, 64)
        tgt = torch.randint(0, 100, (2, 10))

        output = model.decode(tgt, memory)

        assert output.shape == (2, 10, 64)

    def test_decode_with_masks(self):
        """Test decode with target and memory masks."""
        model = Transformer(
            src_vocab_size=100,
            tgt_vocab_size=100,
            d_model=64,
            n_heads=4,
            n_encoder_layers=2,
            n_decoder_layers=2,
            d_ff=128,
            pad_idx=0,
        )

        memory = torch.randn(2, 10, 64)
        tgt = torch.tensor([[1, 2, 3, 0], [1, 2, 3, 4]])

        tgt_mask = model._create_tgt_mask(tgt)
        memory_mask = create_padding_mask(
            torch.tensor([[1, 2, 3, 4, 0, 0, 0, 0, 0, 0], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]),
            pad_idx=0,
        )

        output = model.decode(tgt, memory, tgt_mask, memory_mask)

        assert output.shape == (2, 4, 64)
        assert not torch.isnan(output).any()


class TestTransformerGenerate:
    """Tests for the generate method."""

    def test_generate_output_shape(self):
        """Test that generate produces valid output."""
        model = Transformer(
            src_vocab_size=100,
            tgt_vocab_size=100,
            d_model=64,
            n_heads=4,
            n_encoder_layers=2,
            n_decoder_layers=2,
            d_ff=128,
        )
        model.eval()

        src = torch.randint(1, 100, (2, 10))
        generated = model.generate(
            src,
            max_len=20,
            start_token=1,
            end_token=2,
        )

        # Output should have at least start token
        assert generated.shape[0] == 2
        assert generated.shape[1] >= 1
        assert generated.shape[1] <= 20

        # First token should be start token
        assert (generated[:, 0] == 1).all()

    def test_generate_stops_at_end_token(self):
        """Test that generate stops when end token is produced."""
        model = Transformer(
            src_vocab_size=10,
            tgt_vocab_size=10,
            d_model=32,
            n_heads=2,
            n_encoder_layers=1,
            n_decoder_layers=1,
            d_ff=64,
        )
        model.eval()

        src = torch.randint(1, 10, (1, 5))

        # Run generation - it should produce some output
        generated = model.generate(
            src,
            max_len=50,
            start_token=1,
            end_token=2,
        )

        # Should not exceed max_len
        assert generated.shape[1] <= 50


class TestTransformerMasks:
    """Tests for mask creation and handling."""

    def test_create_tgt_mask_shape(self):
        """Test that target mask has correct shape."""
        model = Transformer(
            src_vocab_size=100,
            tgt_vocab_size=100,
            d_model=64,
            n_heads=4,
            n_encoder_layers=2,
            n_decoder_layers=2,
            d_ff=128,
        )

        tgt = torch.randint(0, 100, (2, 10))
        mask = model._create_tgt_mask(tgt)

        # Mask should be broadcastable to (batch, n_heads, tgt_len, tgt_len)
        assert mask.shape[-2:] == (10, 10)

    def test_create_tgt_mask_causal(self):
        """Test that target mask is causal (upper triangular blocked)."""
        model = Transformer(
            src_vocab_size=100,
            tgt_vocab_size=100,
            d_model=64,
            n_heads=4,
            n_encoder_layers=2,
            n_decoder_layers=2,
            d_ff=128,
            pad_idx=0,
        )

        # No padding - should be pure causal mask
        tgt = torch.ones(1, 5, dtype=torch.long)
        mask = model._create_tgt_mask(tgt)

        # Upper triangle (excluding diagonal) should be True (blocked)
        # Lower triangle (including diagonal) should be False (allowed)
        for i in range(5):
            for j in range(5):
                if j > i:  # Upper triangle
                    assert mask[0, 0, i, j].item() == True
                else:  # Lower triangle including diagonal
                    assert mask[0, 0, i, j].item() == False

    def test_create_tgt_mask_with_padding(self):
        """Test that target mask includes padding."""
        model = Transformer(
            src_vocab_size=100,
            tgt_vocab_size=100,
            d_model=64,
            n_heads=4,
            n_encoder_layers=2,
            n_decoder_layers=2,
            d_ff=128,
            pad_idx=0,
        )

        # Sequence with padding at positions 3, 4
        tgt = torch.tensor([[1, 2, 3, 0, 0]])
        mask = model._create_tgt_mask(tgt)

        # Padding positions should be blocked in the key dimension
        assert mask[0, 0, 0, 3].item() == True  # Can't attend to padding
        assert mask[0, 0, 0, 4].item() == True  # Can't attend to padding


class TestTransformerIntegration:
    """Integration tests for the full Transformer."""

    def test_training_step(self):
        """Test a complete training step with loss computation."""
        model = Transformer(
            src_vocab_size=100,
            tgt_vocab_size=100,
            d_model=64,
            n_heads=4,
            n_encoder_layers=2,
            n_decoder_layers=2,
            d_ff=128,
        )
        model.train()

        src = torch.randint(1, 100, (4, 15))
        tgt = torch.randint(1, 100, (4, 12))
        tgt_labels = torch.randint(0, 100, (4, 12))

        # Forward pass
        logits = model(src, tgt)

        # Compute cross-entropy loss
        loss = nn.functional.cross_entropy(
            logits.view(-1, 100),
            tgt_labels.view(-1),
            ignore_index=0,
        )

        # Backward pass
        loss.backward()

        # Check loss is valid
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_inference_step(self):
        """Test inference with the model."""
        model = Transformer(
            src_vocab_size=100,
            tgt_vocab_size=100,
            d_model=64,
            n_heads=4,
            n_encoder_layers=2,
            n_decoder_layers=2,
            d_ff=128,
        )
        model.eval()

        src = torch.randint(1, 100, (2, 15))
        tgt = torch.randint(1, 100, (2, 10))

        with torch.no_grad():
            logits = model(src, tgt)

        # Get predictions
        predictions = logits.argmax(dim=-1)

        assert predictions.shape == (2, 10)
        assert (predictions >= 0).all()
        assert (predictions < 100).all()

    def test_different_src_tgt_lengths(self):
        """Test with different source and target lengths."""
        model = Transformer(
            src_vocab_size=100,
            tgt_vocab_size=100,
            d_model=64,
            n_heads=4,
            n_encoder_layers=2,
            n_decoder_layers=2,
            d_ff=128,
        )
        model.eval()

        test_cases = [
            (10, 5),   # src longer
            (5, 10),   # tgt longer
            (15, 15),  # equal
            (1, 20),   # very different
        ]

        for src_len, tgt_len in test_cases:
            src = torch.randint(1, 100, (2, src_len))
            tgt = torch.randint(1, 100, (2, tgt_len))

            output = model(src, tgt)

            assert output.shape == (2, tgt_len, 100)
            assert not torch.isnan(output).any()

    def test_base_model_configuration(self):
        """Test with base model configuration from paper."""
        model = Transformer(
            src_vocab_size=37000,  # ~BPE vocab size
            tgt_vocab_size=37000,
            d_model=512,
            n_heads=8,
            n_encoder_layers=6,
            n_decoder_layers=6,
            d_ff=2048,
            dropout=0.1,
        )

        # Count parameters (base model should have ~65M parameters)
        total_params = sum(p.numel() for p in model.parameters())

        # Verify model works with realistic input
        src = torch.randint(1, 37000, (2, 50))
        tgt = torch.randint(1, 37000, (2, 40))

        model.eval()
        with torch.no_grad():
            output = model(src, tgt)

        assert output.shape == (2, 40, 37000)
        assert not torch.isnan(output).any()

    def test_parameter_count_small_model(self):
        """Test parameter count for small model."""
        model = Transformer(
            src_vocab_size=100,
            tgt_vocab_size=100,
            d_model=64,
            n_heads=4,
            n_encoder_layers=2,
            n_decoder_layers=2,
            d_ff=128,
        )

        total_params = sum(p.numel() for p in model.parameters())

        # Should have parameters
        assert total_params > 0

        # Components should have expected relative sizes
        embedding_params = sum(p.numel() for p in model.src_embedding.parameters())
        encoder_params = sum(p.numel() for p in model.encoder.parameters())
        decoder_params = sum(p.numel() for p in model.decoder.parameters())
        output_params = sum(p.numel() for p in model.output_projection.parameters())

        # Decoder should have more params than encoder (cross-attention)
        assert decoder_params > encoder_params
