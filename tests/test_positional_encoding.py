"""
Tests for the Positional Encoding module.
"""

import math
import pytest
import torch
import torch.nn as nn

from src.positional_encoding import PositionalEncoding


class TestPositionalEncoding:
    """Tests for the PositionalEncoding module."""

    def test_initialization_default(self):
        """Test PositionalEncoding initialization with default parameters."""
        pe = PositionalEncoding()
        assert pe.d_model == 512
        assert pe.max_seq_len == 5000
        assert pe.dropout is None

    def test_initialization_custom(self):
        """Test PositionalEncoding initialization with custom parameters."""
        pe = PositionalEncoding(d_model=256, max_seq_len=1000, dropout=0.1)
        assert pe.d_model == 256
        assert pe.max_seq_len == 1000
        assert pe.dropout is not None
        assert pe.dropout.p == 0.1

    def test_output_shape(self):
        """Test that output has correct shape."""
        d_model = 512
        pe = PositionalEncoding(d_model=d_model, max_seq_len=1000)

        batch, seq_len = 2, 100
        x = torch.randn(batch, seq_len, d_model)
        output = pe(x)

        assert output.shape == (batch, seq_len, d_model)

    def test_output_shape_various_sizes(self):
        """Test output shape with various batch and sequence sizes."""
        pe = PositionalEncoding(d_model=256, max_seq_len=500)

        test_cases = [
            (1, 1, 256),
            (1, 100, 256),
            (32, 50, 256),
            (4, 500, 256),
        ]

        for batch, seq_len, d_model in test_cases:
            x = torch.randn(batch, seq_len, d_model)
            output = pe(x)
            assert output.shape == (batch, seq_len, d_model)

    def test_encoding_is_added(self):
        """Test that positional encoding is actually added to input."""
        pe = PositionalEncoding(d_model=64, max_seq_len=100, dropout=0.0)

        x = torch.zeros(1, 10, 64)  # Zero input
        output = pe(x)

        # Output should equal the positional encoding (since input is zero)
        expected = pe.pe[:, :10, :]
        assert torch.allclose(output, expected)

    def test_sinusoidal_pattern_even_indices(self):
        """Test that even indices use sine function."""
        d_model = 64
        max_seq_len = 100
        pe = PositionalEncoding(d_model=d_model, max_seq_len=max_seq_len, dropout=0.0)

        encoding = pe.get_encoding(max_seq_len)

        # Check first position (pos=0): sin(0) = 0 for all even dimensions
        assert torch.allclose(encoding[0, 0::2], torch.zeros(d_model // 2), atol=1e-6)

        # Check that values at pos=1, dim=0 matches sin formula
        # PE(1, 0) = sin(1 / 10000^(0/d_model)) = sin(1)
        expected_val = math.sin(1.0)
        assert abs(encoding[1, 0].item() - expected_val) < 1e-5

    def test_sinusoidal_pattern_odd_indices(self):
        """Test that odd indices use cosine function."""
        d_model = 64
        max_seq_len = 100
        pe = PositionalEncoding(d_model=d_model, max_seq_len=max_seq_len, dropout=0.0)

        encoding = pe.get_encoding(max_seq_len)

        # Check first position (pos=0): cos(0) = 1 for all odd dimensions
        assert torch.allclose(encoding[0, 1::2], torch.ones(d_model // 2), atol=1e-6)

        # Check that values at pos=1, dim=1 matches cos formula
        # PE(1, 1) = cos(1 / 10000^(0/d_model)) = cos(1)
        expected_val = math.cos(1.0)
        assert abs(encoding[1, 1].item() - expected_val) < 1e-5

    def test_encoding_registered_as_buffer(self):
        """Test that positional encoding is registered as a buffer."""
        pe = PositionalEncoding(d_model=64, max_seq_len=100)

        # Check that 'pe' is in the module's buffers
        assert "pe" in dict(pe.named_buffers())

        # Check that it's not in parameters
        assert "pe" not in dict(pe.named_parameters())

    def test_encoding_buffer_shape(self):
        """Test that the pre-computed encoding buffer has correct shape."""
        d_model = 256
        max_seq_len = 1000
        pe = PositionalEncoding(d_model=d_model, max_seq_len=max_seq_len)

        # Buffer should have shape (1, max_seq_len, d_model)
        assert pe.pe.shape == (1, max_seq_len, d_model)

    def test_sequence_too_long_raises_error(self):
        """Test that sequence exceeding max_seq_len raises ValueError."""
        pe = PositionalEncoding(d_model=64, max_seq_len=100)

        x = torch.randn(1, 150, 64)  # seq_len=150 > max_seq_len=100

        with pytest.raises(ValueError, match="exceeds maximum"):
            pe(x)

    def test_get_encoding_too_long_raises_error(self):
        """Test that get_encoding with seq_len > max raises ValueError."""
        pe = PositionalEncoding(d_model=64, max_seq_len=100)

        with pytest.raises(ValueError, match="exceeds maximum"):
            pe.get_encoding(150)

    def test_get_encoding_shape(self):
        """Test that get_encoding returns correct shape."""
        pe = PositionalEncoding(d_model=128, max_seq_len=500)

        encoding = pe.get_encoding(100)
        assert encoding.shape == (100, 128)

    def test_eval_mode_deterministic(self):
        """Test that eval mode produces deterministic outputs."""
        pe = PositionalEncoding(d_model=64, max_seq_len=100, dropout=0.5)
        pe.eval()

        x = torch.randn(2, 50, 64)
        output1 = pe(x)
        output2 = pe(x)

        assert torch.allclose(output1, output2)

    def test_train_mode_with_dropout(self):
        """Test that training mode applies dropout."""
        torch.manual_seed(42)
        pe = PositionalEncoding(d_model=64, max_seq_len=100, dropout=0.5)
        pe.train()

        x = torch.randn(2, 50, 64)
        output1 = pe(x)
        output2 = pe(x)

        # With dropout, outputs should differ
        assert not torch.allclose(output1, output2)

    def test_no_dropout_same_output(self):
        """Test that without dropout, outputs are deterministic."""
        pe = PositionalEncoding(d_model=64, max_seq_len=100, dropout=0.0)

        x = torch.randn(2, 50, 64)
        output1 = pe(x)
        output2 = pe(x)

        assert torch.allclose(output1, output2)

    def test_different_positions_different_encodings(self):
        """Test that different positions have different encodings."""
        pe = PositionalEncoding(d_model=64, max_seq_len=100, dropout=0.0)

        encoding = pe.get_encoding(10)

        # Each position should have a unique encoding
        for i in range(10):
            for j in range(i + 1, 10):
                assert not torch.allclose(encoding[i], encoding[j])

    def test_batch_independence(self):
        """Test that batches are processed independently."""
        pe = PositionalEncoding(d_model=64, max_seq_len=100, dropout=0.0)

        x = torch.randn(2, 50, 64)

        output_batched = pe(x)
        output0 = pe(x[0:1])
        output1 = pe(x[1:2])

        assert torch.allclose(output_batched[0:1], output0, atol=1e-6)
        assert torch.allclose(output_batched[1:2], output1, atol=1e-6)

    def test_encoding_values_bounded(self):
        """Test that encoding values are bounded (from sin/cos)."""
        pe = PositionalEncoding(d_model=512, max_seq_len=5000)

        encoding = pe.get_encoding(5000)

        # sin and cos are bounded between -1 and 1
        assert encoding.min() >= -1.0
        assert encoding.max() <= 1.0

    def test_odd_d_model(self):
        """Test that odd d_model is handled correctly."""
        pe = PositionalEncoding(d_model=63, max_seq_len=100, dropout=0.0)

        x = torch.randn(1, 50, 63)
        output = pe(x)

        assert output.shape == (1, 50, 63)
        assert not torch.isnan(output).any()

    def test_gradient_does_not_flow_to_encoding(self):
        """Test that gradients do not flow to positional encoding buffer."""
        pe = PositionalEncoding(d_model=64, max_seq_len=100, dropout=0.0)

        x = torch.randn(2, 50, 64, requires_grad=True)
        output = pe(x)
        loss = output.sum()
        loss.backward()

        # Gradient should flow to input
        assert x.grad is not None

        # But pe buffer should not have gradient
        assert pe.pe.grad is None or not pe.pe.requires_grad

    def test_extra_repr(self):
        """Test the extra_repr method."""
        pe = PositionalEncoding(d_model=512, max_seq_len=1000)
        repr_str = pe.extra_repr()

        assert "d_model=512" in repr_str
        assert "max_seq_len=1000" in repr_str

    def test_encoding_consistency_across_calls(self):
        """Test that the same positions get the same encoding across calls."""
        pe = PositionalEncoding(d_model=64, max_seq_len=100, dropout=0.0)

        x1 = torch.randn(1, 50, 64)
        x2 = torch.randn(1, 50, 64)

        # The positional encoding added should be the same
        pe_added_1 = pe(torch.zeros(1, 50, 64))
        pe_added_2 = pe(torch.zeros(1, 50, 64))

        assert torch.allclose(pe_added_1, pe_added_2)


class TestPositionalEncodingIntegration:
    """Integration tests for PositionalEncoding."""

    def test_with_embedding_layer(self):
        """Test PositionalEncoding with an embedding layer."""
        vocab_size = 1000
        d_model = 256
        seq_len = 50

        embedding = nn.Embedding(vocab_size, d_model)
        pe = PositionalEncoding(d_model=d_model, max_seq_len=100, dropout=0.0)

        # Create token indices
        tokens = torch.randint(0, vocab_size, (2, seq_len))

        # Embed tokens
        embedded = embedding(tokens)

        # Add positional encoding
        output = pe(embedded)

        assert output.shape == (2, seq_len, d_model)
        assert not torch.isnan(output).any()

    def test_in_transformer_pattern(self):
        """Test PositionalEncoding in a transformer-like pattern."""
        from src.attention import MultiHeadAttention
        from src.feedforward import PositionwiseFeedForward

        d_model = 256
        n_heads = 4

        pe = PositionalEncoding(d_model=d_model, max_seq_len=100, dropout=0.0)
        mha = MultiHeadAttention(d_model=d_model, n_heads=n_heads, dropout=0.0)
        ffn = PositionwiseFeedForward(d_model=d_model, d_ff=1024, dropout=0.0)
        norm1 = nn.LayerNorm(d_model)
        norm2 = nn.LayerNorm(d_model)

        # Input embeddings
        x = torch.randn(2, 50, d_model)

        # Add positional encoding
        x = pe(x)

        # Self-attention sub-layer
        attn_out, _ = mha(x, x, x)
        x = norm1(x + attn_out)

        # Feed-forward sub-layer
        ffn_out = ffn(x)
        x = norm2(x + ffn_out)

        assert x.shape == (2, 50, d_model)
        assert not torch.isnan(x).any()

    def test_move_to_device(self):
        """Test that positional encoding buffer moves with module to device."""
        pe = PositionalEncoding(d_model=64, max_seq_len=100)

        # Initially on CPU
        assert pe.pe.device.type == "cpu"

        # Move to CPU explicitly (no GPU available in test environment)
        pe = pe.to("cpu")
        assert pe.pe.device.type == "cpu"

    def test_save_and_load_state_dict(self):
        """Test that positional encoding is saved/loaded with state dict."""
        pe1 = PositionalEncoding(d_model=64, max_seq_len=100)

        # Save state dict
        state_dict = pe1.state_dict()

        # Create new module and load state
        pe2 = PositionalEncoding(d_model=64, max_seq_len=100)
        pe2.load_state_dict(state_dict)

        # Encodings should match
        assert torch.allclose(pe1.pe, pe2.pe)

    def test_wavelength_varies_with_dimension(self):
        """Test that the wavelength of sinusoids varies with dimension index."""
        pe = PositionalEncoding(d_model=64, max_seq_len=1000, dropout=0.0)
        encoding = pe.get_encoding(1000)

        # Lower dimension indices should have shorter wavelengths (faster oscillation)
        # Higher dimension indices should have longer wavelengths (slower oscillation)

        # Check variance of first dimension vs last dimension across positions
        # Higher dimensions should have less variance as they oscillate more slowly
        var_dim_0 = encoding[:, 0].var().item()
        var_dim_last = encoding[:, -2].var().item()  # Use -2 for even index

        # The first dimension should complete many cycles and have high variance
        assert var_dim_0 > 0.1

        # The last dimension oscillates very slowly (wavelength ~ 10000)
        # so over 1000 positions it won't complete a full cycle
        # Just verify it's not constant (has some variance)
        assert var_dim_last > 0.0

        # Key property: lower dimensions should have higher variance due to
        # completing more oscillation cycles
        assert var_dim_0 > var_dim_last
