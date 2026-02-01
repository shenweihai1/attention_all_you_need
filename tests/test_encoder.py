"""
Tests for the Encoder components.
"""

import pytest
import torch
import torch.nn as nn

from src.encoder import EncoderLayer, Encoder
from src.attention import create_padding_mask


class TestEncoderLayer:
    """Tests for the EncoderLayer module."""

    def test_initialization_default(self):
        """Test EncoderLayer initialization with default parameters."""
        layer = EncoderLayer()
        assert layer.d_model == 512
        assert layer.n_heads == 8
        assert layer.d_ff == 2048

    def test_initialization_custom(self):
        """Test EncoderLayer initialization with custom parameters."""
        layer = EncoderLayer(d_model=256, n_heads=4, d_ff=1024, dropout=0.2)
        assert layer.d_model == 256
        assert layer.n_heads == 4
        assert layer.d_ff == 1024

    def test_output_shape(self):
        """Test that output has correct shape."""
        d_model = 512
        layer = EncoderLayer(d_model=d_model, n_heads=8, d_ff=2048)

        batch, seq_len = 2, 10
        x = torch.randn(batch, seq_len, d_model)
        output = layer(x)

        assert output.shape == (batch, seq_len, d_model)

    def test_output_shape_various_sizes(self):
        """Test output shape with various batch and sequence sizes."""
        layer = EncoderLayer(d_model=256, n_heads=4, d_ff=1024)

        test_cases = [
            (1, 1, 256),
            (1, 100, 256),
            (32, 50, 256),
            (4, 512, 256),
        ]

        for batch, seq_len, d_model in test_cases:
            x = torch.randn(batch, seq_len, d_model)
            output = layer(x)
            assert output.shape == (batch, seq_len, d_model)

    def test_with_padding_mask(self):
        """Test EncoderLayer with padding mask."""
        layer = EncoderLayer(d_model=256, n_heads=4, d_ff=1024, dropout=0.0)

        batch, seq_len = 2, 6
        x = torch.randn(batch, seq_len, 256)

        # Create padding mask
        seq = torch.tensor([[1, 2, 3, 0, 0, 0], [1, 2, 3, 4, 0, 0]])
        pad_mask = create_padding_mask(seq, pad_idx=0)

        output = layer(x, src_mask=pad_mask)

        assert output.shape == (batch, seq_len, 256)
        assert not torch.isnan(output).any()

    def test_gradient_flow(self):
        """Test that gradients flow through EncoderLayer."""
        layer = EncoderLayer(d_model=256, n_heads=4, d_ff=1024)
        x = torch.randn(2, 8, 256, requires_grad=True)

        output = layer(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

        # Check gradients on parameters
        for name, param in layer.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"

    def test_has_all_components(self):
        """Test that EncoderLayer has all required components."""
        layer = EncoderLayer(d_model=512, n_heads=8, d_ff=2048)

        # Check sub-modules exist
        assert hasattr(layer, 'self_attention')
        assert hasattr(layer, 'feed_forward')
        assert hasattr(layer, 'norm1')
        assert hasattr(layer, 'norm2')
        assert hasattr(layer, 'dropout1')
        assert hasattr(layer, 'dropout2')

    def test_layer_norm_applied(self):
        """Test that layer normalization is applied."""
        layer = EncoderLayer(d_model=64, n_heads=4, d_ff=256, dropout=0.0)
        layer.eval()

        x = torch.randn(2, 8, 64) * 100  # Large values

        output = layer(x)

        # Output should be normalized (roughly unit variance per position)
        # Not a strict test, but output should be more "tame" than input
        assert output.std() < x.std()

    def test_residual_connection(self):
        """Test that residual connections are used."""
        layer = EncoderLayer(d_model=64, n_heads=4, d_ff=256, dropout=0.0)
        layer.eval()

        # Create input
        x = torch.randn(1, 4, 64)

        # Zero out attention and FFN weights to isolate residual
        with torch.no_grad():
            for param in layer.self_attention.parameters():
                param.zero_()
            for param in layer.feed_forward.parameters():
                param.zero_()

        output = layer(x)

        # With zeroed sublayers, output should be close to normalized input
        # (due to layer norm on residual)
        # Just check that output is valid and not zero
        assert not torch.isnan(output).any()
        assert output.abs().sum() > 0

    def test_eval_mode_deterministic(self):
        """Test that eval mode produces deterministic outputs."""
        layer = EncoderLayer(d_model=256, n_heads=4, d_ff=1024, dropout=0.5)
        layer.eval()

        x = torch.randn(2, 8, 256)
        output1 = layer(x)
        output2 = layer(x)

        assert torch.allclose(output1, output2)

    def test_train_mode_with_dropout(self):
        """Test that training mode applies dropout."""
        torch.manual_seed(42)
        layer = EncoderLayer(d_model=256, n_heads=4, d_ff=1024, dropout=0.5)
        layer.train()

        x = torch.randn(2, 8, 256)
        output1 = layer(x)
        output2 = layer(x)

        # With dropout, outputs should differ
        assert not torch.allclose(output1, output2)

    def test_batch_independence(self):
        """Test that batches are processed independently."""
        layer = EncoderLayer(d_model=256, n_heads=4, d_ff=1024, dropout=0.0)
        layer.eval()

        x = torch.randn(2, 8, 256)

        output_batched = layer(x)
        output0 = layer(x[0:1])
        output1 = layer(x[1:2])

        assert torch.allclose(output_batched[0:1], output0, atol=1e-5)
        assert torch.allclose(output_batched[1:2], output1, atol=1e-5)

    def test_parameter_count(self):
        """Test that the model has reasonable number of parameters."""
        d_model, n_heads, d_ff = 512, 8, 2048
        layer = EncoderLayer(d_model=d_model, n_heads=n_heads, d_ff=d_ff)

        # Multi-head attention: 4 * d_model^2 (W_Q, W_K, W_V, W_O, no bias)
        # Feed-forward: d_model * d_ff + d_ff + d_ff * d_model + d_model
        # Layer norms: 2 * (d_model + d_model)

        mha_params = 4 * d_model * d_model
        ff_params = d_model * d_ff + d_ff + d_ff * d_model + d_model
        ln_params = 4 * d_model  # 2 layer norms, each with weight and bias

        expected_params = mha_params + ff_params + ln_params

        total_params = sum(p.numel() for p in layer.parameters())
        assert total_params == expected_params

    def test_different_configurations(self):
        """Test various d_model, n_heads, d_ff combinations."""
        configs = [
            (128, 4, 512),
            (256, 8, 1024),
            (512, 8, 2048),
            (768, 12, 3072),
        ]

        for d_model, n_heads, d_ff in configs:
            layer = EncoderLayer(d_model=d_model, n_heads=n_heads, d_ff=d_ff)
            x = torch.randn(1, 4, d_model)
            output = layer(x)

            assert output.shape == (1, 4, d_model)
            assert not torch.isnan(output).any()

    def test_extra_repr(self):
        """Test the extra_repr method."""
        layer = EncoderLayer(d_model=512, n_heads=8, d_ff=2048)
        repr_str = layer.extra_repr()

        assert "d_model=512" in repr_str
        assert "n_heads=8" in repr_str
        assert "d_ff=2048" in repr_str

    def test_no_nan_output(self):
        """Test that output contains no NaN values."""
        layer = EncoderLayer(d_model=256, n_heads=4, d_ff=1024)

        # Test with various input values
        test_inputs = [
            torch.randn(2, 8, 256),
            torch.zeros(2, 8, 256),
            torch.ones(2, 8, 256),
            torch.randn(2, 8, 256) * 10,
        ]

        for x in test_inputs:
            output = layer(x)
            assert not torch.isnan(output).any()
            assert not torch.isinf(output).any()


class TestEncoder:
    """Tests for the Encoder module (full encoder stack)."""

    def test_initialization_default(self):
        """Test Encoder initialization with default parameters."""
        encoder = Encoder()
        assert encoder.n_layers == 6
        assert encoder.d_model == 512
        assert encoder.n_heads == 8
        assert encoder.d_ff == 2048

    def test_initialization_custom(self):
        """Test Encoder initialization with custom parameters."""
        encoder = Encoder(n_layers=4, d_model=256, n_heads=4, d_ff=1024, dropout=0.2)
        assert encoder.n_layers == 4
        assert encoder.d_model == 256
        assert encoder.n_heads == 4
        assert encoder.d_ff == 1024

    def test_output_shape(self):
        """Test that output has correct shape."""
        d_model = 512
        encoder = Encoder(n_layers=6, d_model=d_model, n_heads=8, d_ff=2048)

        batch, seq_len = 2, 10
        x = torch.randn(batch, seq_len, d_model)
        output = encoder(x)

        assert output.shape == (batch, seq_len, d_model)

    def test_output_shape_various_sizes(self):
        """Test output shape with various batch and sequence sizes."""
        encoder = Encoder(n_layers=3, d_model=256, n_heads=4, d_ff=1024)

        test_cases = [
            (1, 1, 256),
            (1, 100, 256),
            (32, 50, 256),
            (4, 512, 256),
        ]

        for batch, seq_len, d_model in test_cases:
            x = torch.randn(batch, seq_len, d_model)
            output = encoder(x)
            assert output.shape == (batch, seq_len, d_model)

    def test_has_correct_number_of_layers(self):
        """Test that Encoder has the correct number of layers."""
        for n_layers in [1, 3, 6, 12]:
            encoder = Encoder(n_layers=n_layers, d_model=256, n_heads=4, d_ff=1024)
            assert len(encoder.layers) == n_layers

    def test_with_padding_mask(self):
        """Test Encoder with padding mask."""
        encoder = Encoder(n_layers=3, d_model=256, n_heads=4, d_ff=1024, dropout=0.0)

        batch, seq_len = 2, 6
        x = torch.randn(batch, seq_len, 256)

        # Create padding mask
        seq = torch.tensor([[1, 2, 3, 0, 0, 0], [1, 2, 3, 4, 0, 0]])
        pad_mask = create_padding_mask(seq, pad_idx=0)

        output = encoder(x, src_mask=pad_mask)

        assert output.shape == (batch, seq_len, 256)
        assert not torch.isnan(output).any()

    def test_gradient_flow(self):
        """Test that gradients flow through all encoder layers."""
        encoder = Encoder(n_layers=3, d_model=256, n_heads=4, d_ff=1024)
        x = torch.randn(2, 8, 256, requires_grad=True)

        output = encoder(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

        # Check gradients on parameters in all layers
        for i, layer in enumerate(encoder.layers):
            for name, param in layer.named_parameters():
                assert param.grad is not None, f"No gradient for layer {i} {name}"
                assert not torch.isnan(param.grad).any(), f"NaN gradient for layer {i} {name}"

    def test_has_final_layer_norm(self):
        """Test that Encoder has final layer normalization."""
        encoder = Encoder(n_layers=6, d_model=512, n_heads=8, d_ff=2048)

        assert hasattr(encoder, 'norm')
        assert isinstance(encoder.norm, nn.LayerNorm)
        assert encoder.norm.normalized_shape == (512,)

    def test_eval_mode_deterministic(self):
        """Test that eval mode produces deterministic outputs."""
        encoder = Encoder(n_layers=3, d_model=256, n_heads=4, d_ff=1024, dropout=0.5)
        encoder.eval()

        x = torch.randn(2, 8, 256)
        output1 = encoder(x)
        output2 = encoder(x)

        assert torch.allclose(output1, output2)

    def test_train_mode_with_dropout(self):
        """Test that training mode applies dropout."""
        torch.manual_seed(42)
        encoder = Encoder(n_layers=3, d_model=256, n_heads=4, d_ff=1024, dropout=0.5)
        encoder.train()

        x = torch.randn(2, 8, 256)
        output1 = encoder(x)
        output2 = encoder(x)

        # With dropout, outputs should differ
        assert not torch.allclose(output1, output2)

    def test_batch_independence(self):
        """Test that batches are processed independently."""
        encoder = Encoder(n_layers=3, d_model=256, n_heads=4, d_ff=1024, dropout=0.0)
        encoder.eval()

        x = torch.randn(2, 8, 256)

        output_batched = encoder(x)
        output0 = encoder(x[0:1])
        output1 = encoder(x[1:2])

        assert torch.allclose(output_batched[0:1], output0, atol=1e-5)
        assert torch.allclose(output_batched[1:2], output1, atol=1e-5)

    def test_parameter_count(self):
        """Test that the model has reasonable number of parameters."""
        n_layers, d_model, n_heads, d_ff = 6, 512, 8, 2048
        encoder = Encoder(n_layers=n_layers, d_model=d_model, n_heads=n_heads, d_ff=d_ff)

        # Per encoder layer:
        # Multi-head attention: 4 * d_model^2
        # Feed-forward: d_model * d_ff + d_ff + d_ff * d_model + d_model
        # Layer norms: 4 * d_model (2 norms per layer)
        mha_params = 4 * d_model * d_model
        ff_params = d_model * d_ff + d_ff + d_ff * d_model + d_model
        ln_params = 4 * d_model
        layer_params = mha_params + ff_params + ln_params

        # Total: N layers + final layer norm (2 * d_model)
        expected_params = n_layers * layer_params + 2 * d_model

        total_params = sum(p.numel() for p in encoder.parameters())
        assert total_params == expected_params

    def test_different_configurations(self):
        """Test various n_layers, d_model, n_heads, d_ff combinations."""
        configs = [
            (2, 128, 4, 512),
            (4, 256, 8, 1024),
            (6, 512, 8, 2048),
            (8, 768, 12, 3072),
        ]

        for n_layers, d_model, n_heads, d_ff in configs:
            encoder = Encoder(n_layers=n_layers, d_model=d_model, n_heads=n_heads, d_ff=d_ff)
            x = torch.randn(1, 4, d_model)
            output = encoder(x)

            assert output.shape == (1, 4, d_model)
            assert not torch.isnan(output).any()

    def test_extra_repr(self):
        """Test the extra_repr method."""
        encoder = Encoder(n_layers=6, d_model=512, n_heads=8, d_ff=2048)
        repr_str = encoder.extra_repr()

        assert "n_layers=6" in repr_str
        assert "d_model=512" in repr_str
        assert "n_heads=8" in repr_str
        assert "d_ff=2048" in repr_str

    def test_no_nan_output(self):
        """Test that output contains no NaN values."""
        encoder = Encoder(n_layers=3, d_model=256, n_heads=4, d_ff=1024)

        # Test with various input values
        test_inputs = [
            torch.randn(2, 8, 256),
            torch.zeros(2, 8, 256),
            torch.ones(2, 8, 256),
            torch.randn(2, 8, 256) * 10,
        ]

        for x in test_inputs:
            output = encoder(x)
            assert not torch.isnan(output).any()
            assert not torch.isinf(output).any()

    def test_single_layer_encoder(self):
        """Test encoder with a single layer."""
        encoder = Encoder(n_layers=1, d_model=256, n_heads=4, d_ff=1024, dropout=0.0)

        x = torch.randn(2, 10, 256)
        output = encoder(x)

        assert output.shape == (2, 10, 256)
        assert not torch.isnan(output).any()

    def test_deep_encoder(self):
        """Test encoder with many layers."""
        encoder = Encoder(n_layers=12, d_model=256, n_heads=4, d_ff=1024, dropout=0.0)

        x = torch.randn(2, 10, 256)
        output = encoder(x)

        assert output.shape == (2, 10, 256)
        assert not torch.isnan(output).any()


class TestEncoderIntegration:
    """Integration tests for the full Encoder."""

    def test_with_positional_encoding(self):
        """Test Encoder with positional encoding added to input."""
        from src.positional_encoding import PositionalEncoding

        d_model = 256
        pe = PositionalEncoding(d_model=d_model, max_seq_len=100, dropout=0.0)
        encoder = Encoder(n_layers=3, d_model=d_model, n_heads=4, d_ff=1024, dropout=0.0)

        # Simulate embedded input
        x = torch.randn(2, 50, d_model)

        # Add positional encoding
        x = pe(x)

        # Pass through encoder
        output = encoder(x)

        assert output.shape == (2, 50, d_model)
        assert not torch.isnan(output).any()

    def test_with_embedding_input(self):
        """Test Encoder with embedding layer input."""
        from src.positional_encoding import PositionalEncoding

        vocab_size = 1000
        d_model = 256

        embedding = nn.Embedding(vocab_size, d_model)
        pe = PositionalEncoding(d_model=d_model, max_seq_len=100, dropout=0.0)
        encoder = Encoder(n_layers=3, d_model=d_model, n_heads=4, d_ff=1024, dropout=0.0)

        # Create token indices
        tokens = torch.randint(0, vocab_size, (2, 50))

        # Embed tokens and add positional encoding
        x = embedding(tokens)
        x = pe(x)

        # Pass through encoder
        output = encoder(x)

        assert output.shape == (2, 50, d_model)
        assert not torch.isnan(output).any()

    def test_full_encoder_pipeline(self):
        """Test full encoder pipeline with embedding, PE, and encoder stack."""
        from src.positional_encoding import PositionalEncoding
        import math

        vocab_size = 1000
        d_model = 256
        n_layers = 6
        seq_len = 50

        # Components
        embedding = nn.Embedding(vocab_size, d_model)
        pe = PositionalEncoding(d_model=d_model, max_seq_len=100, dropout=0.1)
        encoder = Encoder(n_layers=n_layers, d_model=d_model, n_heads=4, d_ff=1024, dropout=0.1)

        # Create input
        tokens = torch.randint(0, vocab_size, (2, seq_len))

        # Embed and scale by sqrt(d_model) as per paper
        x = embedding(tokens) * math.sqrt(d_model)

        # Add positional encoding
        x = pe(x)

        # Pass through encoder
        output = encoder(x)

        assert output.shape == (2, seq_len, d_model)
        assert not torch.isnan(output).any()

    def test_gradient_flow_full_pipeline(self):
        """Test gradient flow through full encoder pipeline."""
        from src.positional_encoding import PositionalEncoding
        import math

        vocab_size = 100
        d_model = 64
        n_layers = 2

        embedding = nn.Embedding(vocab_size, d_model)
        pe = PositionalEncoding(d_model=d_model, max_seq_len=50, dropout=0.0)
        encoder = Encoder(n_layers=n_layers, d_model=d_model, n_heads=4, d_ff=256, dropout=0.0)

        tokens = torch.randint(0, vocab_size, (2, 10))
        x = embedding(tokens) * math.sqrt(d_model)
        x = pe(x)
        output = encoder(x)

        loss = output.sum()
        loss.backward()

        # Check gradients on embedding
        assert embedding.weight.grad is not None
        assert not torch.isnan(embedding.weight.grad).any()

        # Check gradients on encoder
        for param in encoder.parameters():
            assert param.grad is not None


class TestEncoderLayerIntegration:
    """Integration tests for EncoderLayer."""

    def test_with_positional_encoding(self):
        """Test EncoderLayer with positional encoding added to input."""
        from src.positional_encoding import PositionalEncoding

        d_model = 256
        pe = PositionalEncoding(d_model=d_model, max_seq_len=100, dropout=0.0)
        layer = EncoderLayer(d_model=d_model, n_heads=4, d_ff=1024, dropout=0.0)

        # Simulate embedded input
        x = torch.randn(2, 50, d_model)

        # Add positional encoding
        x = pe(x)

        # Pass through encoder layer
        output = layer(x)

        assert output.shape == (2, 50, d_model)
        assert not torch.isnan(output).any()

    def test_stacked_encoder_layers(self):
        """Test multiple encoder layers stacked together."""
        d_model = 256
        n_layers = 3

        layers = nn.ModuleList([
            EncoderLayer(d_model=d_model, n_heads=4, d_ff=1024, dropout=0.0)
            for _ in range(n_layers)
        ])

        x = torch.randn(2, 10, d_model)

        for layer in layers:
            x = layer(x)

        assert x.shape == (2, 10, d_model)
        assert not torch.isnan(x).any()

    def test_with_embedding_input(self):
        """Test EncoderLayer with embedding layer input."""
        vocab_size = 1000
        d_model = 256

        embedding = nn.Embedding(vocab_size, d_model)
        layer = EncoderLayer(d_model=d_model, n_heads=4, d_ff=1024, dropout=0.0)

        # Create token indices
        tokens = torch.randint(0, vocab_size, (2, 50))

        # Embed tokens
        x = embedding(tokens)

        # Pass through encoder layer
        output = layer(x)

        assert output.shape == (2, 50, d_model)
        assert not torch.isnan(output).any()
