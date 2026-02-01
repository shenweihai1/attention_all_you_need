"""
Tests for the Position-wise Feed-Forward Network module.
"""

import pytest
import torch
import torch.nn as nn

from src.feedforward import PositionwiseFeedForward


class TestPositionwiseFeedForward:
    """Tests for the PositionwiseFeedForward module."""

    def test_initialization_default(self):
        """Test FFN initialization with default parameters."""
        ffn = PositionwiseFeedForward()
        assert ffn.d_model == 512
        assert ffn.d_ff == 2048
        assert ffn.dropout is None

    def test_initialization_custom(self):
        """Test FFN initialization with custom parameters."""
        ffn = PositionwiseFeedForward(d_model=256, d_ff=1024, dropout=0.1)
        assert ffn.d_model == 256
        assert ffn.d_ff == 1024
        assert ffn.dropout is not None
        assert ffn.dropout.p == 0.1

    def test_output_shape(self):
        """Test that output has correct shape."""
        d_model = 512
        ffn = PositionwiseFeedForward(d_model=d_model, d_ff=2048)

        batch, seq_len = 2, 10
        x = torch.randn(batch, seq_len, d_model)
        output = ffn(x)

        assert output.shape == (batch, seq_len, d_model)

    def test_output_shape_various_sizes(self):
        """Test output shape with various batch and sequence sizes."""
        ffn = PositionwiseFeedForward(d_model=256, d_ff=1024)

        test_cases = [
            (1, 1, 256),
            (1, 100, 256),
            (32, 50, 256),
            (4, 512, 256),
        ]

        for batch, seq_len, d_model in test_cases:
            x = torch.randn(batch, seq_len, d_model)
            output = ffn(x)
            assert output.shape == (batch, seq_len, d_model)

    def test_position_wise_application(self):
        """Test that FFN is applied independently to each position."""
        ffn = PositionwiseFeedForward(d_model=64, d_ff=128, dropout=0.0)
        ffn.eval()

        # Create input with two positions
        x = torch.randn(1, 2, 64)

        # Get output for both positions together
        output_together = ffn(x)

        # Get output for each position separately
        output_pos0 = ffn(x[:, 0:1, :])
        output_pos1 = ffn(x[:, 1:2, :])

        # Outputs should match
        assert torch.allclose(output_together[:, 0:1, :], output_pos0, atol=1e-6)
        assert torch.allclose(output_together[:, 1:2, :], output_pos1, atol=1e-6)

    def test_gradient_flow(self):
        """Test that gradients flow through FFN."""
        ffn = PositionwiseFeedForward(d_model=128, d_ff=512)
        x = torch.randn(2, 8, 128, requires_grad=True)

        output = ffn(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

        # Check gradients on parameters
        for name, param in ffn.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"

    def test_parameter_count(self):
        """Test that the model has the correct number of parameters."""
        d_model, d_ff = 512, 2048
        ffn = PositionwiseFeedForward(d_model=d_model, d_ff=d_ff)

        # linear1: d_model * d_ff + d_ff (weights + bias)
        # linear2: d_ff * d_model + d_model (weights + bias)
        expected_params = (d_model * d_ff + d_ff) + (d_ff * d_model + d_model)

        total_params = sum(p.numel() for p in ffn.parameters())
        assert total_params == expected_params

    def test_linear_layers_exist(self):
        """Test that linear layers are properly created."""
        ffn = PositionwiseFeedForward(d_model=256, d_ff=1024)

        assert hasattr(ffn, 'linear1')
        assert hasattr(ffn, 'linear2')
        assert isinstance(ffn.linear1, nn.Linear)
        assert isinstance(ffn.linear2, nn.Linear)

        # Check dimensions
        assert ffn.linear1.in_features == 256
        assert ffn.linear1.out_features == 1024
        assert ffn.linear2.in_features == 1024
        assert ffn.linear2.out_features == 256

    def test_relu_activation(self):
        """Test that ReLU activation is applied (negative values become zero)."""
        ffn = PositionwiseFeedForward(d_model=64, d_ff=128, dropout=0.0)

        # Set linear1 weights to identity-like to make hidden values predictable
        with torch.no_grad():
            ffn.linear1.weight.fill_(0)
            ffn.linear1.bias.fill_(-1)  # All hidden values will be -1 before ReLU

        x = torch.ones(1, 1, 64)
        output = ffn(x)

        # Since ReLU(-1) = 0, and linear2(0) = bias, output should be linear2 bias
        expected = ffn.linear2.bias.unsqueeze(0).unsqueeze(0)
        assert torch.allclose(output, expected, atol=1e-6)

    def test_eval_mode_deterministic(self):
        """Test that eval mode produces deterministic outputs."""
        ffn = PositionwiseFeedForward(d_model=128, d_ff=512, dropout=0.5)
        ffn.eval()

        x = torch.randn(2, 8, 128)
        output1 = ffn(x)
        output2 = ffn(x)

        assert torch.allclose(output1, output2)

    def test_train_mode_with_dropout(self):
        """Test that training mode applies dropout."""
        torch.manual_seed(42)
        ffn = PositionwiseFeedForward(d_model=128, d_ff=512, dropout=0.5)
        ffn.train()

        x = torch.randn(2, 8, 128)
        output1 = ffn(x)
        output2 = ffn(x)

        # With dropout, outputs should differ
        assert not torch.allclose(output1, output2)

    def test_batch_independence(self):
        """Test that batches are processed independently."""
        ffn = PositionwiseFeedForward(d_model=128, d_ff=512, dropout=0.0)
        ffn.eval()

        x = torch.randn(2, 8, 128)

        output_batched = ffn(x)
        output0 = ffn(x[0:1])
        output1 = ffn(x[1:2])

        assert torch.allclose(output_batched[0:1], output0, atol=1e-6)
        assert torch.allclose(output_batched[1:2], output1, atol=1e-6)

    def test_different_d_model_and_d_ff(self):
        """Test various d_model and d_ff combinations."""
        configs = [
            (128, 512),
            (256, 1024),
            (512, 2048),
            (768, 3072),
        ]

        for d_model, d_ff in configs:
            ffn = PositionwiseFeedForward(d_model=d_model, d_ff=d_ff)
            x = torch.randn(1, 4, d_model)
            output = ffn(x)

            assert output.shape == (1, 4, d_model)
            assert not torch.isnan(output).any()

    def test_extra_repr(self):
        """Test the extra_repr method."""
        ffn = PositionwiseFeedForward(d_model=512, d_ff=2048)
        repr_str = ffn.extra_repr()

        assert "d_model=512" in repr_str
        assert "d_ff=2048" in repr_str

    def test_no_nan_output(self):
        """Test that output contains no NaN values."""
        ffn = PositionwiseFeedForward(d_model=256, d_ff=1024)

        # Test with various input values
        test_inputs = [
            torch.randn(2, 8, 256),
            torch.zeros(2, 8, 256),
            torch.ones(2, 8, 256),
            torch.randn(2, 8, 256) * 100,  # Large values
            torch.randn(2, 8, 256) * 0.001,  # Small values
        ]

        for x in test_inputs:
            output = ffn(x)
            assert not torch.isnan(output).any()
            assert not torch.isinf(output).any()

    def test_weights_are_trainable(self):
        """Test that weights can be updated during training."""
        ffn = PositionwiseFeedForward(d_model=64, d_ff=128)
        optimizer = torch.optim.SGD(ffn.parameters(), lr=0.1)

        # Store original weights
        original_w1 = ffn.linear1.weight.clone()
        original_w2 = ffn.linear2.weight.clone()

        # Forward and backward pass
        x = torch.randn(2, 4, 64)
        output = ffn(x)
        loss = output.sum()
        loss.backward()
        optimizer.step()

        # Weights should have changed
        assert not torch.allclose(ffn.linear1.weight, original_w1)
        assert not torch.allclose(ffn.linear2.weight, original_w2)


class TestPositionwiseFeedForwardIntegration:
    """Integration tests for PositionwiseFeedForward."""

    def test_with_layer_norm(self):
        """Test FFN with layer normalization (as used in transformer)."""
        d_model = 256
        ffn = PositionwiseFeedForward(d_model=d_model, d_ff=1024, dropout=0.0)
        layer_norm = nn.LayerNorm(d_model)

        x = torch.randn(2, 8, d_model)

        # Simulate a transformer sub-layer with residual connection
        ffn_output = ffn(x)
        output = layer_norm(x + ffn_output)

        assert output.shape == x.shape
        assert not torch.isnan(output).any()

    def test_in_transformer_layer_pattern(self):
        """Test FFN in a pattern similar to transformer encoder layer."""
        from src.attention import MultiHeadAttention

        d_model = 256
        n_heads = 4

        mha = MultiHeadAttention(d_model=d_model, n_heads=n_heads, dropout=0.0)
        ffn = PositionwiseFeedForward(d_model=d_model, d_ff=1024, dropout=0.0)
        norm1 = nn.LayerNorm(d_model)
        norm2 = nn.LayerNorm(d_model)

        x = torch.randn(2, 8, d_model)

        # Self-attention sub-layer
        attn_out, _ = mha(x, x, x)
        x = norm1(x + attn_out)

        # Feed-forward sub-layer
        ffn_out = ffn(x)
        x = norm2(x + ffn_out)

        assert x.shape == (2, 8, d_model)
        assert not torch.isnan(x).any()

    def test_memory_efficiency(self):
        """Test that FFN doesn't use excessive memory for large sequences."""
        ffn = PositionwiseFeedForward(d_model=512, d_ff=2048)

        # Large batch and sequence
        x = torch.randn(4, 256, 512)
        output = ffn(x)

        assert output.shape == x.shape
        assert not torch.isnan(output).any()
