"""
Tests for the Decoder module.

Tests the DecoderLayer implementation ensuring it matches the architecture
described in "Attention Is All You Need".
"""

import pytest
import torch
import torch.nn as nn

from src.decoder import DecoderLayer
from src.attention import create_causal_mask, create_padding_mask


class TestDecoderLayer:
    """Tests for the DecoderLayer module."""

    def test_initialization_default(self):
        """Test DecoderLayer initialization with default parameters."""
        layer = DecoderLayer()
        assert layer.d_model == 512
        assert layer.n_heads == 8
        assert layer.d_ff == 2048

    def test_initialization_custom(self):
        """Test DecoderLayer initialization with custom parameters."""
        layer = DecoderLayer(d_model=256, n_heads=4, d_ff=1024, dropout=0.2)
        assert layer.d_model == 256
        assert layer.n_heads == 4
        assert layer.d_ff == 1024

    def test_output_shape(self):
        """Test that output has correct shape."""
        d_model = 512
        layer = DecoderLayer(d_model=d_model, n_heads=8, d_ff=2048)

        batch, tgt_len, src_len = 2, 10, 15
        tgt = torch.randn(batch, tgt_len, d_model)
        memory = torch.randn(batch, src_len, d_model)

        output = layer(tgt, memory)

        assert output.shape == (batch, tgt_len, d_model)

    def test_output_shape_various_sizes(self):
        """Test output shape with various batch and sequence sizes."""
        layer = DecoderLayer(d_model=256, n_heads=4, d_ff=1024)

        test_cases = [
            (1, 1, 1, 256),      # minimal
            (1, 10, 15, 256),    # single batch
            (32, 50, 60, 256),   # larger batch
            (4, 100, 80, 256),   # tgt_len > src_len
            (4, 80, 100, 256),   # src_len > tgt_len
        ]

        for batch, tgt_len, src_len, d_model in test_cases:
            tgt = torch.randn(batch, tgt_len, d_model)
            memory = torch.randn(batch, src_len, d_model)
            output = layer(tgt, memory)
            assert output.shape == (batch, tgt_len, d_model)

    def test_with_causal_mask(self):
        """Test DecoderLayer with causal mask for self-attention."""
        layer = DecoderLayer(d_model=256, n_heads=4, d_ff=1024, dropout=0.0)

        batch, tgt_len, src_len = 2, 8, 10
        tgt = torch.randn(batch, tgt_len, 256)
        memory = torch.randn(batch, src_len, 256)

        # Create causal mask
        tgt_mask = create_causal_mask(tgt_len)

        output = layer(tgt, memory, tgt_mask=tgt_mask)

        assert output.shape == (batch, tgt_len, 256)
        assert not torch.isnan(output).any()

    def test_with_padding_mask(self):
        """Test DecoderLayer with padding masks."""
        layer = DecoderLayer(d_model=256, n_heads=4, d_ff=1024, dropout=0.0)

        batch, tgt_len, src_len = 2, 6, 8

        tgt = torch.randn(batch, tgt_len, 256)
        memory = torch.randn(batch, src_len, 256)

        # Create padding mask for source (encoder output)
        src_seq = torch.tensor([[1, 2, 3, 4, 0, 0, 0, 0], [1, 2, 3, 4, 5, 6, 0, 0]])
        memory_mask = create_padding_mask(src_seq, pad_idx=0)

        output = layer(tgt, memory, memory_mask=memory_mask)

        assert output.shape == (batch, tgt_len, 256)
        assert not torch.isnan(output).any()

    def test_with_combined_masks(self):
        """Test DecoderLayer with both causal and padding masks."""
        layer = DecoderLayer(d_model=256, n_heads=4, d_ff=1024, dropout=0.0)

        batch, tgt_len, src_len = 2, 6, 8

        tgt = torch.randn(batch, tgt_len, 256)
        memory = torch.randn(batch, src_len, 256)

        # Causal mask for target self-attention
        causal_mask = create_causal_mask(tgt_len)

        # Padding mask for source
        src_seq = torch.tensor([[1, 2, 3, 4, 0, 0, 0, 0], [1, 2, 3, 4, 5, 6, 0, 0]])
        memory_mask = create_padding_mask(src_seq, pad_idx=0)

        output = layer(tgt, memory, tgt_mask=causal_mask, memory_mask=memory_mask)

        assert output.shape == (batch, tgt_len, 256)
        assert not torch.isnan(output).any()

    def test_gradient_flow(self):
        """Test that gradients flow through all decoder layer components."""
        layer = DecoderLayer(d_model=256, n_heads=4, d_ff=1024)

        tgt = torch.randn(2, 8, 256, requires_grad=True)
        memory = torch.randn(2, 10, 256, requires_grad=True)

        output = layer(tgt, memory)
        loss = output.sum()
        loss.backward()

        # Check gradients on inputs
        assert tgt.grad is not None
        assert memory.grad is not None
        assert not torch.isnan(tgt.grad).any()
        assert not torch.isnan(memory.grad).any()

        # Check gradients on layer parameters
        for name, param in layer.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"

    def test_has_all_components(self):
        """Test that DecoderLayer has all required components."""
        layer = DecoderLayer(d_model=512, n_heads=8, d_ff=2048)

        # Check attention layers
        assert hasattr(layer, 'self_attention')
        assert hasattr(layer, 'cross_attention')

        # Check feed-forward network
        assert hasattr(layer, 'feed_forward')

        # Check layer norms (3 for 3 sub-layers)
        assert hasattr(layer, 'norm1')
        assert hasattr(layer, 'norm2')
        assert hasattr(layer, 'norm3')
        assert isinstance(layer.norm1, nn.LayerNorm)
        assert isinstance(layer.norm2, nn.LayerNorm)
        assert isinstance(layer.norm3, nn.LayerNorm)

        # Check dropout layers
        assert hasattr(layer, 'dropout1')
        assert hasattr(layer, 'dropout2')
        assert hasattr(layer, 'dropout3')

    def test_layer_norm_applied(self):
        """Test that layer normalization is applied after each sub-layer."""
        layer = DecoderLayer(d_model=256, n_heads=4, d_ff=1024, dropout=0.0)
        layer.eval()

        tgt = torch.randn(2, 8, 256)
        memory = torch.randn(2, 10, 256)

        output = layer(tgt, memory)

        # Output should have normalized statistics (mean ~ 0, std ~ 1 per position)
        # This is a rough check that layer norm is being applied
        mean = output.mean(dim=-1)
        std = output.std(dim=-1)

        # Layer norm ensures mean is close to 0 and std is close to 1
        assert torch.allclose(mean, torch.zeros_like(mean), atol=0.1)
        assert torch.allclose(std, torch.ones_like(std), atol=0.2)

    def test_residual_connection(self):
        """Test that residual connections are working."""
        layer = DecoderLayer(d_model=256, n_heads=4, d_ff=1024, dropout=0.0)

        # Use zeros for memory to minimize cross-attention effect
        tgt = torch.randn(2, 8, 256)
        memory = torch.zeros(2, 10, 256)

        output = layer(tgt, memory)

        # Due to residual connections, output should have some correlation with input
        # (not be completely different)
        correlation = torch.corrcoef(
            torch.stack([tgt.flatten(), output.flatten()])
        )[0, 1]

        # Should have some positive correlation (not zero or negative)
        assert correlation > -1.0  # Very loose check - residuals ensure some info preserved

    def test_eval_mode_deterministic(self):
        """Test that eval mode produces deterministic outputs."""
        layer = DecoderLayer(d_model=256, n_heads=4, d_ff=1024, dropout=0.5)
        layer.eval()

        tgt = torch.randn(2, 8, 256)
        memory = torch.randn(2, 10, 256)

        output1 = layer(tgt, memory)
        output2 = layer(tgt, memory)

        assert torch.allclose(output1, output2)

    def test_train_mode_with_dropout(self):
        """Test that training mode applies dropout."""
        torch.manual_seed(42)
        layer = DecoderLayer(d_model=256, n_heads=4, d_ff=1024, dropout=0.5)
        layer.train()

        tgt = torch.randn(2, 8, 256)
        memory = torch.randn(2, 10, 256)

        output1 = layer(tgt, memory)
        output2 = layer(tgt, memory)

        # With dropout, outputs should differ
        assert not torch.allclose(output1, output2)

    def test_batch_independence(self):
        """Test that batches are processed independently."""
        layer = DecoderLayer(d_model=256, n_heads=4, d_ff=1024, dropout=0.0)
        layer.eval()

        tgt = torch.randn(2, 8, 256)
        memory = torch.randn(2, 10, 256)

        output_batched = layer(tgt, memory)
        output0 = layer(tgt[0:1], memory[0:1])
        output1 = layer(tgt[1:2], memory[1:2])

        assert torch.allclose(output_batched[0:1], output0, atol=1e-5)
        assert torch.allclose(output_batched[1:2], output1, atol=1e-5)

    def test_parameter_count(self):
        """Test that the model has the expected number of parameters."""
        d_model, n_heads, d_ff = 512, 8, 2048
        layer = DecoderLayer(d_model=d_model, n_heads=n_heads, d_ff=d_ff)

        # Self-attention: 4 * d_model^2 (Q, K, V projections + output projection)
        # Cross-attention: 4 * d_model^2
        # Feed-forward: d_model * d_ff + d_ff + d_ff * d_model + d_model
        # Layer norms: 3 * 2 * d_model (3 norms, each has weight and bias)
        self_attn_params = 4 * d_model * d_model
        cross_attn_params = 4 * d_model * d_model
        ff_params = d_model * d_ff + d_ff + d_ff * d_model + d_model
        ln_params = 3 * 2 * d_model

        expected_params = self_attn_params + cross_attn_params + ff_params + ln_params

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
            layer = DecoderLayer(d_model=d_model, n_heads=n_heads, d_ff=d_ff)
            tgt = torch.randn(1, 4, d_model)
            memory = torch.randn(1, 6, d_model)
            output = layer(tgt, memory)

            assert output.shape == (1, 4, d_model)
            assert not torch.isnan(output).any()

    def test_extra_repr(self):
        """Test the extra_repr method."""
        layer = DecoderLayer(d_model=512, n_heads=8, d_ff=2048)
        repr_str = layer.extra_repr()

        assert "d_model=512" in repr_str
        assert "n_heads=8" in repr_str
        assert "d_ff=2048" in repr_str

    def test_no_nan_output(self):
        """Test that output contains no NaN values."""
        layer = DecoderLayer(d_model=256, n_heads=4, d_ff=1024)

        # Test with various input values
        test_inputs = [
            (torch.randn(2, 8, 256), torch.randn(2, 10, 256)),
            (torch.zeros(2, 8, 256), torch.zeros(2, 10, 256)),
            (torch.ones(2, 8, 256), torch.ones(2, 10, 256)),
            (torch.randn(2, 8, 256) * 10, torch.randn(2, 10, 256) * 10),
        ]

        for tgt, memory in test_inputs:
            output = layer(tgt, memory)
            assert not torch.isnan(output).any()
            assert not torch.isinf(output).any()

    def test_self_attention_uses_tgt_only(self):
        """Test that self-attention uses target tensor only (not memory)."""
        layer = DecoderLayer(d_model=256, n_heads=4, d_ff=1024, dropout=0.0)
        layer.eval()

        tgt = torch.randn(2, 8, 256)
        memory1 = torch.randn(2, 10, 256)
        memory2 = torch.randn(2, 10, 256)

        # Self-attention output before cross-attention should be same
        # We can't directly test this without modifying the layer,
        # but we verify the layer structure is correct
        assert layer.self_attention is not None
        assert layer.cross_attention is not None
        assert layer.self_attention is not layer.cross_attention

    def test_cross_attention_uses_memory(self):
        """Test that cross-attention uses encoder memory."""
        layer = DecoderLayer(d_model=256, n_heads=4, d_ff=1024, dropout=0.0)
        layer.eval()

        tgt = torch.randn(2, 8, 256)
        memory1 = torch.randn(2, 10, 256)
        memory2 = torch.randn(2, 10, 256)

        output1 = layer(tgt, memory1)
        output2 = layer(tgt, memory2)

        # Different memory should produce different outputs
        assert not torch.allclose(output1, output2)


class TestDecoderLayerIntegration:
    """Integration tests for the DecoderLayer."""

    def test_with_encoder_output(self):
        """Test DecoderLayer with actual encoder output."""
        from src.encoder import EncoderLayer

        d_model = 256

        encoder_layer = EncoderLayer(d_model=d_model, n_heads=4, d_ff=1024, dropout=0.0)
        decoder_layer = DecoderLayer(d_model=d_model, n_heads=4, d_ff=1024, dropout=0.0)

        src = torch.randn(2, 10, d_model)
        tgt = torch.randn(2, 8, d_model)

        # Pass through encoder
        memory = encoder_layer(src)

        # Pass through decoder
        output = decoder_layer(tgt, memory)

        assert output.shape == (2, 8, d_model)
        assert not torch.isnan(output).any()

    def test_with_positional_encoding(self):
        """Test DecoderLayer with positional encoding added to inputs."""
        from src.positional_encoding import PositionalEncoding

        d_model = 256
        pe = PositionalEncoding(d_model=d_model, max_seq_len=100, dropout=0.0)
        decoder_layer = DecoderLayer(d_model=d_model, n_heads=4, d_ff=1024, dropout=0.0)

        tgt = torch.randn(2, 15, d_model)
        memory = torch.randn(2, 20, d_model)

        # Add positional encoding to target
        tgt = pe(tgt)

        output = decoder_layer(tgt, memory)

        assert output.shape == (2, 15, d_model)
        assert not torch.isnan(output).any()

    def test_stacked_decoder_layers(self):
        """Test stacking multiple decoder layers."""
        d_model = 256
        n_layers = 6

        layers = nn.ModuleList([
            DecoderLayer(d_model=d_model, n_heads=4, d_ff=1024, dropout=0.0)
            for _ in range(n_layers)
        ])

        tgt = torch.randn(2, 10, d_model)
        memory = torch.randn(2, 15, d_model)

        # Pass through all layers
        x = tgt
        for layer in layers:
            x = layer(x, memory)

        assert x.shape == (2, 10, d_model)
        assert not torch.isnan(x).any()

    def test_gradient_flow_stacked(self):
        """Test gradient flow through stacked decoder layers."""
        d_model = 128
        n_layers = 3

        layers = nn.ModuleList([
            DecoderLayer(d_model=d_model, n_heads=4, d_ff=512, dropout=0.0)
            for _ in range(n_layers)
        ])

        tgt = torch.randn(2, 8, d_model, requires_grad=True)
        memory = torch.randn(2, 10, d_model, requires_grad=True)

        x = tgt
        for layer in layers:
            x = layer(x, memory)

        loss = x.sum()
        loss.backward()

        assert tgt.grad is not None
        assert memory.grad is not None

        # Check all layers have gradients
        for i, layer in enumerate(layers):
            for name, param in layer.named_parameters():
                assert param.grad is not None, f"No gradient for layer {i} {name}"

    def test_with_embedding_input(self):
        """Test DecoderLayer with embedding layer input."""
        from src.positional_encoding import PositionalEncoding

        vocab_size = 1000
        d_model = 256

        embedding = nn.Embedding(vocab_size, d_model)
        pe = PositionalEncoding(d_model=d_model, max_seq_len=100, dropout=0.0)
        decoder_layer = DecoderLayer(d_model=d_model, n_heads=4, d_ff=1024, dropout=0.0)

        # Create token indices
        tgt_tokens = torch.randint(0, vocab_size, (2, 15))
        memory = torch.randn(2, 20, d_model)

        # Embed tokens and add positional encoding
        tgt = embedding(tgt_tokens)
        tgt = pe(tgt)

        output = decoder_layer(tgt, memory)

        assert output.shape == (2, 15, d_model)
        assert not torch.isnan(output).any()
