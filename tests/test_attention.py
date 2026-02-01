"""
Tests for the Scaled Dot-Product Attention module.
"""

import math
import pytest
import torch
import torch.nn as nn

from src.attention import (
    scaled_dot_product_attention,
    ScaledDotProductAttention,
    create_causal_mask,
    create_padding_mask,
)


class TestScaledDotProductAttentionFunction:
    """Tests for the scaled_dot_product_attention function."""

    def test_output_shape(self):
        """Test that output has correct shape."""
        batch, heads, seq_len, d_k = 2, 8, 10, 64
        q = torch.randn(batch, heads, seq_len, d_k)
        k = torch.randn(batch, heads, seq_len, d_k)
        v = torch.randn(batch, heads, seq_len, d_k)

        output, attn_weights = scaled_dot_product_attention(q, k, v)

        assert output.shape == (batch, heads, seq_len, d_k)
        assert attn_weights.shape == (batch, heads, seq_len, seq_len)

    def test_different_seq_lengths(self):
        """Test with different query and key/value sequence lengths."""
        batch, heads, d_k = 2, 8, 64
        seq_len_q, seq_len_kv = 5, 10

        q = torch.randn(batch, heads, seq_len_q, d_k)
        k = torch.randn(batch, heads, seq_len_kv, d_k)
        v = torch.randn(batch, heads, seq_len_kv, d_k)

        output, attn_weights = scaled_dot_product_attention(q, k, v)

        assert output.shape == (batch, heads, seq_len_q, d_k)
        assert attn_weights.shape == (batch, heads, seq_len_q, seq_len_kv)

    def test_attention_weights_sum_to_one(self):
        """Test that attention weights sum to 1 along the key dimension."""
        q = torch.randn(2, 4, 8, 32)
        k = torch.randn(2, 4, 8, 32)
        v = torch.randn(2, 4, 8, 32)

        _, attn_weights = scaled_dot_product_attention(q, k, v)

        # Sum along last dimension (key positions)
        weight_sums = attn_weights.sum(dim=-1)
        assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-6)

    def test_attention_weights_non_negative(self):
        """Test that attention weights are non-negative."""
        q = torch.randn(2, 4, 8, 32)
        k = torch.randn(2, 4, 8, 32)
        v = torch.randn(2, 4, 8, 32)

        _, attn_weights = scaled_dot_product_attention(q, k, v)

        assert (attn_weights >= 0).all()

    def test_scaling_factor(self):
        """Test that scaling by sqrt(d_k) is applied."""
        torch.manual_seed(42)
        d_k = 64
        q = torch.randn(1, 1, 1, d_k)
        k = torch.randn(1, 1, 1, d_k)
        v = torch.randn(1, 1, 1, d_k)

        # The raw dot product divided by sqrt(d_k)
        expected_score = (q @ k.transpose(-2, -1)) / math.sqrt(d_k)
        expected_attn = torch.softmax(expected_score, dim=-1)
        expected_output = expected_attn @ v

        output, _ = scaled_dot_product_attention(q, k, v)

        assert torch.allclose(output, expected_output, atol=1e-6)

    def test_mask_zeroes_attention(self):
        """Test that masked positions receive zero attention weight."""
        q = torch.randn(1, 1, 3, 4)
        k = torch.randn(1, 1, 3, 4)
        v = torch.randn(1, 1, 3, 4)

        # Mask the last position
        mask = torch.tensor([[[[False, False, True]]]])  # (1, 1, 1, 3)

        _, attn_weights = scaled_dot_product_attention(q, k, v, mask=mask)

        # Last column should be zero (masked)
        assert torch.allclose(attn_weights[..., -1], torch.zeros(1, 1, 3), atol=1e-6)

    def test_causal_mask_application(self):
        """Test causal masking prevents attending to future positions."""
        seq_len = 4
        q = torch.randn(1, 1, seq_len, 8)
        k = torch.randn(1, 1, seq_len, 8)
        v = torch.randn(1, 1, seq_len, 8)

        mask = create_causal_mask(seq_len)

        _, attn_weights = scaled_dot_product_attention(q, k, v, mask=mask)

        # Check upper triangle (excluding diagonal) is zero
        for i in range(seq_len):
            for j in range(i + 1, seq_len):
                assert attn_weights[0, 0, i, j].item() < 1e-6

    def test_dropout_applied(self):
        """Test that dropout is applied when provided."""
        torch.manual_seed(42)
        q = torch.randn(2, 4, 8, 32)
        k = torch.randn(2, 4, 8, 32)
        v = torch.randn(2, 4, 8, 32)

        dropout = nn.Dropout(p=0.5)

        # With dropout in training mode, some weights should be zeroed
        output1, attn1 = scaled_dot_product_attention(q, k, v, dropout=dropout)

        # Run again - should get different results due to dropout
        output2, attn2 = scaled_dot_product_attention(q, k, v, dropout=dropout)

        # Outputs should differ due to random dropout
        assert not torch.allclose(output1, output2)

    def test_no_dropout_when_none(self):
        """Test deterministic output when no dropout."""
        q = torch.randn(2, 4, 8, 32)
        k = torch.randn(2, 4, 8, 32)
        v = torch.randn(2, 4, 8, 32)

        output1, _ = scaled_dot_product_attention(q, k, v, dropout=None)
        output2, _ = scaled_dot_product_attention(q, k, v, dropout=None)

        assert torch.allclose(output1, output2)

    def test_batch_independence(self):
        """Test that batches are processed independently."""
        q = torch.randn(2, 1, 4, 8)
        k = torch.randn(2, 1, 4, 8)
        v = torch.randn(2, 1, 4, 8)

        output_batched, _ = scaled_dot_product_attention(q, k, v)

        # Process each batch separately
        output0, _ = scaled_dot_product_attention(
            q[0:1], k[0:1], v[0:1]
        )
        output1, _ = scaled_dot_product_attention(
            q[1:2], k[1:2], v[1:2]
        )

        assert torch.allclose(output_batched[0:1], output0)
        assert torch.allclose(output_batched[1:2], output1)

    def test_gradient_flow(self):
        """Test that gradients flow through attention."""
        q = torch.randn(2, 4, 8, 32, requires_grad=True)
        k = torch.randn(2, 4, 8, 32, requires_grad=True)
        v = torch.randn(2, 4, 8, 32, requires_grad=True)

        output, _ = scaled_dot_product_attention(q, k, v)
        loss = output.sum()
        loss.backward()

        assert q.grad is not None
        assert k.grad is not None
        assert v.grad is not None
        assert not torch.isnan(q.grad).any()
        assert not torch.isnan(k.grad).any()
        assert not torch.isnan(v.grad).any()


class TestScaledDotProductAttentionModule:
    """Tests for the ScaledDotProductAttention nn.Module."""

    def test_initialization(self):
        """Test module initialization."""
        attn = ScaledDotProductAttention(dropout=0.1)
        assert attn.dropout is not None
        assert attn.dropout.p == 0.1

    def test_no_dropout_initialization(self):
        """Test module without dropout."""
        attn = ScaledDotProductAttention(dropout=0.0)
        assert attn.dropout is None

    def test_forward_pass(self):
        """Test forward pass produces correct shapes."""
        attn = ScaledDotProductAttention(dropout=0.1)

        q = torch.randn(2, 8, 10, 64)
        k = torch.randn(2, 8, 10, 64)
        v = torch.randn(2, 8, 10, 64)

        attn.eval()  # Disable dropout for deterministic test
        output, weights = attn(q, k, v)

        assert output.shape == (2, 8, 10, 64)
        assert weights.shape == (2, 8, 10, 10)

    def test_with_mask(self):
        """Test forward pass with mask."""
        attn = ScaledDotProductAttention()

        q = torch.randn(2, 8, 10, 64)
        k = torch.randn(2, 8, 10, 64)
        v = torch.randn(2, 8, 10, 64)
        mask = create_causal_mask(10)

        output, weights = attn(q, k, v, mask=mask)

        assert output.shape == (2, 8, 10, 64)
        # Check causal structure in weights
        for i in range(10):
            for j in range(i + 1, 10):
                assert (weights[:, :, i, j] < 1e-6).all()

    def test_eval_mode(self):
        """Test that eval mode disables dropout."""
        attn = ScaledDotProductAttention(dropout=0.5)

        q = torch.randn(2, 8, 10, 64)
        k = torch.randn(2, 8, 10, 64)
        v = torch.randn(2, 8, 10, 64)

        attn.eval()
        output1, _ = attn(q, k, v)
        output2, _ = attn(q, k, v)

        # In eval mode, outputs should be identical
        assert torch.allclose(output1, output2)

    def test_train_mode_with_dropout(self):
        """Test that training mode applies dropout."""
        torch.manual_seed(42)
        attn = ScaledDotProductAttention(dropout=0.5)

        q = torch.randn(2, 8, 10, 64)
        k = torch.randn(2, 8, 10, 64)
        v = torch.randn(2, 8, 10, 64)

        attn.train()
        output1, _ = attn(q, k, v)
        output2, _ = attn(q, k, v)

        # In train mode with dropout, outputs should differ
        assert not torch.allclose(output1, output2)


class TestCreateCausalMask:
    """Tests for create_causal_mask function."""

    def test_mask_shape(self):
        """Test that mask has correct shape."""
        mask = create_causal_mask(5)
        assert mask.shape == (5, 5)

    def test_mask_is_boolean(self):
        """Test that mask is boolean type."""
        mask = create_causal_mask(5)
        assert mask.dtype == torch.bool

    def test_diagonal_is_false(self):
        """Test that diagonal elements are False (can attend to self)."""
        mask = create_causal_mask(5)
        for i in range(5):
            assert mask[i, i].item() == False

    def test_lower_triangle_is_false(self):
        """Test that lower triangle is False (can attend to past)."""
        mask = create_causal_mask(5)
        for i in range(5):
            for j in range(i):
                assert mask[i, j].item() == False

    def test_upper_triangle_is_true(self):
        """Test that upper triangle is True (cannot attend to future)."""
        mask = create_causal_mask(5)
        for i in range(5):
            for j in range(i + 1, 5):
                assert mask[i, j].item() == True

    def test_specific_mask_values(self):
        """Test specific mask for seq_len=4."""
        mask = create_causal_mask(4)
        expected = torch.tensor([
            [False, True, True, True],
            [False, False, True, True],
            [False, False, False, True],
            [False, False, False, False],
        ])
        assert torch.equal(mask, expected)

    def test_device_placement(self):
        """Test mask can be created on specified device."""
        mask = create_causal_mask(4, device=torch.device("cpu"))
        assert mask.device.type == "cpu"

    def test_seq_len_one(self):
        """Test mask with sequence length 1."""
        mask = create_causal_mask(1)
        expected = torch.tensor([[False]])
        assert torch.equal(mask, expected)


class TestCreatePaddingMask:
    """Tests for create_padding_mask function."""

    def test_mask_shape(self):
        """Test that mask has correct shape for broadcasting."""
        seq = torch.tensor([[1, 2, 3, 0, 0], [1, 2, 0, 0, 0]])
        mask = create_padding_mask(seq, pad_idx=0)
        assert mask.shape == (2, 1, 1, 5)

    def test_mask_is_boolean(self):
        """Test that mask is boolean type."""
        seq = torch.tensor([[1, 2, 0]])
        mask = create_padding_mask(seq, pad_idx=0)
        assert mask.dtype == torch.bool

    def test_padding_positions_are_true(self):
        """Test that padding positions are True."""
        seq = torch.tensor([[1, 2, 3, 0, 0]])
        mask = create_padding_mask(seq, pad_idx=0)
        expected = torch.tensor([[[[False, False, False, True, True]]]])
        assert torch.equal(mask, expected)

    def test_no_padding(self):
        """Test sequence with no padding."""
        seq = torch.tensor([[1, 2, 3, 4, 5]])
        mask = create_padding_mask(seq, pad_idx=0)
        assert not mask.any()

    def test_all_padding(self):
        """Test sequence with all padding."""
        seq = torch.tensor([[0, 0, 0]])
        mask = create_padding_mask(seq, pad_idx=0)
        assert mask.all()

    def test_different_pad_idx(self):
        """Test with different padding index."""
        seq = torch.tensor([[1, 2, 99, 99]])
        mask = create_padding_mask(seq, pad_idx=99)
        expected = torch.tensor([[[[False, False, True, True]]]])
        assert torch.equal(mask, expected)

    def test_batch_independence(self):
        """Test that each batch is masked independently."""
        seq = torch.tensor([
            [1, 2, 0, 0],
            [1, 0, 0, 0],
        ])
        mask = create_padding_mask(seq, pad_idx=0)

        expected = torch.tensor([
            [[[False, False, True, True]]],
            [[[False, True, True, True]]],
        ])
        assert torch.equal(mask, expected)


class TestAttentionIntegration:
    """Integration tests for attention components."""

    def test_attention_with_causal_mask(self):
        """Test attention with causal mask produces valid output."""
        attn = ScaledDotProductAttention(dropout=0.0)
        seq_len = 8

        q = torch.randn(2, 4, seq_len, 32)
        k = torch.randn(2, 4, seq_len, 32)
        v = torch.randn(2, 4, seq_len, 32)
        mask = create_causal_mask(seq_len)

        output, weights = attn(q, k, v, mask=mask)

        # Output should be valid
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

        # Weights should sum to 1 for non-fully-masked rows
        weight_sums = weights.sum(dim=-1)
        assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-5)

    def test_attention_with_padding_mask(self):
        """Test attention with padding mask."""
        attn = ScaledDotProductAttention(dropout=0.0)

        batch, heads, seq_len, d = 2, 4, 6, 32
        q = torch.randn(batch, heads, seq_len, d)
        k = torch.randn(batch, heads, seq_len, d)
        v = torch.randn(batch, heads, seq_len, d)

        # Create padding mask
        seq = torch.tensor([[1, 2, 3, 0, 0, 0], [1, 2, 3, 4, 5, 0]])
        pad_mask = create_padding_mask(seq, pad_idx=0)

        output, weights = attn(q, k, v, mask=pad_mask)

        # Check padding positions have zero attention
        # First batch: positions 3, 4, 5 are padding
        assert (weights[0, :, :, 3:] < 1e-6).all()
        # Second batch: position 5 is padding
        assert (weights[1, :, :, 5] < 1e-6).all()

    def test_combined_causal_and_padding_mask(self):
        """Test combining causal and padding masks."""
        seq_len = 5
        batch = 2

        # Create both masks
        causal_mask = create_causal_mask(seq_len)  # (seq_len, seq_len)
        seq = torch.tensor([[1, 2, 3, 0, 0], [1, 2, 3, 4, 0]])
        pad_mask = create_padding_mask(seq, pad_idx=0)  # (batch, 1, 1, seq_len)

        # Combine masks (logical OR)
        combined_mask = causal_mask | pad_mask

        assert combined_mask.shape == (batch, 1, seq_len, seq_len)

        attn = ScaledDotProductAttention(dropout=0.0)
        q = torch.randn(batch, 4, seq_len, 32)
        k = torch.randn(batch, 4, seq_len, 32)
        v = torch.randn(batch, 4, seq_len, 32)

        output, weights = attn(q, k, v, mask=combined_mask)

        # Should produce valid output
        assert not torch.isnan(output).any()
