"""
Tests for weight initialization utilities.

Tests the initialization functions ensuring they properly initialize
Transformer weights as described in "Attention Is All You Need".
"""

import math

import pytest
import torch
import torch.nn as nn

from src.init import (
    init_transformer_weights,
    init_bert_weights,
    count_parameters,
    get_parameter_stats,
)
from src.transformer import Transformer


class TestInitTransformerWeights:
    """Tests for init_transformer_weights function."""

    def test_linear_xavier_initialization(self):
        """Test that Linear layers are initialized with Xavier uniform."""
        linear = nn.Linear(512, 256)

        # Store original weights for comparison
        original_weight = linear.weight.clone()

        # Apply initialization
        init_transformer_weights(linear, d_model=512)

        # Weights should have changed
        assert not torch.allclose(linear.weight, original_weight)

        # Check weight statistics (Xavier uniform should be in reasonable range)
        # Xavier uniform: U(-sqrt(6/(fan_in+fan_out)), sqrt(6/(fan_in+fan_out)))
        fan_in, fan_out = 512, 256
        bound = math.sqrt(6.0 / (fan_in + fan_out))

        assert linear.weight.min() >= -bound - 0.01
        assert linear.weight.max() <= bound + 0.01

    def test_linear_bias_zeros(self):
        """Test that Linear bias is initialized to zeros."""
        linear = nn.Linear(512, 256)
        init_transformer_weights(linear, d_model=512)

        assert torch.allclose(linear.bias, torch.zeros_like(linear.bias))

    def test_linear_no_bias(self):
        """Test initialization works for Linear without bias."""
        linear = nn.Linear(512, 256, bias=False)
        init_transformer_weights(linear, d_model=512)

        # Should not raise error
        assert linear.bias is None

    def test_embedding_normal_initialization(self):
        """Test that Embedding layers are initialized with normal distribution."""
        d_model = 512
        embedding = nn.Embedding(1000, d_model)

        init_transformer_weights(embedding, d_model=d_model)

        # Check mean is close to 0
        assert abs(embedding.weight.mean().item()) < 0.1

        # Check std is close to d_model^(-0.5)
        expected_std = d_model ** -0.5
        actual_std = embedding.weight.std().item()
        assert abs(actual_std - expected_std) < 0.01

    def test_embedding_padding_idx_zero(self):
        """Test that padding index embedding is zero."""
        d_model = 512
        padding_idx = 0
        embedding = nn.Embedding(1000, d_model, padding_idx=padding_idx)

        init_transformer_weights(embedding, d_model=d_model)

        # Padding embedding should be zero
        assert torch.allclose(
            embedding.weight[padding_idx],
            torch.zeros(d_model)
        )

    def test_layer_norm_initialization(self):
        """Test that LayerNorm is initialized with weight=1, bias=0."""
        ln = nn.LayerNorm(512)
        init_transformer_weights(ln, d_model=512)

        assert torch.allclose(ln.weight, torch.ones_like(ln.weight))
        assert torch.allclose(ln.bias, torch.zeros_like(ln.bias))

    def test_different_d_model_values(self):
        """Test initialization with various d_model values."""
        for d_model in [64, 128, 256, 512, 1024]:
            embedding = nn.Embedding(100, d_model)
            init_transformer_weights(embedding, d_model=d_model)

            expected_std = d_model ** -0.5
            actual_std = embedding.weight.std().item()
            assert abs(actual_std - expected_std) < 0.02


class TestInitBertWeights:
    """Tests for init_bert_weights function."""

    def test_linear_normal_initialization(self):
        """Test that Linear layers are initialized with normal distribution."""
        linear = nn.Linear(512, 256)
        init_bert_weights(linear, std=0.02)

        # Check std is close to 0.02
        actual_std = linear.weight.std().item()
        assert abs(actual_std - 0.02) < 0.01

    def test_linear_bias_zeros(self):
        """Test that Linear bias is initialized to zeros."""
        linear = nn.Linear(512, 256)
        init_bert_weights(linear, std=0.02)

        assert torch.allclose(linear.bias, torch.zeros_like(linear.bias))

    def test_embedding_normal_initialization(self):
        """Test that Embedding is initialized with normal distribution."""
        embedding = nn.Embedding(1000, 512)
        init_bert_weights(embedding, std=0.02)

        actual_std = embedding.weight.std().item()
        assert abs(actual_std - 0.02) < 0.01

    def test_custom_std(self):
        """Test initialization with custom std values."""
        for std in [0.01, 0.02, 0.05, 0.1]:
            linear = nn.Linear(512, 256)
            init_bert_weights(linear, std=std)

            actual_std = linear.weight.std().item()
            assert abs(actual_std - std) < 0.01


class TestTransformerInitialization:
    """Tests for Transformer weight initialization."""

    def test_auto_initialization(self):
        """Test that Transformer auto-initializes weights on creation."""
        model = Transformer(
            src_vocab_size=100,
            tgt_vocab_size=100,
            d_model=64,
            n_heads=4,
            n_encoder_layers=2,
            n_decoder_layers=2,
            d_ff=128,
        )

        # Check that output projection bias is zero (from init)
        assert torch.allclose(
            model.output_projection.bias,
            torch.zeros_like(model.output_projection.bias)
        )

        # Check embedding std
        expected_std = 64 ** -0.5
        actual_std = model.src_embedding.embedding.weight.std().item()
        assert abs(actual_std - expected_std) < 0.02

    def test_reinitialize_xavier(self):
        """Test reinitializing with Xavier method."""
        model = Transformer(
            src_vocab_size=100,
            tgt_vocab_size=100,
            d_model=64,
            n_heads=4,
            n_encoder_layers=2,
            n_decoder_layers=2,
            d_ff=128,
        )

        # Modify weights
        nn.init.ones_(model.output_projection.weight)

        # Reinitialize
        model.init_weights(method="xavier")

        # Weights should no longer be all ones
        assert not torch.allclose(
            model.output_projection.weight,
            torch.ones_like(model.output_projection.weight)
        )

    def test_reinitialize_bert(self):
        """Test reinitializing with BERT method."""
        model = Transformer(
            src_vocab_size=100,
            tgt_vocab_size=100,
            d_model=64,
            n_heads=4,
            n_encoder_layers=2,
            n_decoder_layers=2,
            d_ff=128,
        )

        # Reinitialize with BERT method
        model.init_weights(method="bert")

        # Check std is close to 0.02 for linear layers
        actual_std = model.output_projection.weight.std().item()
        assert abs(actual_std - 0.02) < 0.01

    def test_invalid_init_method(self):
        """Test that invalid initialization method raises error."""
        model = Transformer(
            src_vocab_size=100,
            tgt_vocab_size=100,
            d_model=64,
            n_heads=4,
            n_encoder_layers=2,
            n_decoder_layers=2,
            d_ff=128,
        )

        with pytest.raises(ValueError, match="Unknown initialization method"):
            model.init_weights(method="invalid")

    def test_model_still_works_after_init(self):
        """Test that model produces valid output after initialization."""
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

        assert output.shape == (2, 8, 100)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_gradient_flow_after_init(self):
        """Test that gradients flow properly after initialization."""
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

        # Check all parameters have gradients
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"


class TestCountParameters:
    """Tests for count_parameters function."""

    def test_count_all_parameters(self):
        """Test counting all parameters."""
        linear = nn.Linear(512, 256)
        expected = 512 * 256 + 256  # weight + bias
        assert count_parameters(linear, trainable_only=False) == expected

    def test_count_trainable_only(self):
        """Test counting only trainable parameters."""
        linear = nn.Linear(512, 256)
        linear.weight.requires_grad = False

        trainable = count_parameters(linear, trainable_only=True)
        total = count_parameters(linear, trainable_only=False)

        assert trainable == 256  # only bias
        assert total == 512 * 256 + 256

    def test_count_transformer_parameters(self):
        """Test counting parameters in Transformer model."""
        model = Transformer(
            src_vocab_size=100,
            tgt_vocab_size=100,
            d_model=64,
            n_heads=4,
            n_encoder_layers=2,
            n_decoder_layers=2,
            d_ff=128,
        )

        total_params = count_parameters(model)
        assert total_params > 0


class TestGetParameterStats:
    """Tests for get_parameter_stats function."""

    def test_stats_structure(self):
        """Test that stats have correct structure."""
        linear = nn.Linear(512, 256)
        stats = get_parameter_stats(linear)

        assert "total_params" in stats
        assert "trainable_params" in stats
        assert "non_trainable_params" in stats
        assert "layer_stats" in stats

    def test_stats_values(self):
        """Test that stats have correct values."""
        linear = nn.Linear(512, 256)
        stats = get_parameter_stats(linear)

        expected_total = 512 * 256 + 256
        assert stats["total_params"] == expected_total
        assert stats["trainable_params"] == expected_total
        assert stats["non_trainable_params"] == 0

    def test_stats_with_frozen_params(self):
        """Test stats with some frozen parameters."""
        linear = nn.Linear(512, 256)
        linear.weight.requires_grad = False

        stats = get_parameter_stats(linear)

        assert stats["trainable_params"] == 256  # only bias
        assert stats["non_trainable_params"] == 512 * 256  # weight

    def test_transformer_stats(self):
        """Test stats for Transformer model."""
        model = Transformer(
            src_vocab_size=100,
            tgt_vocab_size=100,
            d_model=64,
            n_heads=4,
            n_encoder_layers=2,
            n_decoder_layers=2,
            d_ff=128,
        )

        stats = get_parameter_stats(model)

        # Should have multiple layer types
        assert len(stats["layer_stats"]) > 0
        assert stats["total_params"] > 0


class TestInitializationReproducibility:
    """Tests for reproducibility of initialization."""

    def test_same_seed_same_weights(self):
        """Test that same seed produces same weights."""
        torch.manual_seed(42)
        model1 = Transformer(
            src_vocab_size=100,
            tgt_vocab_size=100,
            d_model=64,
            n_heads=4,
            n_encoder_layers=2,
            n_decoder_layers=2,
            d_ff=128,
        )

        torch.manual_seed(42)
        model2 = Transformer(
            src_vocab_size=100,
            tgt_vocab_size=100,
            d_model=64,
            n_heads=4,
            n_encoder_layers=2,
            n_decoder_layers=2,
            d_ff=128,
        )

        # Weights should be identical
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            assert torch.allclose(p1, p2)

    def test_different_seed_different_weights(self):
        """Test that different seeds produce different weights."""
        torch.manual_seed(42)
        model1 = Transformer(
            src_vocab_size=100,
            tgt_vocab_size=100,
            d_model=64,
            n_heads=4,
            n_encoder_layers=2,
            n_decoder_layers=2,
            d_ff=128,
        )

        torch.manual_seed(123)
        model2 = Transformer(
            src_vocab_size=100,
            tgt_vocab_size=100,
            d_model=64,
            n_heads=4,
            n_encoder_layers=2,
            n_decoder_layers=2,
            d_ff=128,
        )

        # At least some weights should be different
        different = False
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            if not torch.allclose(p1, p2):
                different = True
                break
        assert different
