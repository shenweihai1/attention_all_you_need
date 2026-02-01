"""
Tests for label smoothing loss.

Tests the label smoothing implementation as described in
"Attention Is All You Need" (Vaswani et al., 2017).
"""

import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.label_smoothing import (
    LabelSmoothingLoss,
    LabelSmoothingCrossEntropy,
    label_smoothing_loss,
)


class TestLabelSmoothingLoss:
    """Tests for LabelSmoothingLoss class."""

    def test_creation_default_params(self):
        """Test that loss can be created with default parameters."""
        criterion = LabelSmoothingLoss()
        assert criterion.smoothing == 0.1
        assert criterion.padding_idx is None
        assert criterion.reduction == "mean"
        assert criterion.confidence == 0.9

    def test_creation_custom_params(self):
        """Test loss creation with custom parameters."""
        criterion = LabelSmoothingLoss(
            smoothing=0.2,
            padding_idx=0,
            reduction="sum",
        )
        assert criterion.smoothing == 0.2
        assert criterion.padding_idx == 0
        assert criterion.reduction == "sum"
        assert criterion.confidence == 0.8

    def test_invalid_smoothing_negative(self):
        """Test that negative smoothing raises error."""
        with pytest.raises(ValueError, match="smoothing must be in"):
            LabelSmoothingLoss(smoothing=-0.1)

    def test_invalid_smoothing_too_large(self):
        """Test that smoothing >= 1 raises error."""
        with pytest.raises(ValueError, match="smoothing must be in"):
            LabelSmoothingLoss(smoothing=1.0)

    def test_invalid_reduction(self):
        """Test that invalid reduction raises error."""
        with pytest.raises(ValueError, match="reduction must be"):
            LabelSmoothingLoss(reduction="invalid")

    def test_output_shape_mean_reduction(self):
        """Test output shape with mean reduction."""
        criterion = LabelSmoothingLoss(smoothing=0.1, reduction="mean")
        logits = torch.randn(10, 100)  # batch_size=10, vocab_size=100
        target = torch.randint(0, 100, (10,))
        loss = criterion(logits, target)
        assert loss.shape == ()  # Scalar

    def test_output_shape_sum_reduction(self):
        """Test output shape with sum reduction."""
        criterion = LabelSmoothingLoss(smoothing=0.1, reduction="sum")
        logits = torch.randn(10, 100)
        target = torch.randint(0, 100, (10,))
        loss = criterion(logits, target)
        assert loss.shape == ()  # Scalar

    def test_output_shape_no_reduction(self):
        """Test output shape with no reduction."""
        criterion = LabelSmoothingLoss(smoothing=0.1, reduction="none")
        logits = torch.randn(10, 100)
        target = torch.randint(0, 100, (10,))
        loss = criterion(logits, target)
        assert loss.shape == (10,)  # Same as target

    def test_zero_smoothing_equals_cross_entropy(self):
        """Test that zero smoothing is equivalent to cross-entropy."""
        logits = torch.randn(32, 100)
        target = torch.randint(0, 100, (32,))

        criterion_smooth = LabelSmoothingLoss(smoothing=0.0, reduction="mean")
        criterion_ce = nn.CrossEntropyLoss(reduction="mean")

        loss_smooth = criterion_smooth(logits, target)
        loss_ce = criterion_ce(logits, target)

        assert torch.allclose(loss_smooth, loss_ce, atol=1e-5)

    def test_smoothing_increases_loss(self):
        """Test that label smoothing generally increases loss on correct predictions."""
        # Create logits that strongly predict the correct class
        logits = torch.zeros(10, 100)
        target = torch.arange(10) % 100
        for i in range(10):
            logits[i, target[i]] = 10.0  # High logit for correct class

        criterion_no_smooth = LabelSmoothingLoss(smoothing=0.0)
        criterion_smooth = LabelSmoothingLoss(smoothing=0.1)

        loss_no_smooth = criterion_no_smooth(logits, target)
        loss_smooth = criterion_smooth(logits, target)

        # With smoothing, some probability mass goes to other classes,
        # so loss should be higher
        assert loss_smooth > loss_no_smooth

    def test_padding_tokens_ignored(self):
        """Test that padding tokens are ignored in loss calculation."""
        criterion = LabelSmoothingLoss(smoothing=0.1, padding_idx=0, reduction="sum")

        logits = torch.randn(10, 100)
        # Half of targets are padding (0)
        target = torch.tensor([0, 0, 0, 0, 0, 1, 2, 3, 4, 5])

        loss_with_pad = criterion(logits, target)

        # Loss should only count non-padding tokens
        criterion_no_pad = LabelSmoothingLoss(smoothing=0.1, reduction="sum")
        loss_non_pad = criterion_no_pad(logits[5:], target[5:])

        assert torch.allclose(loss_with_pad, loss_non_pad, atol=1e-5)

    def test_mean_reduction_excludes_padding(self):
        """Test that mean reduction correctly excludes padding from count."""
        criterion = LabelSmoothingLoss(smoothing=0.1, padding_idx=0, reduction="mean")

        logits = torch.randn(10, 100)
        target = torch.tensor([0, 0, 0, 0, 0, 1, 2, 3, 4, 5])

        loss = criterion(logits, target)

        # Manual calculation: sum of non-padding losses / 5
        criterion_sum = LabelSmoothingLoss(smoothing=0.1, padding_idx=0, reduction="sum")
        loss_sum = criterion_sum(logits, target)
        expected_mean = loss_sum / 5

        assert torch.allclose(loss, expected_mean, atol=1e-5)

    def test_gradient_flow(self):
        """Test that gradients flow correctly through the loss."""
        logits = torch.randn(10, 100, requires_grad=True)
        target = torch.randint(0, 100, (10,))

        criterion = LabelSmoothingLoss(smoothing=0.1)
        loss = criterion(logits, target)

        loss.backward()
        assert logits.grad is not None
        assert not torch.all(logits.grad == 0)

    def test_batched_sequence_input(self):
        """Test with flattened sequence input (batch * seq_len, vocab_size)."""
        batch_size = 4
        seq_len = 20
        vocab_size = 1000

        logits = torch.randn(batch_size * seq_len, vocab_size)
        target = torch.randint(0, vocab_size, (batch_size * seq_len,))

        criterion = LabelSmoothingLoss(smoothing=0.1)
        loss = criterion(logits, target)

        assert loss.shape == ()
        assert not torch.isnan(loss)

    def test_deterministic_output(self):
        """Test that output is deterministic for same input."""
        torch.manual_seed(42)
        logits = torch.randn(10, 100)
        target = torch.randint(0, 100, (10,))

        criterion = LabelSmoothingLoss(smoothing=0.1)
        loss1 = criterion(logits, target)
        loss2 = criterion(logits, target)

        assert torch.equal(loss1, loss2)


class TestLabelSmoothingCrossEntropy:
    """Tests for LabelSmoothingCrossEntropy class."""

    def test_creation(self):
        """Test that loss can be created."""
        criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
        assert criterion.smoothing == 0.1
        assert criterion.confidence == 0.9

    def test_output_shape(self):
        """Test output shape."""
        criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
        logits = torch.randn(10, 100)
        target = torch.randint(0, 100, (10,))
        loss = criterion(logits, target)
        assert loss.shape == ()

    def test_zero_smoothing_equals_cross_entropy(self):
        """Test that zero smoothing is equivalent to cross-entropy."""
        logits = torch.randn(32, 100)
        target = torch.randint(0, 100, (32,))

        criterion_smooth = LabelSmoothingCrossEntropy(smoothing=0.0)
        criterion_ce = nn.CrossEntropyLoss()

        loss_smooth = criterion_smooth(logits, target)
        loss_ce = criterion_ce(logits, target)

        assert torch.allclose(loss_smooth, loss_ce, atol=1e-5)

    def test_padding_handled(self):
        """Test that padding is handled correctly."""
        criterion = LabelSmoothingCrossEntropy(smoothing=0.1, padding_idx=0)

        logits = torch.randn(10, 100)
        target = torch.tensor([0, 0, 0, 0, 0, 1, 2, 3, 4, 5])

        loss = criterion(logits, target)
        assert not torch.isnan(loss)

    def test_equivalent_to_label_smoothing_loss(self):
        """Test that both implementations give similar results."""
        torch.manual_seed(123)
        logits = torch.randn(32, 100)
        target = torch.randint(0, 100, (32,))

        criterion1 = LabelSmoothingLoss(smoothing=0.1)
        criterion2 = LabelSmoothingCrossEntropy(smoothing=0.1)

        loss1 = criterion1(logits, target)
        loss2 = criterion2(logits, target)

        # They should be close but not exactly equal due to different computation
        assert torch.allclose(loss1, loss2, atol=1e-4)


class TestLabelSmoothingLossFunction:
    """Tests for label_smoothing_loss function."""

    def test_basic_call(self):
        """Test basic function call."""
        logits = torch.randn(10, 100)
        target = torch.randint(0, 100, (10,))

        loss = label_smoothing_loss(logits, target, smoothing=0.1)
        assert loss.shape == ()
        assert not torch.isnan(loss)

    def test_with_padding(self):
        """Test function with padding."""
        logits = torch.randn(10, 100)
        target = torch.tensor([0, 0, 0, 0, 0, 1, 2, 3, 4, 5])

        loss = label_smoothing_loss(logits, target, smoothing=0.1, padding_idx=0)
        assert not torch.isnan(loss)

    def test_reduction_none(self):
        """Test function with no reduction."""
        logits = torch.randn(10, 100)
        target = torch.randint(0, 100, (10,))

        loss = label_smoothing_loss(logits, target, smoothing=0.1, reduction="none")
        assert loss.shape == (10,)

    def test_equivalent_to_class(self):
        """Test that function gives same result as class."""
        torch.manual_seed(42)
        logits = torch.randn(32, 100)
        target = torch.randint(0, 100, (32,))

        criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
        loss_class = criterion(logits, target)
        loss_func = label_smoothing_loss(logits, target, smoothing=0.1)

        assert torch.allclose(loss_class, loss_func, atol=1e-6)


class TestLabelSmoothingMathematics:
    """Tests for mathematical properties of label smoothing."""

    def test_smoothed_distribution_sums_to_one(self):
        """Test that smoothed target distribution sums to 1."""
        vocab_size = 100
        smoothing = 0.1

        # Create a simple test case
        logits = torch.randn(1, vocab_size)
        target = torch.tensor([42])

        # Manually create smoothed distribution
        smooth_target = torch.full((1, vocab_size), smoothing / vocab_size)
        confidence = 1.0 - smoothing
        smooth_target[0, target[0]] = confidence + smoothing / vocab_size

        # Should sum to 1
        assert torch.allclose(smooth_target.sum(), torch.tensor(1.0), atol=1e-6)

    def test_correct_class_probability(self):
        """Test that correct class gets right probability."""
        vocab_size = 100
        smoothing = 0.1
        confidence = 1.0 - smoothing

        # Probability for correct class: (1 - epsilon) + epsilon/K
        expected_prob = confidence + smoothing / vocab_size

        # Create smoothed distribution
        smooth_target = torch.full((vocab_size,), smoothing / vocab_size)
        target_idx = 42
        smooth_target[target_idx] = confidence + smoothing / vocab_size

        assert torch.allclose(smooth_target[target_idx], torch.tensor(expected_prob), atol=1e-6)

    def test_incorrect_class_probability(self):
        """Test that incorrect classes get right probability."""
        vocab_size = 100
        smoothing = 0.1

        # Probability for incorrect class: epsilon/K
        expected_prob = smoothing / vocab_size

        # Create smoothed distribution
        smooth_target = torch.full((vocab_size,), smoothing / vocab_size)
        target_idx = 42
        smooth_target[target_idx] = (1.0 - smoothing) + smoothing / vocab_size

        # Check any incorrect class
        incorrect_idx = 0 if target_idx != 0 else 1
        assert torch.allclose(smooth_target[incorrect_idx], torch.tensor(expected_prob), atol=1e-6)

    def test_loss_bounded(self):
        """Test that loss is bounded and reasonable."""
        logits = torch.randn(100, 1000)
        target = torch.randint(0, 1000, (100,))

        criterion = LabelSmoothingLoss(smoothing=0.1)
        loss = criterion(logits, target)

        # Loss should be positive
        assert loss > 0

        # Loss should be finite
        assert not torch.isinf(loss)


class TestLabelSmoothingIntegration:
    """Integration tests with Transformer-like scenarios."""

    def test_with_transformer_output_shape(self):
        """Test with Transformer-like output shapes."""
        batch_size = 4
        seq_len = 32
        vocab_size = 10000

        # Simulate Transformer output
        logits = torch.randn(batch_size, seq_len, vocab_size)
        target = torch.randint(0, vocab_size, (batch_size, seq_len))

        # Flatten for loss calculation (common pattern)
        logits_flat = logits.view(-1, vocab_size)
        target_flat = target.view(-1)

        criterion = LabelSmoothingLoss(smoothing=0.1, padding_idx=0)
        loss = criterion(logits_flat, target_flat)

        assert loss.shape == ()
        assert not torch.isnan(loss)

    def test_training_step_simulation(self):
        """Simulate a training step with label smoothing."""
        # Simple model
        model = nn.Linear(512, 1000)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = LabelSmoothingLoss(smoothing=0.1, padding_idx=0)

        # Simulate input
        x = torch.randn(32, 512)
        target = torch.randint(1, 1000, (32,))  # Avoid padding token 0
        target[0] = 0  # Add one padding token

        # Forward pass
        logits = model(x)
        loss = criterion(logits, target)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Should complete without error
        assert not torch.isnan(loss)

    def test_different_smoothing_values(self):
        """Test with different smoothing values from the literature."""
        logits = torch.randn(32, 100)
        target = torch.randint(0, 100, (32,))

        # Common smoothing values
        for smoothing in [0.0, 0.05, 0.1, 0.15, 0.2]:
            criterion = LabelSmoothingLoss(smoothing=smoothing)
            loss = criterion(logits, target)
            assert not torch.isnan(loss), f"NaN loss with smoothing={smoothing}"
            assert loss >= 0, f"Negative loss with smoothing={smoothing}"


class TestLabelSmoothingEdgeCases:
    """Tests for edge cases."""

    def test_single_sample(self):
        """Test with single sample."""
        logits = torch.randn(1, 100)
        target = torch.tensor([42])

        criterion = LabelSmoothingLoss(smoothing=0.1)
        loss = criterion(logits, target)

        assert loss.shape == ()
        assert not torch.isnan(loss)

    def test_two_class_problem(self):
        """Test with binary classification."""
        logits = torch.randn(10, 2)
        target = torch.randint(0, 2, (10,))

        criterion = LabelSmoothingLoss(smoothing=0.1)
        loss = criterion(logits, target)

        assert not torch.isnan(loss)

    def test_large_vocab(self):
        """Test with large vocabulary."""
        logits = torch.randn(10, 50000)
        target = torch.randint(0, 50000, (10,))

        criterion = LabelSmoothingLoss(smoothing=0.1)
        loss = criterion(logits, target)

        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_all_padding(self):
        """Test when all tokens are padding."""
        logits = torch.randn(10, 100)
        target = torch.zeros(10, dtype=torch.long)  # All padding

        criterion = LabelSmoothingLoss(smoothing=0.1, padding_idx=0, reduction="mean")
        loss = criterion(logits, target)

        # Loss should be 0 when all are padding (or handle gracefully)
        assert not torch.isnan(loss)

    def test_very_small_smoothing(self):
        """Test with very small smoothing value."""
        logits = torch.randn(32, 100)
        target = torch.randint(0, 100, (32,))

        criterion = LabelSmoothingLoss(smoothing=1e-6)
        loss = criterion(logits, target)

        # Should be very close to cross-entropy
        ce_criterion = nn.CrossEntropyLoss()
        ce_loss = ce_criterion(logits, target)

        assert torch.allclose(loss, ce_loss, atol=1e-4)

    def test_high_smoothing(self):
        """Test with high (but valid) smoothing value."""
        logits = torch.randn(32, 100)
        target = torch.randint(0, 100, (32,))

        criterion = LabelSmoothingLoss(smoothing=0.5)
        loss = criterion(logits, target)

        assert not torch.isnan(loss)
        assert loss >= 0


class TestLabelSmoothingGPU:
    """Tests for GPU compatibility (if available)."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_computation(self):
        """Test that label smoothing works on GPU."""
        device = torch.device("cuda")

        logits = torch.randn(32, 100, device=device)
        target = torch.randint(0, 100, (32,), device=device)

        criterion = LabelSmoothingLoss(smoothing=0.1)
        loss = criterion(logits, target)

        assert loss.device.type == "cuda"
        assert not torch.isnan(loss)
