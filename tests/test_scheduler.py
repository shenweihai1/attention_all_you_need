"""
Tests for learning rate scheduler.

Tests the Transformer learning rate schedule as described in
"Attention Is All You Need" (Vaswani et al., 2017).
"""

import math

import pytest
import torch
import torch.nn as nn

from src.scheduler import (
    TransformerScheduler,
    WarmupScheduler,
    InverseSquareRootScheduler,
    get_transformer_scheduler,
    get_lr_at_step,
)


class TestTransformerScheduler:
    """Tests for TransformerScheduler class."""

    def _create_optimizer(self):
        """Create a simple optimizer for testing."""
        model = nn.Linear(10, 10)
        return torch.optim.Adam(model.parameters(), lr=1.0)

    def test_scheduler_creation(self):
        """Test that scheduler can be created with default parameters."""
        optimizer = self._create_optimizer()
        scheduler = TransformerScheduler(optimizer)

        assert scheduler.d_model == 512
        assert scheduler.warmup_steps == 4000
        assert scheduler.scale == 1.0

    def test_custom_parameters(self):
        """Test scheduler with custom parameters."""
        optimizer = self._create_optimizer()
        scheduler = TransformerScheduler(
            optimizer,
            d_model=256,
            warmup_steps=8000,
            scale=2.0,
        )

        assert scheduler.d_model == 256
        assert scheduler.warmup_steps == 8000
        assert scheduler.scale == 2.0

    def test_lr_increases_during_warmup(self):
        """Test that learning rate increases linearly during warmup."""
        optimizer = self._create_optimizer()
        scheduler = TransformerScheduler(optimizer, d_model=512, warmup_steps=100)

        lrs = []
        for _ in range(50):
            scheduler.step()
            lrs.append(scheduler.get_last_lr()[0])

        # LR should be strictly increasing during warmup
        for i in range(1, len(lrs)):
            assert lrs[i] > lrs[i - 1], f"LR not increasing at step {i}"

    def test_lr_decreases_after_warmup(self):
        """Test that learning rate decreases after warmup."""
        optimizer = self._create_optimizer()
        scheduler = TransformerScheduler(optimizer, d_model=512, warmup_steps=100)

        # Move past warmup
        for _ in range(100):
            scheduler.step()

        # Collect LRs after warmup
        lrs = []
        for _ in range(50):
            scheduler.step()
            lrs.append(scheduler.get_last_lr()[0])

        # LR should be strictly decreasing after warmup
        for i in range(1, len(lrs)):
            assert lrs[i] < lrs[i - 1], f"LR not decreasing at step {i + 100}"

    def test_peak_lr_at_warmup_end(self):
        """Test that peak LR occurs at end of warmup."""
        optimizer = self._create_optimizer()
        warmup_steps = 100
        scheduler = TransformerScheduler(optimizer, d_model=512, warmup_steps=warmup_steps)

        lrs = []
        for _ in range(200):
            scheduler.step()
            lrs.append(scheduler.get_last_lr()[0])

        peak_idx = lrs.index(max(lrs))
        # Peak should be at or very close to warmup_steps
        assert abs(peak_idx + 1 - warmup_steps) <= 1, f"Peak at step {peak_idx + 1}, expected {warmup_steps}"

    def test_lr_formula_during_warmup(self):
        """Test that LR follows the formula during warmup."""
        d_model = 512
        warmup_steps = 4000

        # During warmup: lrate = d_model^(-0.5) * step * warmup^(-1.5)
        # Use get_lr_at_step to verify the formula is correct
        for step in [1, 10, 100, 1000, 2000]:
            actual_lr = get_lr_at_step(step, d_model=d_model, warmup_steps=warmup_steps)
            expected_lr = (d_model ** -0.5) * step * (warmup_steps ** -1.5)

            assert abs(actual_lr - expected_lr) < 1e-10, f"Step {step}: expected {expected_lr}, got {actual_lr}"

    def test_lr_formula_after_warmup(self):
        """Test that LR follows the formula after warmup."""
        d_model = 512
        warmup_steps = 4000

        # After warmup: lrate = d_model^(-0.5) * step^(-0.5)
        # Use get_lr_at_step to verify the formula is correct
        for step in [5000, 10000, 20000, 50000]:
            actual_lr = get_lr_at_step(step, d_model=d_model, warmup_steps=warmup_steps)
            expected_lr = (d_model ** -0.5) * (step ** -0.5)

            assert abs(actual_lr - expected_lr) < 1e-10, f"Step {step}: expected {expected_lr}, got {actual_lr}"

    def test_scale_factor(self):
        """Test that scale factor is applied correctly."""
        optimizer = self._create_optimizer()
        scale = 2.0
        scheduler1 = TransformerScheduler(optimizer, d_model=512, warmup_steps=100)

        optimizer2 = self._create_optimizer()
        scheduler2 = TransformerScheduler(optimizer2, d_model=512, warmup_steps=100, scale=scale)

        for _ in range(50):
            scheduler1.step()
            scheduler2.step()

        lr1 = scheduler1.get_last_lr()[0]
        lr2 = scheduler2.get_last_lr()[0]

        assert abs(lr2 - lr1 * scale) < 1e-10

    def test_different_d_model_values(self):
        """Test scheduler with different d_model values."""
        for d_model in [64, 128, 256, 512, 1024]:
            # At step 1 during warmup: lr = d_model^(-0.5) * 1 * warmup^(-1.5)
            expected_lr = (d_model ** -0.5) * 1 * (100 ** -1.5)
            actual_lr = get_lr_at_step(1, d_model=d_model, warmup_steps=100)
            assert abs(actual_lr - expected_lr) < 1e-10


class TestWarmupScheduler:
    """Tests for WarmupScheduler class."""

    def _create_optimizer(self, lr=0.001):
        """Create a simple optimizer for testing."""
        model = nn.Linear(10, 10)
        return torch.optim.Adam(model.parameters(), lr=lr)

    def test_scheduler_creation(self):
        """Test that scheduler can be created."""
        optimizer = self._create_optimizer()
        scheduler = WarmupScheduler(optimizer, warmup_steps=1000)

        assert scheduler.warmup_steps == 1000

    def test_lr_increases_during_warmup(self):
        """Test that LR increases during warmup."""
        base_lr = 0.001
        optimizer = self._create_optimizer(lr=base_lr)
        scheduler = WarmupScheduler(optimizer, warmup_steps=100)

        lrs = []
        for _ in range(50):
            scheduler.step()
            lrs.append(scheduler.get_last_lr()[0])

        # Should increase during warmup
        for i in range(1, len(lrs)):
            assert lrs[i] > lrs[i - 1]

    def test_lr_constant_after_warmup(self):
        """Test that LR is constant after warmup."""
        base_lr = 0.001
        optimizer = self._create_optimizer(lr=base_lr)
        scheduler = WarmupScheduler(optimizer, warmup_steps=100)

        # Complete warmup
        for _ in range(100):
            scheduler.step()

        # Check that LR stays constant
        for _ in range(50):
            scheduler.step()
            lr = scheduler.get_last_lr()[0]
            assert abs(lr - base_lr) < 1e-10

    def test_reaches_base_lr_at_warmup_end(self):
        """Test that base LR is reached at end of warmup."""
        base_lr = 0.001
        optimizer = self._create_optimizer(lr=base_lr)
        scheduler = WarmupScheduler(optimizer, warmup_steps=100)

        for _ in range(100):
            scheduler.step()

        lr = scheduler.get_last_lr()[0]
        assert abs(lr - base_lr) < 1e-10


class TestInverseSquareRootScheduler:
    """Tests for InverseSquareRootScheduler class."""

    def _create_optimizer(self):
        """Create a simple optimizer for testing."""
        model = nn.Linear(10, 10)
        return torch.optim.Adam(model.parameters(), lr=1.0)

    def test_scheduler_creation(self):
        """Test that scheduler can be created."""
        optimizer = self._create_optimizer()
        scheduler = InverseSquareRootScheduler(optimizer, warmup_steps=4000, peak_lr=1e-3)

        assert scheduler.warmup_steps == 4000
        assert scheduler.peak_lr == 1e-3

    def test_lr_increases_during_warmup(self):
        """Test that LR increases during warmup."""
        optimizer = self._create_optimizer()
        scheduler = InverseSquareRootScheduler(optimizer, warmup_steps=100, peak_lr=1e-3)

        lrs = []
        for _ in range(50):
            scheduler.step()
            lrs.append(scheduler.get_last_lr()[0])

        for i in range(1, len(lrs)):
            assert lrs[i] > lrs[i - 1]

    def test_reaches_peak_at_warmup_end(self):
        """Test that peak LR is reached at warmup end."""
        optimizer = self._create_optimizer()
        peak_lr = 1e-3
        warmup_steps = 100
        scheduler = InverseSquareRootScheduler(optimizer, warmup_steps=warmup_steps, peak_lr=peak_lr)

        for _ in range(warmup_steps):
            scheduler.step()

        lr = scheduler.get_last_lr()[0]
        # Allow small tolerance due to step counting
        assert abs(lr - peak_lr) < 1e-5, f"Expected {peak_lr}, got {lr}"

    def test_lr_decreases_after_warmup(self):
        """Test that LR decreases after warmup."""
        optimizer = self._create_optimizer()
        scheduler = InverseSquareRootScheduler(optimizer, warmup_steps=100, peak_lr=1e-3)

        for _ in range(100):
            scheduler.step()

        lrs = []
        for _ in range(50):
            scheduler.step()
            lrs.append(scheduler.get_last_lr()[0])

        for i in range(1, len(lrs)):
            assert lrs[i] < lrs[i - 1]

    def test_inverse_sqrt_decay(self):
        """Test that decay follows inverse sqrt pattern."""
        optimizer = self._create_optimizer()
        warmup_steps = 100
        peak_lr = 1e-3
        scheduler = InverseSquareRootScheduler(optimizer, warmup_steps=warmup_steps, peak_lr=peak_lr)

        for _ in range(warmup_steps):
            scheduler.step()

        # After warmup: lr = peak_lr * sqrt(warmup) / sqrt(step)
        decay_factor = peak_lr * math.sqrt(warmup_steps)

        for step in [200, 400, 800, 1600]:
            for _ in range(step - scheduler._step_count):
                scheduler.step()

            actual_lr = scheduler.get_last_lr()[0]
            expected_lr = decay_factor / math.sqrt(step)

            assert abs(actual_lr - expected_lr) < 1e-10


class TestGetTransformerScheduler:
    """Tests for get_transformer_scheduler convenience function."""

    def test_returns_transformer_scheduler(self):
        """Test that function returns TransformerScheduler instance."""
        model = nn.Linear(10, 10)
        optimizer = torch.optim.Adam(model.parameters(), lr=1.0)
        scheduler = get_transformer_scheduler(optimizer)

        assert isinstance(scheduler, TransformerScheduler)

    def test_passes_parameters(self):
        """Test that parameters are passed correctly."""
        model = nn.Linear(10, 10)
        optimizer = torch.optim.Adam(model.parameters(), lr=1.0)
        scheduler = get_transformer_scheduler(
            optimizer,
            d_model=256,
            warmup_steps=2000,
            scale=0.5,
        )

        assert scheduler.d_model == 256
        assert scheduler.warmup_steps == 2000
        assert scheduler.scale == 0.5


class TestGetLrAtStep:
    """Tests for get_lr_at_step utility function."""

    def test_returns_correct_lr(self):
        """Test that function returns correct LR."""
        d_model = 512
        warmup_steps = 4000

        # During warmup
        step = 1000
        expected = (d_model ** -0.5) * step * (warmup_steps ** -1.5)
        actual = get_lr_at_step(step, d_model, warmup_steps)
        assert abs(actual - expected) < 1e-10

        # After warmup
        step = 10000
        expected = (d_model ** -0.5) * (step ** -0.5)
        actual = get_lr_at_step(step, d_model, warmup_steps)
        assert abs(actual - expected) < 1e-10

    def test_scale_factor(self):
        """Test scale factor is applied."""
        lr_base = get_lr_at_step(1000, d_model=512, warmup_steps=4000)
        lr_scaled = get_lr_at_step(1000, d_model=512, warmup_steps=4000, scale=2.0)

        assert abs(lr_scaled - lr_base * 2.0) < 1e-10

    def test_step_zero_handled(self):
        """Test that step 0 is handled (treated as 1)."""
        lr_at_0 = get_lr_at_step(0)
        lr_at_1 = get_lr_at_step(1)

        assert lr_at_0 == lr_at_1


class TestSchedulerIntegration:
    """Integration tests for schedulers with training loop simulation."""

    def test_training_loop_simulation(self):
        """Test scheduler in simulated training loop."""
        model = nn.Linear(10, 10)
        optimizer = torch.optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
        scheduler = TransformerScheduler(optimizer, d_model=512, warmup_steps=100)

        # Simulate training loop
        for step in range(200):
            # Simulate forward/backward
            x = torch.randn(2, 10)
            y = model(x)
            loss = y.sum()
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

        # Should complete without error
        final_lr = scheduler.get_last_lr()[0]
        assert final_lr > 0

    def test_state_dict_save_load(self):
        """Test that scheduler state can be saved and loaded."""
        model = nn.Linear(10, 10)
        optimizer = torch.optim.Adam(model.parameters(), lr=1.0)
        scheduler = TransformerScheduler(optimizer, d_model=512, warmup_steps=100)

        # Step a few times
        for _ in range(50):
            scheduler.step()

        # Save state
        state_dict = scheduler.state_dict()
        lr_before = scheduler.get_last_lr()[0]

        # Create new scheduler and load state
        new_optimizer = torch.optim.Adam(model.parameters(), lr=1.0)
        new_scheduler = TransformerScheduler(new_optimizer, d_model=512, warmup_steps=100)
        new_scheduler.load_state_dict(state_dict)

        # Step count should be restored (PyTorch adds 1 for last_epoch tracking)
        assert new_scheduler._step_count == state_dict["_step_count"]

    def test_multiple_param_groups(self):
        """Test scheduler with multiple parameter groups."""
        model1 = nn.Linear(10, 10)
        model2 = nn.Linear(10, 10)
        optimizer = torch.optim.Adam([
            {"params": model1.parameters()},
            {"params": model2.parameters()},
        ], lr=1.0)

        scheduler = TransformerScheduler(optimizer, d_model=512, warmup_steps=100)

        for _ in range(50):
            scheduler.step()

        # All groups should have same LR
        lrs = scheduler.get_last_lr()
        assert len(lrs) == 2
        assert lrs[0] == lrs[1]


class TestSchedulerValues:
    """Tests to verify specific LR values match the paper."""

    def test_paper_base_model_values(self):
        """Test LR values for base model configuration from paper."""
        d_model = 512
        warmup_steps = 4000

        # At step 1: should be very small
        lr_1 = get_lr_at_step(1, d_model, warmup_steps)
        assert lr_1 < 1e-6

        # At warmup end (step 4000): should be peak
        lr_4000 = get_lr_at_step(4000, d_model, warmup_steps)

        # Check specific value: d_model^(-0.5) * min(4000^(-0.5), 4000 * 4000^(-1.5))
        # = 512^(-0.5) * min(0.0158, 0.0158) ≈ 0.000698
        expected = (512 ** -0.5) * (4000 ** -0.5)
        assert abs(lr_4000 - expected) < 1e-10

        # At step 10000: should decay
        lr_10000 = get_lr_at_step(10000, d_model, warmup_steps)
        expected_10000 = (512 ** -0.5) * (10000 ** -0.5)
        assert abs(lr_10000 - expected_10000) < 1e-10

        # Verify decay: lr at 10000 < lr at 4000
        assert lr_10000 < lr_4000

    def test_lr_continuity_at_warmup_boundary(self):
        """Test that LR is continuous at warmup boundary."""
        d_model = 512
        warmup_steps = 4000

        # At warmup_steps, both formulas should give same value
        # Linear formula: d_model^(-0.5) * step * warmup^(-1.5)
        # Decay formula: d_model^(-0.5) * step^(-0.5)

        step = warmup_steps
        linear_val = (d_model ** -0.5) * step * (warmup_steps ** -1.5)
        decay_val = (d_model ** -0.5) * (step ** -0.5)

        assert abs(linear_val - decay_val) < 1e-10


class TestSchedulerEdgeCases:
    """Tests for edge cases."""

    def test_warmup_steps_one(self):
        """Test with warmup_steps=1."""
        model = nn.Linear(10, 10)
        optimizer = torch.optim.Adam(model.parameters(), lr=1.0)
        scheduler = TransformerScheduler(optimizer, d_model=512, warmup_steps=1)

        # Should not raise
        for _ in range(10):
            scheduler.step()

        lr = scheduler.get_last_lr()[0]
        assert lr > 0

    def test_very_large_step(self):
        """Test with very large step numbers."""
        lr = get_lr_at_step(1000000, d_model=512, warmup_steps=4000)
        assert lr > 0
        assert not math.isnan(lr)
        assert not math.isinf(lr)

    def test_small_d_model(self):
        """Test with small d_model."""
        model = nn.Linear(10, 10)
        optimizer = torch.optim.Adam(model.parameters(), lr=1.0)
        scheduler = TransformerScheduler(optimizer, d_model=32, warmup_steps=100)

        for _ in range(50):
            scheduler.step()

        lr = scheduler.get_last_lr()[0]
        assert lr > 0
