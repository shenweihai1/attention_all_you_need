"""
Learning rate scheduler for the Transformer model.

This module implements the learning rate schedule described in
"Attention Is All You Need" (Vaswani et al., 2017):

    lrate = d_model^(-0.5) * min(step_num^(-0.5), step_num * warmup_steps^(-1.5))

This corresponds to increasing the learning rate linearly for the first
warmup_steps training steps, and decreasing it thereafter proportionally
to the inverse square root of the step number.
"""

import math
from typing import List

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class TransformerScheduler(_LRScheduler):
    """
    Learning rate scheduler as described in "Attention Is All You Need".

    The learning rate follows the formula:
        lrate = d_model^(-0.5) * min(step_num^(-0.5), step_num * warmup_steps^(-1.5))

    This increases the learning rate linearly during warmup, then decreases it
    proportionally to the inverse square root of the step number.

    Args:
        optimizer: The optimizer to schedule
        d_model: Model dimension (default: 512)
        warmup_steps: Number of warmup steps (default: 4000)
        last_epoch: The index of the last epoch (default: -1)
        scale: Optional scaling factor for the learning rate (default: 1.0)

    Example:
        >>> optimizer = torch.optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
        >>> scheduler = TransformerScheduler(optimizer, d_model=512, warmup_steps=4000)
        >>> for step in range(10000):
        ...     optimizer.step()
        ...     scheduler.step()
    """

    def __init__(
        self,
        optimizer: Optimizer,
        d_model: int = 512,
        warmup_steps: int = 4000,
        last_epoch: int = -1,
        scale: float = 1.0,
    ):
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.scale = scale

        # Store d_model factor for efficiency
        self._d_model_factor = d_model ** -0.5

        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """
        Calculate the learning rate for the current step.

        Returns:
            List of learning rates for each parameter group
        """
        # _step_count starts at 1 after first call to step()
        # Use max(1, ...) to avoid division by zero at step 0
        step_num = max(1, self._step_count)

        # Calculate the learning rate multiplier
        # lrate = d_model^(-0.5) * min(step^(-0.5), step * warmup^(-1.5))
        lr = self._d_model_factor * min(
            step_num ** -0.5,
            step_num * (self.warmup_steps ** -1.5)
        )

        # Apply optional scaling factor
        lr *= self.scale

        return [lr for _ in self.base_lrs]

    def _get_closed_form_lr(self) -> List[float]:
        """
        Return the learning rate computed with a closed-form expression.

        This is more efficient than get_lr() when called with a specific step.
        """
        return self.get_lr()


class WarmupScheduler(_LRScheduler):
    """
    Simple warmup scheduler that linearly increases the learning rate.

    After warmup, maintains the base learning rate.

    Args:
        optimizer: The optimizer to schedule
        warmup_steps: Number of warmup steps
        last_epoch: The index of the last epoch (default: -1)

    Example:
        >>> optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        >>> scheduler = WarmupScheduler(optimizer, warmup_steps=1000)
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """
        Calculate the learning rate for the current step.

        Returns:
            List of learning rates for each parameter group
        """
        step_num = max(1, self._step_count)

        if step_num <= self.warmup_steps:
            # Linear warmup
            scale = step_num / self.warmup_steps
        else:
            # Maintain base learning rate
            scale = 1.0

        return [base_lr * scale for base_lr in self.base_lrs]


class InverseSquareRootScheduler(_LRScheduler):
    """
    Inverse square root decay scheduler with optional warmup.

    The learning rate follows:
        - During warmup: linear increase from 0 to peak_lr
        - After warmup: decay proportionally to 1/sqrt(step)

    Args:
        optimizer: The optimizer to schedule
        warmup_steps: Number of warmup steps (default: 4000)
        peak_lr: Peak learning rate reached at end of warmup (default: 1e-3)
        last_epoch: The index of the last epoch (default: -1)

    Example:
        >>> optimizer = torch.optim.Adam(model.parameters(), lr=1.0)
        >>> scheduler = InverseSquareRootScheduler(optimizer, warmup_steps=4000, peak_lr=1e-3)
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int = 4000,
        peak_lr: float = 1e-3,
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps
        self.peak_lr = peak_lr

        # Calculate decay factor so that lr at warmup_steps equals peak_lr
        # After warmup: lr = peak_lr * sqrt(warmup_steps) / sqrt(step)
        self._decay_factor = peak_lr * math.sqrt(warmup_steps)

        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """
        Calculate the learning rate for the current step.

        Returns:
            List of learning rates for each parameter group
        """
        step_num = max(1, self._step_count)

        if step_num <= self.warmup_steps:
            # Linear warmup
            lr = self.peak_lr * step_num / self.warmup_steps
        else:
            # Inverse square root decay
            lr = self._decay_factor / math.sqrt(step_num)

        return [lr for _ in self.base_lrs]


def get_transformer_scheduler(
    optimizer: Optimizer,
    d_model: int = 512,
    warmup_steps: int = 4000,
    scale: float = 1.0,
) -> TransformerScheduler:
    """
    Create a learning rate scheduler for Transformer training.

    This is a convenience function that creates the standard Transformer
    scheduler as described in the paper.

    Args:
        optimizer: The optimizer to schedule
        d_model: Model dimension (default: 512)
        warmup_steps: Number of warmup steps (default: 4000)
        scale: Optional scaling factor (default: 1.0)

    Returns:
        TransformerScheduler instance

    Example:
        >>> model = Transformer(src_vocab_size=10000, tgt_vocab_size=10000)
        >>> optimizer = torch.optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
        >>> scheduler = get_transformer_scheduler(optimizer, d_model=512, warmup_steps=4000)
    """
    return TransformerScheduler(
        optimizer,
        d_model=d_model,
        warmup_steps=warmup_steps,
        scale=scale,
    )


def get_lr_at_step(
    step: int,
    d_model: int = 512,
    warmup_steps: int = 4000,
    scale: float = 1.0,
) -> float:
    """
    Calculate the learning rate at a specific step without creating a scheduler.

    Useful for debugging and visualization.

    Args:
        step: The step number (1-indexed)
        d_model: Model dimension (default: 512)
        warmup_steps: Number of warmup steps (default: 4000)
        scale: Optional scaling factor (default: 1.0)

    Returns:
        Learning rate at the specified step

    Example:
        >>> lr_at_4000 = get_lr_at_step(4000, d_model=512, warmup_steps=4000)
        >>> print(f"Learning rate at step 4000: {lr_at_4000:.6f}")
    """
    step = max(1, step)
    d_model_factor = d_model ** -0.5

    lr = d_model_factor * min(
        step ** -0.5,
        step * (warmup_steps ** -1.5)
    )

    return lr * scale
