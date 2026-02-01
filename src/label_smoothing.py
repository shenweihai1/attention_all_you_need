"""
Label smoothing loss for Transformer training.

This module implements label smoothing as described in
"Attention Is All You Need" (Vaswani et al., 2017) and
"Rethinking the Inception Architecture for Computer Vision" (Szegedy et al., 2016).

Label smoothing is a regularization technique that softens the target distribution:
- True class: 1 - epsilon + epsilon/K
- Other classes: epsilon/K

where K is the number of classes and epsilon is the smoothing parameter.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing loss with optional padding token handling.

    This implements the KL divergence loss with smoothed labels as described
    in the Transformer paper. It's mathematically equivalent to:
    (1 - epsilon) * cross_entropy + epsilon * uniform_loss

    Args:
        smoothing: Label smoothing factor epsilon (default: 0.1)
        padding_idx: Token index to ignore in loss calculation (default: None)
        reduction: Reduction method - 'mean', 'sum', or 'none' (default: 'mean')

    Example:
        >>> criterion = LabelSmoothingLoss(smoothing=0.1, padding_idx=0)
        >>> logits = model(src, tgt)  # (batch, seq_len, vocab_size)
        >>> target = tgt[:, 1:]  # Shifted targets
        >>> loss = criterion(logits.view(-1, vocab_size), target.view(-1))
    """

    def __init__(
        self,
        smoothing: float = 0.1,
        padding_idx: Optional[int] = None,
        reduction: str = "mean",
    ):
        super().__init__()

        if not 0.0 <= smoothing < 1.0:
            raise ValueError(f"smoothing must be in [0, 1), got {smoothing}")
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"reduction must be 'mean', 'sum', or 'none', got {reduction}")

        self.smoothing = smoothing
        self.padding_idx = padding_idx
        self.reduction = reduction
        self.confidence = 1.0 - smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute label smoothing loss.

        Args:
            logits: Model predictions of shape (batch_size * seq_len, vocab_size)
                    or (batch_size, vocab_size)
            target: Target indices of shape (batch_size * seq_len,) or (batch_size,)

        Returns:
            Loss value (scalar if reduction is 'mean' or 'sum', otherwise same shape as target)
        """
        vocab_size = logits.size(-1)

        # Create smoothed target distribution
        # Start with uniform distribution: epsilon / vocab_size
        smooth_target = torch.full_like(logits, self.smoothing / vocab_size)

        # Set the true class probability: 1 - epsilon + epsilon/vocab_size = confidence + epsilon/vocab_size
        smooth_target.scatter_(
            dim=-1,
            index=target.unsqueeze(-1),
            value=self.confidence + self.smoothing / vocab_size
        )

        # Handle padding tokens by zeroing out their contribution
        if self.padding_idx is not None:
            padding_mask = target.eq(self.padding_idx)
            smooth_target[padding_mask] = 0.0

        # Compute log probabilities
        log_probs = F.log_softmax(logits, dim=-1)

        # KL divergence: sum over vocab dimension
        # KL(p||q) = sum(p * log(p/q)) = sum(p * log(p)) - sum(p * log(q))
        # Since we want loss = -sum(p * log(q)), we compute:
        loss = -(smooth_target * log_probs).sum(dim=-1)

        # Apply reduction
        if self.reduction == "none":
            return loss
        elif self.reduction == "sum":
            return loss.sum()
        else:  # mean
            # For mean, we exclude padding tokens from the count
            if self.padding_idx is not None:
                non_padding = target.ne(self.padding_idx)
                return loss.sum() / non_padding.sum().clamp(min=1)
            else:
                return loss.mean()


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Alternative implementation using cross-entropy formulation.

    This computes:
    loss = (1 - epsilon) * CE(logits, target) + epsilon * mean(-log_softmax(logits))

    This is mathematically equivalent to LabelSmoothingLoss but uses a different
    computational approach that may be more numerically stable in some cases.

    Args:
        smoothing: Label smoothing factor epsilon (default: 0.1)
        padding_idx: Token index to ignore in loss calculation (default: None)
        reduction: Reduction method - 'mean', 'sum', or 'none' (default: 'mean')
    """

    def __init__(
        self,
        smoothing: float = 0.1,
        padding_idx: Optional[int] = None,
        reduction: str = "mean",
    ):
        super().__init__()

        if not 0.0 <= smoothing < 1.0:
            raise ValueError(f"smoothing must be in [0, 1), got {smoothing}")
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"reduction must be 'mean', 'sum', or 'none', got {reduction}")

        self.smoothing = smoothing
        self.padding_idx = padding_idx
        self.reduction = reduction
        self.confidence = 1.0 - smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute label smoothing cross-entropy loss.

        Args:
            logits: Model predictions of shape (batch_size * seq_len, vocab_size)
                    or (batch_size, vocab_size)
            target: Target indices of shape (batch_size * seq_len,) or (batch_size,)

        Returns:
            Loss value
        """
        log_probs = F.log_softmax(logits, dim=-1)
        vocab_size = logits.size(-1)

        # Standard cross-entropy term: -log(p[target])
        nll_loss = -log_probs.gather(dim=-1, index=target.unsqueeze(-1)).squeeze(-1)

        # Smoothing term: mean negative log probability across all classes
        smooth_loss = -log_probs.mean(dim=-1)

        # Combined loss
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss

        # Handle padding
        if self.padding_idx is not None:
            padding_mask = target.eq(self.padding_idx)
            loss = loss.masked_fill(padding_mask, 0.0)

        # Apply reduction
        if self.reduction == "none":
            return loss
        elif self.reduction == "sum":
            return loss.sum()
        else:  # mean
            if self.padding_idx is not None:
                non_padding = target.ne(self.padding_idx)
                return loss.sum() / non_padding.sum().clamp(min=1)
            else:
                return loss.mean()


def label_smoothing_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    smoothing: float = 0.1,
    padding_idx: Optional[int] = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Functional interface for label smoothing loss.

    This is a convenience function that creates a LabelSmoothingLoss
    and applies it in one call.

    Args:
        logits: Model predictions of shape (batch_size * seq_len, vocab_size)
        target: Target indices of shape (batch_size * seq_len,)
        smoothing: Label smoothing factor epsilon (default: 0.1)
        padding_idx: Token index to ignore (default: None)
        reduction: Reduction method (default: 'mean')

    Returns:
        Loss value

    Example:
        >>> loss = label_smoothing_loss(logits, target, smoothing=0.1, padding_idx=0)
    """
    log_probs = F.log_softmax(logits, dim=-1)
    vocab_size = logits.size(-1)
    confidence = 1.0 - smoothing

    # Standard cross-entropy term
    nll_loss = -log_probs.gather(dim=-1, index=target.unsqueeze(-1)).squeeze(-1)

    # Smoothing term
    smooth_loss = -log_probs.mean(dim=-1)

    # Combined loss
    loss = confidence * nll_loss + smoothing * smooth_loss

    # Handle padding
    if padding_idx is not None:
        padding_mask = target.eq(padding_idx)
        loss = loss.masked_fill(padding_mask, 0.0)

    # Apply reduction
    if reduction == "none":
        return loss
    elif reduction == "sum":
        return loss.sum()
    else:  # mean
        if padding_idx is not None:
            non_padding = target.ne(padding_idx)
            return loss.sum() / non_padding.sum().clamp(min=1)
        else:
            return loss.mean()
