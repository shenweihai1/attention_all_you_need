"""
Training loop for Transformer model with gradient accumulation support.

This module provides a Trainer class that handles:
- Training with gradient accumulation for simulating larger batch sizes
- Validation loop
- Metrics tracking (loss, tokens/sec, learning rate)
- Checkpoint saving/loading
- Integration with learning rate scheduler and label smoothing
"""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader


@dataclass
class TrainingMetrics:
    """Container for training metrics."""

    loss: float = 0.0
    tokens: int = 0
    samples: int = 0
    steps: int = 0
    elapsed_time: float = 0.0

    @property
    def avg_loss(self) -> float:
        """Average loss per step."""
        return self.loss / max(1, self.steps)

    @property
    def tokens_per_sec(self) -> float:
        """Tokens processed per second."""
        return self.tokens / max(1e-6, self.elapsed_time)

    def reset(self) -> None:
        """Reset all metrics."""
        self.loss = 0.0
        self.tokens = 0
        self.samples = 0
        self.steps = 0
        self.elapsed_time = 0.0

    def update(
        self,
        loss: float,
        tokens: int,
        samples: int,
        elapsed: float,
    ) -> None:
        """Update metrics with new values."""
        self.loss += loss
        self.tokens += tokens
        self.samples += samples
        self.steps += 1
        self.elapsed_time += elapsed


@dataclass
class TrainerConfig:
    """Configuration for the Trainer."""

    # Training parameters
    max_steps: int = 100000
    gradient_accumulation_steps: int = 1
    max_grad_norm: Optional[float] = 1.0

    # Validation
    eval_steps: int = 1000
    eval_samples: Optional[int] = None  # None = full validation set

    # Logging
    log_steps: int = 100

    # Checkpointing
    save_steps: int = 5000
    save_dir: Optional[str] = None
    save_total_limit: Optional[int] = 5

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Padding token (for counting non-padding tokens)
    padding_idx: int = 0


class Trainer:
    """
    Trainer for Transformer models with gradient accumulation support.

    This trainer implements the training procedure described in
    "Attention Is All You Need" paper, including:
    - Gradient accumulation for effective larger batch sizes
    - Gradient clipping
    - Learning rate scheduling
    - Checkpoint management

    Args:
        model: The Transformer model to train
        optimizer: The optimizer
        criterion: The loss function (e.g., LabelSmoothingLoss)
        config: Training configuration
        scheduler: Optional learning rate scheduler
        train_loader: Training data loader
        eval_loader: Optional validation data loader

    Example:
        >>> model = Transformer(src_vocab_size=10000, tgt_vocab_size=10000)
        >>> optimizer = torch.optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98))
        >>> criterion = LabelSmoothingLoss(smoothing=0.1, padding_idx=0)
        >>> scheduler = TransformerScheduler(optimizer, d_model=512)
        >>> config = TrainerConfig(max_steps=100000, gradient_accumulation_steps=4)
        >>> trainer = Trainer(model, optimizer, criterion, config, scheduler, train_loader)
        >>> trainer.train()
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        criterion: nn.Module,
        config: TrainerConfig,
        scheduler: Optional[_LRScheduler] = None,
        train_loader: Optional[DataLoader] = None,
        eval_loader: Optional[DataLoader] = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.config = config
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.eval_loader = eval_loader

        # Move model to device
        self.device = torch.device(config.device)
        self.model = self.model.to(self.device)

        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_eval_loss = float("inf")

        # Metrics
        self.train_metrics = TrainingMetrics()

        # Callbacks
        self._log_callback: Optional[Callable[[Dict[str, Any]], None]] = None

    def set_log_callback(
        self,
        callback: Callable[[Dict[str, Any]], None],
    ) -> None:
        """Set a callback function for logging."""
        self._log_callback = callback

    def _log(self, metrics: Dict[str, Any]) -> None:
        """Log metrics using callback or print."""
        if self._log_callback:
            self._log_callback(metrics)

    def _count_tokens(
        self,
        target: torch.Tensor,
        padding_idx: int,
    ) -> int:
        """Count non-padding tokens in target."""
        return int((target != padding_idx).sum().item())

    def _get_lr(self) -> float:
        """Get current learning rate."""
        if self.scheduler:
            return self.scheduler.get_last_lr()[0]
        return self.optimizer.param_groups[0]["lr"]

    def train_step(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[float, int]:
        """
        Perform a single training step.

        Args:
            src: Source sequences (batch_size, src_len)
            tgt: Target sequences (batch_size, tgt_len)
            src_mask: Optional source padding mask
            tgt_mask: Optional target padding mask

        Returns:
            Tuple of (loss value, number of tokens)
        """
        self.model.train()

        # Move to device
        src = src.to(self.device)
        tgt = tgt.to(self.device)
        if src_mask is not None:
            src_mask = src_mask.to(self.device)
        if tgt_mask is not None:
            tgt_mask = tgt_mask.to(self.device)

        # Teacher forcing: input is tgt[:-1], target is tgt[1:]
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        # Forward pass
        logits = self.model(src, tgt_input, src_mask, tgt_mask)

        # Compute loss
        # Flatten: (batch * seq_len, vocab_size) and (batch * seq_len,)
        vocab_size = logits.size(-1)
        loss = self.criterion(
            logits.contiguous().view(-1, vocab_size),
            tgt_output.contiguous().view(-1),
        )

        # Scale loss for gradient accumulation
        scaled_loss = loss / self.config.gradient_accumulation_steps

        # Backward pass
        scaled_loss.backward()

        # Count tokens (excluding padding)
        n_tokens = self._count_tokens(tgt_output, self.config.padding_idx)

        return loss.item(), n_tokens

    def optimizer_step(self) -> None:
        """Perform optimizer step with gradient clipping."""
        # Gradient clipping
        if self.config.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.max_grad_norm,
            )

        # Optimizer step
        self.optimizer.step()
        self.optimizer.zero_grad()

        # Scheduler step
        if self.scheduler:
            self.scheduler.step()

        self.global_step += 1

    @torch.no_grad()
    def evaluate(
        self,
        eval_loader: Optional[DataLoader] = None,
        max_samples: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Evaluate the model on validation data.

        Args:
            eval_loader: Validation data loader (uses self.eval_loader if None)
            max_samples: Maximum number of samples to evaluate

        Returns:
            Dictionary of evaluation metrics
        """
        loader = eval_loader or self.eval_loader
        if loader is None:
            return {}

        self.model.eval()

        total_loss = 0.0
        total_tokens = 0
        total_samples = 0

        for batch in loader:
            src, tgt = batch["src"], batch["tgt"]
            src_mask = batch.get("src_mask")
            tgt_mask = batch.get("tgt_mask")

            # Move to device
            src = src.to(self.device)
            tgt = tgt.to(self.device)
            if src_mask is not None:
                src_mask = src_mask.to(self.device)
            if tgt_mask is not None:
                tgt_mask = tgt_mask.to(self.device)

            # Teacher forcing
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            # Forward pass
            logits = self.model(src, tgt_input, src_mask, tgt_mask)

            # Compute loss
            vocab_size = logits.size(-1)
            loss = self.criterion(
                logits.contiguous().view(-1, vocab_size),
                tgt_output.contiguous().view(-1),
            )

            n_tokens = self._count_tokens(tgt_output, self.config.padding_idx)

            total_loss += loss.item() * n_tokens
            total_tokens += n_tokens
            total_samples += src.size(0)

            if max_samples and total_samples >= max_samples:
                break

        avg_loss = total_loss / max(1, total_tokens)

        return {
            "eval_loss": avg_loss,
            "eval_tokens": total_tokens,
            "eval_samples": total_samples,
        }

    def save_checkpoint(
        self,
        path: Union[str, Path],
        extra_state: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Save a training checkpoint.

        Args:
            path: Path to save checkpoint
            extra_state: Additional state to save
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "epoch": self.epoch,
            "best_eval_loss": self.best_eval_loss,
            "config": self.config,
        }

        if self.scheduler:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        if extra_state:
            checkpoint.update(extra_state)

        torch.save(checkpoint, path)

    def load_checkpoint(
        self,
        path: Union[str, Path],
        load_optimizer: bool = True,
        load_scheduler: bool = True,
    ) -> Dict[str, Any]:
        """
        Load a training checkpoint.

        Args:
            path: Path to checkpoint file
            load_optimizer: Whether to load optimizer state
            load_scheduler: Whether to load scheduler state

        Returns:
            The loaded checkpoint dictionary
        """
        path = Path(path)
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        self.model.load_state_dict(checkpoint["model_state_dict"])

        if load_optimizer and "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if (
            load_scheduler
            and self.scheduler
            and "scheduler_state_dict" in checkpoint
        ):
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.global_step = checkpoint.get("global_step", 0)
        self.epoch = checkpoint.get("epoch", 0)
        self.best_eval_loss = checkpoint.get("best_eval_loss", float("inf"))

        return checkpoint

    def _create_data_iterator(
        self,
        loader: DataLoader,
    ) -> Iterator:
        """Create an infinite iterator over the data loader."""
        while True:
            for batch in loader:
                yield batch
            self.epoch += 1

    def train(
        self,
        train_loader: Optional[DataLoader] = None,
        eval_loader: Optional[DataLoader] = None,
        max_steps: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Run the full training loop.

        Args:
            train_loader: Training data loader (uses self.train_loader if None)
            eval_loader: Validation data loader (uses self.eval_loader if None)
            max_steps: Maximum training steps (uses config.max_steps if None)

        Returns:
            Dictionary with training history
        """
        loader = train_loader or self.train_loader
        eval_loader = eval_loader or self.eval_loader

        if loader is None:
            raise ValueError("No training data loader provided")

        max_steps = max_steps or self.config.max_steps
        data_iter = self._create_data_iterator(loader)

        self.optimizer.zero_grad()
        self.train_metrics.reset()

        history = {
            "train_loss": [],
            "eval_loss": [],
            "learning_rate": [],
        }

        accumulation_count = 0
        start_time = time.time()

        while self.global_step < max_steps:
            batch = next(data_iter)
            src, tgt = batch["src"], batch["tgt"]
            src_mask = batch.get("src_mask")
            tgt_mask = batch.get("tgt_mask")

            step_start = time.time()

            # Training step
            loss, n_tokens = self.train_step(src, tgt, src_mask, tgt_mask)

            step_elapsed = time.time() - step_start
            self.train_metrics.update(
                loss=loss,
                tokens=n_tokens,
                samples=src.size(0),
                elapsed=step_elapsed,
            )

            accumulation_count += 1

            # Optimizer step after accumulation
            if accumulation_count >= self.config.gradient_accumulation_steps:
                self.optimizer_step()
                accumulation_count = 0

                # Logging
                if self.global_step % self.config.log_steps == 0:
                    metrics = {
                        "step": self.global_step,
                        "epoch": self.epoch,
                        "train_loss": self.train_metrics.avg_loss,
                        "tokens_per_sec": self.train_metrics.tokens_per_sec,
                        "learning_rate": self._get_lr(),
                    }
                    self._log(metrics)
                    history["train_loss"].append(
                        (self.global_step, self.train_metrics.avg_loss)
                    )
                    history["learning_rate"].append(
                        (self.global_step, self._get_lr())
                    )
                    self.train_metrics.reset()

                # Evaluation
                if (
                    eval_loader
                    and self.config.eval_steps > 0
                    and self.global_step % self.config.eval_steps == 0
                ):
                    eval_metrics = self.evaluate(
                        eval_loader,
                        max_samples=self.config.eval_samples,
                    )
                    self._log({"step": self.global_step, **eval_metrics})
                    history["eval_loss"].append(
                        (self.global_step, eval_metrics["eval_loss"])
                    )

                    if eval_metrics["eval_loss"] < self.best_eval_loss:
                        self.best_eval_loss = eval_metrics["eval_loss"]

                # Checkpointing
                if (
                    self.config.save_dir
                    and self.config.save_steps > 0
                    and self.global_step % self.config.save_steps == 0
                ):
                    save_path = Path(self.config.save_dir) / f"checkpoint_{self.global_step}.pt"
                    self.save_checkpoint(save_path)

        total_time = time.time() - start_time
        history["total_time"] = total_time
        history["final_step"] = self.global_step

        return history


def create_trainer(
    model: nn.Module,
    optimizer: Optimizer,
    criterion: nn.Module,
    scheduler: Optional[_LRScheduler] = None,
    train_loader: Optional[DataLoader] = None,
    eval_loader: Optional[DataLoader] = None,
    max_steps: int = 100000,
    gradient_accumulation_steps: int = 1,
    max_grad_norm: float = 1.0,
    log_steps: int = 100,
    eval_steps: int = 1000,
    save_steps: int = 5000,
    save_dir: Optional[str] = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    padding_idx: int = 0,
) -> Trainer:
    """
    Convenience function to create a Trainer with common settings.

    Args:
        model: The model to train
        optimizer: The optimizer
        criterion: The loss function
        scheduler: Optional LR scheduler
        train_loader: Training data loader
        eval_loader: Validation data loader
        max_steps: Maximum training steps
        gradient_accumulation_steps: Number of steps to accumulate gradients
        max_grad_norm: Maximum gradient norm for clipping
        log_steps: Log every N steps
        eval_steps: Evaluate every N steps
        save_steps: Save checkpoint every N steps
        save_dir: Directory to save checkpoints
        device: Device to train on
        padding_idx: Padding token index

    Returns:
        Configured Trainer instance
    """
    config = TrainerConfig(
        max_steps=max_steps,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_grad_norm=max_grad_norm,
        log_steps=log_steps,
        eval_steps=eval_steps,
        save_steps=save_steps,
        save_dir=save_dir,
        device=device,
        padding_idx=padding_idx,
    )

    return Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        config=config,
        scheduler=scheduler,
        train_loader=train_loader,
        eval_loader=eval_loader,
    )
