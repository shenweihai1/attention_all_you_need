"""
Tests for Transformer trainer with gradient accumulation.
"""

import tempfile
from pathlib import Path
from typing import Dict, Iterator, List

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.trainer import (
    Trainer,
    TrainerConfig,
    TrainingMetrics,
    create_trainer,
)
from src.transformer import Transformer
from src.scheduler import TransformerScheduler
from src.label_smoothing import LabelSmoothingLoss


class DummyTranslationDataset(Dataset):
    """Simple dummy dataset for testing."""

    def __init__(
        self,
        num_samples: int = 100,
        src_vocab_size: int = 100,
        tgt_vocab_size: int = 100,
        max_len: int = 20,
        padding_idx: int = 0,
    ):
        self.num_samples = num_samples
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.max_len = max_len
        self.padding_idx = padding_idx

        # Pre-generate data
        self.data = []
        for _ in range(num_samples):
            src_len = torch.randint(5, max_len, (1,)).item()
            tgt_len = torch.randint(5, max_len, (1,)).item()

            # Generate sequences (avoid padding_idx=0 for content)
            src = torch.randint(1, src_vocab_size, (src_len,))
            tgt = torch.randint(1, tgt_vocab_size, (tgt_len,))

            self.data.append({"src": src, "tgt": tgt})

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.data[idx]


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Collate function with padding."""
    src_list = [item["src"] for item in batch]
    tgt_list = [item["tgt"] for item in batch]

    # Pad sequences
    src_padded = nn.utils.rnn.pad_sequence(src_list, batch_first=True, padding_value=0)
    tgt_padded = nn.utils.rnn.pad_sequence(tgt_list, batch_first=True, padding_value=0)

    return {
        "src": src_padded,
        "tgt": tgt_padded,
    }


class TestTrainingMetrics:
    """Tests for TrainingMetrics dataclass."""

    def test_creation(self):
        """Test metrics creation."""
        metrics = TrainingMetrics()
        assert metrics.loss == 0.0
        assert metrics.tokens == 0
        assert metrics.samples == 0
        assert metrics.steps == 0
        assert metrics.elapsed_time == 0.0

    def test_avg_loss(self):
        """Test average loss calculation."""
        metrics = TrainingMetrics(loss=10.0, steps=5)
        assert metrics.avg_loss == 2.0

    def test_avg_loss_zero_steps(self):
        """Test average loss with zero steps doesn't crash."""
        metrics = TrainingMetrics(loss=10.0, steps=0)
        assert metrics.avg_loss == 10.0  # Falls back to max(1, steps)

    def test_tokens_per_sec(self):
        """Test tokens per second calculation."""
        metrics = TrainingMetrics(tokens=1000, elapsed_time=2.0)
        assert metrics.tokens_per_sec == 500.0

    def test_tokens_per_sec_zero_time(self):
        """Test tokens per second with zero time doesn't crash."""
        metrics = TrainingMetrics(tokens=1000, elapsed_time=0.0)
        assert metrics.tokens_per_sec > 0  # Should use max(1e-6, time)

    def test_reset(self):
        """Test metrics reset."""
        metrics = TrainingMetrics(
            loss=10.0,
            tokens=100,
            samples=10,
            steps=5,
            elapsed_time=1.0,
        )
        metrics.reset()
        assert metrics.loss == 0.0
        assert metrics.tokens == 0
        assert metrics.samples == 0
        assert metrics.steps == 0
        assert metrics.elapsed_time == 0.0

    def test_update(self):
        """Test metrics update."""
        metrics = TrainingMetrics()
        metrics.update(loss=2.0, tokens=100, samples=10, elapsed=0.5)
        metrics.update(loss=3.0, tokens=150, samples=15, elapsed=0.7)

        assert metrics.loss == 5.0
        assert metrics.tokens == 250
        assert metrics.samples == 25
        assert metrics.steps == 2
        assert abs(metrics.elapsed_time - 1.2) < 1e-6


class TestTrainerConfig:
    """Tests for TrainerConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = TrainerConfig()
        assert config.max_steps == 100000
        assert config.gradient_accumulation_steps == 1
        assert config.max_grad_norm == 1.0
        assert config.eval_steps == 1000
        assert config.log_steps == 100
        assert config.save_steps == 5000
        assert config.padding_idx == 0

    def test_custom_values(self):
        """Test custom configuration values."""
        config = TrainerConfig(
            max_steps=10000,
            gradient_accumulation_steps=4,
            max_grad_norm=0.5,
            device="cpu",
        )
        assert config.max_steps == 10000
        assert config.gradient_accumulation_steps == 4
        assert config.max_grad_norm == 0.5
        assert config.device == "cpu"


class TestTrainer:
    """Tests for Trainer class."""

    @pytest.fixture
    def small_model(self):
        """Create a small model for testing."""
        return Transformer(
            src_vocab_size=100,
            tgt_vocab_size=100,
            d_model=32,
            n_heads=2,
            n_encoder_layers=1,
            n_decoder_layers=1,
            d_ff=64,
            dropout=0.0,
        )

    @pytest.fixture
    def train_loader(self):
        """Create a small training data loader."""
        dataset = DummyTranslationDataset(num_samples=50, src_vocab_size=100, tgt_vocab_size=100)
        return DataLoader(dataset, batch_size=4, collate_fn=collate_fn)

    @pytest.fixture
    def eval_loader(self):
        """Create a small evaluation data loader."""
        dataset = DummyTranslationDataset(num_samples=20, src_vocab_size=100, tgt_vocab_size=100)
        return DataLoader(dataset, batch_size=4, collate_fn=collate_fn)

    def test_trainer_creation(self, small_model, train_loader):
        """Test trainer creation."""
        optimizer = torch.optim.Adam(small_model.parameters(), lr=0.001)
        criterion = LabelSmoothingLoss(smoothing=0.1, padding_idx=0)
        config = TrainerConfig(max_steps=10, device="cpu")

        trainer = Trainer(
            model=small_model,
            optimizer=optimizer,
            criterion=criterion,
            config=config,
            train_loader=train_loader,
        )

        assert trainer.global_step == 0
        assert trainer.epoch == 0

    def test_train_step(self, small_model, train_loader):
        """Test single training step."""
        optimizer = torch.optim.Adam(small_model.parameters(), lr=0.001)
        criterion = LabelSmoothingLoss(smoothing=0.1, padding_idx=0)
        config = TrainerConfig(device="cpu")

        trainer = Trainer(
            model=small_model,
            optimizer=optimizer,
            criterion=criterion,
            config=config,
        )

        # Get a batch
        batch = next(iter(train_loader))
        src, tgt = batch["src"], batch["tgt"]

        loss, n_tokens = trainer.train_step(src, tgt)

        assert isinstance(loss, float)
        assert loss > 0
        assert n_tokens > 0

    def test_optimizer_step(self, small_model):
        """Test optimizer step increments global_step."""
        optimizer = torch.optim.Adam(small_model.parameters(), lr=0.001)
        criterion = LabelSmoothingLoss(smoothing=0.1, padding_idx=0)
        config = TrainerConfig(device="cpu")

        trainer = Trainer(
            model=small_model,
            optimizer=optimizer,
            criterion=criterion,
            config=config,
        )

        initial_step = trainer.global_step
        trainer.optimizer_step()
        assert trainer.global_step == initial_step + 1

    def test_gradient_accumulation(self, small_model, train_loader):
        """Test gradient accumulation."""
        optimizer = torch.optim.Adam(small_model.parameters(), lr=0.001)
        criterion = LabelSmoothingLoss(smoothing=0.1, padding_idx=0)
        config = TrainerConfig(
            max_steps=4,
            gradient_accumulation_steps=2,
            device="cpu",
        )

        trainer = Trainer(
            model=small_model,
            optimizer=optimizer,
            criterion=criterion,
            config=config,
            train_loader=train_loader,
        )

        history = trainer.train()

        # With accumulation_steps=2, we should have 4 steps
        assert trainer.global_step == 4

    def test_gradient_clipping(self, small_model, train_loader):
        """Test that gradient clipping is applied."""
        optimizer = torch.optim.Adam(small_model.parameters(), lr=0.001)
        criterion = LabelSmoothingLoss(smoothing=0.1, padding_idx=0)
        config = TrainerConfig(
            max_steps=2,
            max_grad_norm=0.1,  # Very small to ensure clipping happens
            device="cpu",
        )

        trainer = Trainer(
            model=small_model,
            optimizer=optimizer,
            criterion=criterion,
            config=config,
            train_loader=train_loader,
        )

        # Should complete without error
        trainer.train()

    def test_training_loop(self, small_model, train_loader):
        """Test full training loop."""
        optimizer = torch.optim.Adam(small_model.parameters(), lr=0.001)
        criterion = LabelSmoothingLoss(smoothing=0.1, padding_idx=0)
        config = TrainerConfig(
            max_steps=10,
            log_steps=5,
            eval_steps=0,  # Disable eval
            save_steps=0,  # Disable saving
            device="cpu",
        )

        trainer = Trainer(
            model=small_model,
            optimizer=optimizer,
            criterion=criterion,
            config=config,
            train_loader=train_loader,
        )

        history = trainer.train()

        assert trainer.global_step == 10
        assert "train_loss" in history
        assert len(history["train_loss"]) >= 1

    def test_evaluation(self, small_model, eval_loader):
        """Test evaluation loop."""
        optimizer = torch.optim.Adam(small_model.parameters(), lr=0.001)
        criterion = LabelSmoothingLoss(smoothing=0.1, padding_idx=0)
        config = TrainerConfig(device="cpu")

        trainer = Trainer(
            model=small_model,
            optimizer=optimizer,
            criterion=criterion,
            config=config,
            eval_loader=eval_loader,
        )

        eval_metrics = trainer.evaluate()

        assert "eval_loss" in eval_metrics
        assert "eval_tokens" in eval_metrics
        assert "eval_samples" in eval_metrics
        assert eval_metrics["eval_loss"] > 0

    def test_checkpoint_save_load(self, small_model, train_loader):
        """Test checkpoint saving and loading."""
        optimizer = torch.optim.Adam(small_model.parameters(), lr=0.001)
        criterion = LabelSmoothingLoss(smoothing=0.1, padding_idx=0)
        config = TrainerConfig(max_steps=5, device="cpu")

        trainer = Trainer(
            model=small_model,
            optimizer=optimizer,
            criterion=criterion,
            config=config,
            train_loader=train_loader,
        )

        # Train a bit
        trainer.train()

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint.pt"
            trainer.save_checkpoint(checkpoint_path)

            # Create new trainer and load
            new_model = Transformer(
                src_vocab_size=100,
                tgt_vocab_size=100,
                d_model=32,
                n_heads=2,
                n_encoder_layers=1,
                n_decoder_layers=1,
                d_ff=64,
                dropout=0.0,
            )
            new_optimizer = torch.optim.Adam(new_model.parameters(), lr=0.001)

            new_trainer = Trainer(
                model=new_model,
                optimizer=new_optimizer,
                criterion=criterion,
                config=config,
            )

            new_trainer.load_checkpoint(checkpoint_path)

            assert new_trainer.global_step == trainer.global_step
            assert new_trainer.epoch == trainer.epoch

    def test_with_scheduler(self, small_model, train_loader):
        """Test training with learning rate scheduler."""
        optimizer = torch.optim.Adam(small_model.parameters(), lr=1.0)
        criterion = LabelSmoothingLoss(smoothing=0.1, padding_idx=0)
        scheduler = TransformerScheduler(optimizer, d_model=32, warmup_steps=10)
        config = TrainerConfig(max_steps=5, device="cpu")

        trainer = Trainer(
            model=small_model,
            optimizer=optimizer,
            criterion=criterion,
            config=config,
            scheduler=scheduler,
            train_loader=train_loader,
        )

        trainer.train()

        # Scheduler should have stepped
        assert scheduler._step_count >= 5

    def test_log_callback(self, small_model, train_loader):
        """Test log callback functionality."""
        optimizer = torch.optim.Adam(small_model.parameters(), lr=0.001)
        criterion = LabelSmoothingLoss(smoothing=0.1, padding_idx=0)
        config = TrainerConfig(
            max_steps=10,
            log_steps=5,
            eval_steps=0,
            device="cpu",
        )

        trainer = Trainer(
            model=small_model,
            optimizer=optimizer,
            criterion=criterion,
            config=config,
            train_loader=train_loader,
        )

        logged_metrics = []
        trainer.set_log_callback(lambda m: logged_metrics.append(m))

        trainer.train()

        assert len(logged_metrics) >= 1
        assert "step" in logged_metrics[0]
        assert "train_loss" in logged_metrics[0]

    def test_count_tokens(self, small_model):
        """Test token counting."""
        optimizer = torch.optim.Adam(small_model.parameters(), lr=0.001)
        criterion = LabelSmoothingLoss(smoothing=0.1, padding_idx=0)
        config = TrainerConfig(device="cpu", padding_idx=0)

        trainer = Trainer(
            model=small_model,
            optimizer=optimizer,
            criterion=criterion,
            config=config,
        )

        target = torch.tensor([[1, 2, 3, 0, 0], [4, 5, 0, 0, 0]])
        n_tokens = trainer._count_tokens(target, padding_idx=0)
        assert n_tokens == 5  # 3 + 2 non-padding tokens


class TestCreateTrainer:
    """Tests for create_trainer convenience function."""

    def test_basic_creation(self):
        """Test basic trainer creation."""
        model = Transformer(
            src_vocab_size=100,
            tgt_vocab_size=100,
            d_model=32,
            n_heads=2,
            n_encoder_layers=1,
            n_decoder_layers=1,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = LabelSmoothingLoss(smoothing=0.1)

        trainer = create_trainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            max_steps=100,
            device="cpu",
        )

        assert isinstance(trainer, Trainer)
        assert trainer.config.max_steps == 100

    def test_with_all_options(self):
        """Test creation with all options."""
        model = Transformer(
            src_vocab_size=100,
            tgt_vocab_size=100,
            d_model=32,
            n_heads=2,
            n_encoder_layers=1,
            n_decoder_layers=1,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = LabelSmoothingLoss(smoothing=0.1)
        scheduler = TransformerScheduler(optimizer, d_model=32)

        trainer = create_trainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            scheduler=scheduler,
            max_steps=1000,
            gradient_accumulation_steps=4,
            max_grad_norm=0.5,
            log_steps=50,
            eval_steps=100,
            save_steps=500,
            device="cpu",
            padding_idx=1,
        )

        assert trainer.config.max_steps == 1000
        assert trainer.config.gradient_accumulation_steps == 4
        assert trainer.config.max_grad_norm == 0.5
        assert trainer.config.log_steps == 50
        assert trainer.config.padding_idx == 1


class TestTrainerEdgeCases:
    """Tests for edge cases."""

    def test_no_eval_loader(self):
        """Test training without evaluation loader."""
        model = Transformer(
            src_vocab_size=100,
            tgt_vocab_size=100,
            d_model=32,
            n_heads=2,
            n_encoder_layers=1,
            n_decoder_layers=1,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = LabelSmoothingLoss(smoothing=0.1, padding_idx=0)
        config = TrainerConfig(max_steps=5, eval_steps=2, device="cpu")

        dataset = DummyTranslationDataset(num_samples=20, src_vocab_size=100, tgt_vocab_size=100)
        train_loader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            config=config,
            train_loader=train_loader,
            # No eval_loader
        )

        # Should complete without error
        trainer.train()

    def test_no_train_loader_raises(self):
        """Test that training without train loader raises error."""
        model = Transformer(
            src_vocab_size=100,
            tgt_vocab_size=100,
            d_model=32,
            n_heads=2,
            n_encoder_layers=1,
            n_decoder_layers=1,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = LabelSmoothingLoss(smoothing=0.1)
        config = TrainerConfig(max_steps=5, device="cpu")

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            config=config,
        )

        with pytest.raises(ValueError, match="No training data loader"):
            trainer.train()

    def test_empty_evaluate_without_loader(self):
        """Test that evaluate returns empty dict without loader."""
        model = Transformer(
            src_vocab_size=100,
            tgt_vocab_size=100,
            d_model=32,
            n_heads=2,
            n_encoder_layers=1,
            n_decoder_layers=1,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = LabelSmoothingLoss(smoothing=0.1)
        config = TrainerConfig(device="cpu")

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            config=config,
        )

        result = trainer.evaluate()
        assert result == {}

    def test_max_samples_in_eval(self):
        """Test evaluation with max_samples limit."""
        model = Transformer(
            src_vocab_size=100,
            tgt_vocab_size=100,
            d_model=32,
            n_heads=2,
            n_encoder_layers=1,
            n_decoder_layers=1,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = LabelSmoothingLoss(smoothing=0.1, padding_idx=0)
        config = TrainerConfig(device="cpu")

        dataset = DummyTranslationDataset(num_samples=100, src_vocab_size=100, tgt_vocab_size=100)
        eval_loader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            config=config,
            eval_loader=eval_loader,
        )

        # Limit to 8 samples (2 batches)
        result = trainer.evaluate(max_samples=8)
        assert result["eval_samples"] <= 8

    def test_no_grad_clipping(self):
        """Test training without gradient clipping."""
        model = Transformer(
            src_vocab_size=100,
            tgt_vocab_size=100,
            d_model=32,
            n_heads=2,
            n_encoder_layers=1,
            n_decoder_layers=1,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = LabelSmoothingLoss(smoothing=0.1, padding_idx=0)
        config = TrainerConfig(
            max_steps=5,
            max_grad_norm=None,  # Disable gradient clipping
            device="cpu",
        )

        dataset = DummyTranslationDataset(num_samples=20, src_vocab_size=100, tgt_vocab_size=100)
        train_loader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            config=config,
            train_loader=train_loader,
        )

        # Should complete without error
        trainer.train()
