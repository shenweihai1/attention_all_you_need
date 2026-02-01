"""
Configuration classes for the Transformer model.

Based on "Attention Is All You Need" paper hyperparameters.
"""

from dataclasses import dataclass, field
from typing import Optional
import json


@dataclass
class TransformerConfig:
    """
    Configuration for the Transformer model.

    Default values correspond to the "base" model from the paper:
    - d_model = 512 (model dimension)
    - n_heads = 8 (number of attention heads)
    - n_layers = 6 (number of encoder/decoder layers)
    - d_ff = 2048 (feed-forward inner dimension)
    - dropout = 0.1
    - max_seq_len = 512 (maximum sequence length)
    """

    # Model architecture
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 2048

    # Vocabulary
    src_vocab_size: int = 32000
    tgt_vocab_size: int = 32000

    # Regularization
    dropout: float = 0.1
    attention_dropout: float = 0.1

    # Sequence length
    max_seq_len: int = 512

    # Padding token index
    pad_idx: int = 0

    # Derived dimensions (computed from d_model and n_heads)
    @property
    def d_k(self) -> int:
        """Dimension of keys/queries per head."""
        return self.d_model // self.n_heads

    @property
    def d_v(self) -> int:
        """Dimension of values per head."""
        return self.d_model // self.n_heads

    def __post_init__(self):
        """Validate configuration parameters."""
        # Check positive values first to avoid division by zero
        if self.d_model <= 0:
            raise ValueError(f"d_model must be positive, got {self.d_model}")
        if self.n_heads <= 0:
            raise ValueError(f"n_heads must be positive, got {self.n_heads}")
        if self.n_layers <= 0:
            raise ValueError(f"n_layers must be positive, got {self.n_layers}")
        # Now safe to check divisibility
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
            )
        if not 0 <= self.dropout < 1:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "n_layers": self.n_layers,
            "d_ff": self.d_ff,
            "src_vocab_size": self.src_vocab_size,
            "tgt_vocab_size": self.tgt_vocab_size,
            "dropout": self.dropout,
            "attention_dropout": self.attention_dropout,
            "max_seq_len": self.max_seq_len,
            "pad_idx": self.pad_idx,
        }

    @classmethod
    def from_dict(cls, config_dict: dict) -> "TransformerConfig":
        """Create config from dictionary."""
        return cls(**config_dict)

    def save(self, path: str) -> None:
        """Save config to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "TransformerConfig":
        """Load config from JSON file."""
        with open(path, "r") as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


@dataclass
class TrainingConfig:
    """
    Configuration for training the Transformer model.

    Default values based on the paper's training setup.
    """

    # Optimizer settings
    learning_rate: float = 0.0001  # Will be adjusted by scheduler
    beta1: float = 0.9
    beta2: float = 0.98
    epsilon: float = 1e-9

    # Learning rate schedule (warmup + inverse sqrt decay)
    warmup_steps: int = 4000

    # Batch settings
    batch_size: int = 32
    accumulation_steps: int = 1  # For gradient accumulation

    # Training duration
    max_steps: int = 100000

    # Label smoothing
    label_smoothing: float = 0.1

    # Checkpointing
    save_every: int = 5000
    eval_every: int = 1000

    # Logging
    log_every: int = 100

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            "learning_rate": self.learning_rate,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "epsilon": self.epsilon,
            "warmup_steps": self.warmup_steps,
            "batch_size": self.batch_size,
            "accumulation_steps": self.accumulation_steps,
            "max_steps": self.max_steps,
            "label_smoothing": self.label_smoothing,
            "save_every": self.save_every,
            "eval_every": self.eval_every,
            "log_every": self.log_every,
        }

    @classmethod
    def from_dict(cls, config_dict: dict) -> "TrainingConfig":
        """Create config from dictionary."""
        return cls(**config_dict)

    def save(self, path: str) -> None:
        """Save config to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "TrainingConfig":
        """Load config from JSON file."""
        with open(path, "r") as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


def get_base_config() -> TransformerConfig:
    """Get the base Transformer configuration from the paper."""
    return TransformerConfig()


def get_big_config() -> TransformerConfig:
    """Get the big Transformer configuration from the paper."""
    return TransformerConfig(
        d_model=1024,
        n_heads=16,
        n_layers=6,
        d_ff=4096,
        dropout=0.3,
    )
