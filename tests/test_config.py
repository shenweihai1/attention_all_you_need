"""
Tests for the Transformer configuration module.
"""

import json
import os
import tempfile
import pytest

from configs.transformer_config import (
    TransformerConfig,
    TrainingConfig,
    get_base_config,
    get_big_config,
)


class TestTransformerConfig:
    """Tests for TransformerConfig class."""

    def test_default_config(self):
        """Test that default config matches base model from paper."""
        config = TransformerConfig()
        assert config.d_model == 512
        assert config.n_heads == 8
        assert config.n_layers == 6
        assert config.d_ff == 2048
        assert config.dropout == 0.1
        assert config.max_seq_len == 512

    def test_derived_dimensions(self):
        """Test d_k and d_v are correctly computed."""
        config = TransformerConfig()
        assert config.d_k == 64  # 512 / 8
        assert config.d_v == 64  # 512 / 8

        config = TransformerConfig(d_model=1024, n_heads=16)
        assert config.d_k == 64  # 1024 / 16
        assert config.d_v == 64  # 1024 / 16

    def test_custom_config(self):
        """Test creating config with custom values."""
        config = TransformerConfig(
            d_model=256,
            n_heads=4,
            n_layers=3,
            d_ff=1024,
            dropout=0.2,
        )
        assert config.d_model == 256
        assert config.n_heads == 4
        assert config.n_layers == 3
        assert config.d_ff == 1024
        assert config.dropout == 0.2
        assert config.d_k == 64  # 256 / 4

    def test_validation_d_model_divisible_by_n_heads(self):
        """Test that d_model must be divisible by n_heads."""
        with pytest.raises(ValueError, match="divisible"):
            TransformerConfig(d_model=512, n_heads=7)

    def test_validation_positive_d_model(self):
        """Test that d_model must be positive."""
        with pytest.raises(ValueError, match="positive"):
            TransformerConfig(d_model=0)
        with pytest.raises(ValueError, match="positive"):
            TransformerConfig(d_model=-512)

    def test_validation_positive_n_heads(self):
        """Test that n_heads must be positive."""
        with pytest.raises(ValueError, match="positive"):
            TransformerConfig(n_heads=0)

    def test_validation_positive_n_layers(self):
        """Test that n_layers must be positive."""
        with pytest.raises(ValueError, match="positive"):
            TransformerConfig(n_layers=0)

    def test_validation_dropout_range(self):
        """Test that dropout must be in [0, 1)."""
        with pytest.raises(ValueError, match="dropout"):
            TransformerConfig(dropout=1.0)
        with pytest.raises(ValueError, match="dropout"):
            TransformerConfig(dropout=-0.1)
        # Valid edge case
        config = TransformerConfig(dropout=0.0)
        assert config.dropout == 0.0

    def test_to_dict(self):
        """Test config serialization to dict."""
        config = TransformerConfig()
        config_dict = config.to_dict()

        assert isinstance(config_dict, dict)
        assert config_dict["d_model"] == 512
        assert config_dict["n_heads"] == 8
        assert config_dict["n_layers"] == 6
        assert "d_k" not in config_dict  # Derived, not stored

    def test_from_dict(self):
        """Test config deserialization from dict."""
        config_dict = {
            "d_model": 256,
            "n_heads": 4,
            "n_layers": 3,
            "d_ff": 1024,
            "src_vocab_size": 16000,
            "tgt_vocab_size": 16000,
            "dropout": 0.2,
            "attention_dropout": 0.1,
            "max_seq_len": 256,
            "pad_idx": 0,
        }
        config = TransformerConfig.from_dict(config_dict)

        assert config.d_model == 256
        assert config.n_heads == 4
        assert config.n_layers == 3

    def test_save_and_load(self):
        """Test saving and loading config to/from JSON file."""
        config = TransformerConfig(d_model=256, n_heads=4, n_layers=3)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            config.save(temp_path)
            loaded_config = TransformerConfig.load(temp_path)

            assert loaded_config.d_model == config.d_model
            assert loaded_config.n_heads == config.n_heads
            assert loaded_config.n_layers == config.n_layers
        finally:
            os.unlink(temp_path)

    def test_roundtrip_dict(self):
        """Test that config survives dict roundtrip."""
        original = TransformerConfig(
            d_model=256,
            n_heads=4,
            n_layers=3,
            d_ff=1024,
            dropout=0.15,
        )
        restored = TransformerConfig.from_dict(original.to_dict())

        assert original.d_model == restored.d_model
        assert original.n_heads == restored.n_heads
        assert original.n_layers == restored.n_layers
        assert original.d_ff == restored.d_ff
        assert original.dropout == restored.dropout


class TestTrainingConfig:
    """Tests for TrainingConfig class."""

    def test_default_config(self):
        """Test default training config values."""
        config = TrainingConfig()
        assert config.warmup_steps == 4000
        assert config.beta1 == 0.9
        assert config.beta2 == 0.98
        assert config.label_smoothing == 0.1

    def test_custom_config(self):
        """Test creating custom training config."""
        config = TrainingConfig(
            batch_size=64,
            warmup_steps=8000,
            max_steps=200000,
        )
        assert config.batch_size == 64
        assert config.warmup_steps == 8000
        assert config.max_steps == 200000

    def test_to_dict(self):
        """Test training config serialization."""
        config = TrainingConfig()
        config_dict = config.to_dict()

        assert isinstance(config_dict, dict)
        assert config_dict["warmup_steps"] == 4000
        assert config_dict["batch_size"] == 32

    def test_from_dict(self):
        """Test training config deserialization."""
        config_dict = {
            "learning_rate": 0.0002,
            "beta1": 0.9,
            "beta2": 0.98,
            "epsilon": 1e-9,
            "warmup_steps": 8000,
            "batch_size": 64,
            "accumulation_steps": 2,
            "max_steps": 200000,
            "label_smoothing": 0.1,
            "save_every": 10000,
            "eval_every": 2000,
            "log_every": 50,
        }
        config = TrainingConfig.from_dict(config_dict)
        assert config.warmup_steps == 8000
        assert config.batch_size == 64

    def test_save_and_load(self):
        """Test saving and loading training config."""
        config = TrainingConfig(batch_size=64, warmup_steps=8000)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            config.save(temp_path)
            loaded_config = TrainingConfig.load(temp_path)

            assert loaded_config.batch_size == config.batch_size
            assert loaded_config.warmup_steps == config.warmup_steps
        finally:
            os.unlink(temp_path)


class TestConfigFactoryFunctions:
    """Tests for configuration factory functions."""

    def test_get_base_config(self):
        """Test base config factory function."""
        config = get_base_config()
        assert config.d_model == 512
        assert config.n_heads == 8
        assert config.n_layers == 6
        assert config.d_ff == 2048

    def test_get_big_config(self):
        """Test big config factory function."""
        config = get_big_config()
        assert config.d_model == 1024
        assert config.n_heads == 16
        assert config.n_layers == 6
        assert config.d_ff == 4096
        assert config.dropout == 0.3


class TestConfigJSONFile:
    """Tests for the base_config.json file."""

    def test_base_config_json_loads(self):
        """Test that base_config.json is valid and loadable."""
        config_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "configs",
            "base_config.json",
        )

        config = TransformerConfig.load(config_path)

        assert config.d_model == 512
        assert config.n_heads == 8
        assert config.n_layers == 6

    def test_base_config_json_valid_syntax(self):
        """Test that base_config.json has valid JSON syntax."""
        config_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "configs",
            "base_config.json",
        )

        with open(config_path, "r") as f:
            config_dict = json.load(f)

        assert isinstance(config_dict, dict)
        assert "d_model" in config_dict
