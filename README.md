# Attention Is All You Need - Transformer Implementation

A from-scratch PyTorch implementation of the Transformer architecture based on the seminal paper ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) by Vaswani et al.

## Overview

This project implements the Transformer model without using `torch.nn.Transformer` or `torch.nn.MultiheadAttention`, providing a clear and educational implementation of all core components.

## Project Structure

```
attention_all_you_need/
├── src/                    # Source code for the Transformer
├── tests/                  # Test suite
├── configs/                # Configuration files and classes
│   ├── transformer_config.py  # Model and training configurations
│   └── base_config.json       # Default base model config
├── docs/                   # Documentation
├── requirements.txt        # Python dependencies
└── pytest.ini             # Pytest configuration
```

## Installation

### Requirements

- Python 3.8+
- PyTorch 2.0+

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd attention_all_you_need

# Install dependencies
pip install -r requirements.txt
```

### Target Server Setup (RTX 5090)

For the target server running `pytorch:1.0.2-cu1281-torch280-ubuntu2404`:

```bash
# The server image already includes PyTorch with CUDA support
# Install additional dependencies
pip install -r requirements.txt
```

## Usage

### Configuration

The model uses a dataclass-based configuration system:

```python
from configs.transformer_config import TransformerConfig, get_base_config

# Use default base configuration (matches paper)
config = get_base_config()

# Or customize
config = TransformerConfig(
    d_model=512,
    n_heads=8,
    n_layers=6,
    d_ff=2048,
    dropout=0.1,
)

# Save/load configuration
config.save("my_config.json")
config = TransformerConfig.load("my_config.json")
```

### Base Model Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| d_model | 512 | Model dimension |
| n_heads | 8 | Number of attention heads |
| n_layers | 6 | Number of encoder/decoder layers |
| d_ff | 2048 | Feed-forward inner dimension |
| d_k, d_v | 64 | Key/Value dimensions per head |
| dropout | 0.1 | Dropout rate |

## Testing

Run the test suite:

```bash
pytest tests/ -v
```

## Development Status

See [TODO.md](TODO.md) for the current development status and planned features.

## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Vaswani et al., 2017
