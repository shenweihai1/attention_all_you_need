# Chapter 1: Introduction to the Transformer

## What is a Transformer?

The Transformer is a neural network architecture introduced in the 2017 paper "Attention Is All You Need" by Vaswani et al. It revolutionized natural language processing by replacing recurrent neural networks (RNNs) with a purely attention-based mechanism.

### Why Transformers Matter

Before Transformers, sequence-to-sequence models relied on RNNs (like LSTMs and GRUs) which had two major problems:

1. **Sequential Processing**: RNNs process tokens one at a time, making them slow to train
2. **Long-Range Dependencies**: Information from early tokens tends to get "forgotten" in long sequences

Transformers solve both problems by using **self-attention**, which:
- Processes all tokens in parallel (faster training)
- Directly connects any two positions regardless of distance

## Architecture Overview

The Transformer follows an **encoder-decoder** architecture:

```
Input Sequence → [Encoder] → Hidden Representations → [Decoder] → Output Sequence
```

### High-Level Components

```
┌─────────────────────────────────────────────────────────────┐
│                      TRANSFORMER                            │
├─────────────────────────────┬───────────────────────────────┤
│         ENCODER             │           DECODER             │
├─────────────────────────────┼───────────────────────────────┤
│                             │                               │
│  ┌───────────────────────┐  │  ┌───────────────────────┐   │
│  │    Encoder Layer      │  │  │    Decoder Layer      │   │
│  │  ┌─────────────────┐  │  │  │  ┌─────────────────┐  │   │
│  │  │ Self-Attention  │  │  │  │  │Masked Self-Attn │  │   │
│  │  └─────────────────┘  │  │  │  └─────────────────┘  │   │
│  │  ┌─────────────────┐  │  │  │  ┌─────────────────┐  │   │
│  │  │  Feed-Forward   │  │  │  │  │ Cross-Attention │  │   │
│  │  └─────────────────┘  │  │  │  └─────────────────┘  │   │
│  └───────────────────────┘  │  │  ┌─────────────────┐  │   │
│           × N               │  │  │  Feed-Forward   │  │   │
│                             │  │  └─────────────────┘  │   │
│                             │  └───────────────────────┘   │
│                             │           × N                │
└─────────────────────────────┴───────────────────────────────┘
```

### Key Components in This Codebase

| Component | File | Purpose |
|-----------|------|---------|
| Attention | `src/attention.py` | Scaled dot-product and multi-head attention |
| Feed-Forward | `src/feedforward.py` | Position-wise feed-forward network |
| Positional Encoding | `src/positional_encoding.py` | Sinusoidal position information |
| Encoder | `src/encoder.py` | Encoder layer and stack |
| Decoder | `src/decoder.py` | Decoder layer and stack |
| Embedding | `src/embedding.py` | Token embeddings with scaling |
| Transformer | `src/transformer.py` | Complete encoder-decoder model |

## The Data Flow

Let's trace how data flows through the Transformer during translation:

### Step 1: Input Embedding
```python
# Source sentence: "Hello world"
# Token IDs: [101, 7592, 2088, 102]  (after tokenization)

# Embedding lookup + scaling
embeddings = Embedding(token_ids) * sqrt(d_model)
# Shape: (batch_size, seq_len, d_model)
```

### Step 2: Add Positional Encoding
```python
# Add position information (tokens don't know their position otherwise)
x = embeddings + positional_encoding[:seq_len]
```

### Step 3: Encoder Processing
```python
# Pass through N encoder layers
for encoder_layer in encoder_layers:
    x = encoder_layer(x)  # Self-attention → FFN
# Output: encoder_output (memory for decoder)
```

### Step 4: Decoder Processing
```python
# Target sequence (shifted right for teacher forcing)
# Each decoder layer does:
#   1. Masked self-attention (can't see future tokens)
#   2. Cross-attention (attend to encoder output)
#   3. Feed-forward network
for decoder_layer in decoder_layers:
    y = decoder_layer(y, encoder_output)
```

### Step 5: Output Projection
```python
# Project to vocabulary size and apply softmax
logits = linear(decoder_output)  # Shape: (batch, seq_len, vocab_size)
probabilities = softmax(logits)
```

## Base Model Configuration

The "base" Transformer from the paper uses these hyperparameters:

| Parameter | Value | Description |
|-----------|-------|-------------|
| d_model | 512 | Model dimension (embedding size) |
| n_heads | 8 | Number of attention heads |
| d_k = d_v | 64 | Dimension per head (512 / 8) |
| d_ff | 2048 | Feed-forward inner dimension |
| N | 6 | Number of encoder/decoder layers |
| dropout | 0.1 | Dropout probability |

In our code (from `configs/transformer_config.py`):

```python
from configs.transformer_config import get_base_config

config = get_base_config()
# Returns TransformerConfig with paper's base settings
```

## Project Structure

```
attention_all_you_need/
├── src/                          # Core implementation
│   ├── attention.py              # Attention mechanisms
│   ├── feedforward.py            # FFN module
│   ├── positional_encoding.py    # Position encoding
│   ├── encoder.py                # Encoder layer/stack
│   ├── decoder.py                # Decoder layer/stack
│   ├── embedding.py              # Token embeddings
│   ├── transformer.py            # Complete model
│   ├── init.py                   # Weight initialization
│   ├── scheduler.py              # Learning rate scheduling
│   ├── label_smoothing.py        # Label smoothing loss
│   ├── trainer.py                # Training loop
│   ├── tokenizer.py              # Tokenization
│   └── data.py                   # Data loading
├── tests/                        # Test suite
├── configs/                      # Configuration
└── docs/                         # Documentation
```

## Quick Start Example

```python
from src import Transformer

# Create a Transformer model
model = Transformer(
    src_vocab_size=10000,
    tgt_vocab_size=10000,
    d_model=512,
    n_heads=8,
    n_encoder_layers=6,
    n_decoder_layers=6,
    d_ff=2048,
    dropout=0.1,
    max_seq_len=512,
    pad_idx=0,
)

# Forward pass
import torch
src = torch.randint(1, 10000, (2, 20))  # Batch of 2, length 20
tgt = torch.randint(1, 10000, (2, 15))  # Batch of 2, length 15

logits = model(src, tgt)  # Shape: (2, 15, 10000)
```

## What's Next?

In the following chapters, we'll dive deep into each component:

1. **Chapter 2: Attention Mechanism** - The core innovation of Transformers
2. **Chapter 3: Encoder and Decoder** - How layers are stacked
3. **Chapter 4: Training Infrastructure** - Learning rate, loss, training loop
4. **Chapter 5: Data Processing** - Tokenization and batching
5. **Chapter 6: Practical Examples** - End-to-end tutorials

Each chapter includes:
- Conceptual explanation (the "why")
- Code walkthrough (the "what")
- Mathematical formulas (the "how")
- Practical examples (hands-on learning)

---

*Next: [Chapter 2: The Attention Mechanism](02_attention.md)*
