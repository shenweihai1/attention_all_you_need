# Attention Is All You Need - Transformer Implementation

A complete from-scratch PyTorch implementation of the Transformer architecture based on the paper ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) by Vaswani et al. (2017).

## Overview

This project implements the Transformer model **without using `torch.nn.Transformer` or `torch.nn.MultiheadAttention`**, providing a clear and educational implementation of all core components as described in the original paper.

### Key Features

- **Complete Transformer architecture**: Encoder-decoder with multi-head attention
- **Training infrastructure**: Learning rate scheduling, label smoothing, gradient accumulation
- **Data processing**: BPE tokenization, dynamic batching, WMT dataset support
- **Extensive test suite**: 466+ tests covering all components
- **Production-ready**: Checkpoint saving/loading, configurable training

## Project Structure

```
attention_all_you_need/
├── src/                          # Source code
│   ├── attention.py              # Scaled dot-product & multi-head attention
│   ├── feedforward.py            # Position-wise feed-forward network
│   ├── positional_encoding.py    # Sinusoidal positional encoding
│   ├── encoder.py                # Encoder layer and stack
│   ├── decoder.py                # Decoder layer and stack
│   ├── embedding.py              # Scaled embeddings
│   ├── transformer.py            # Full Transformer model
│   ├── init.py                   # Weight initialization utilities
│   ├── scheduler.py              # Learning rate schedulers
│   ├── label_smoothing.py        # Label smoothing loss
│   ├── trainer.py                # Training loop with gradient accumulation
│   ├── tokenizer.py              # Tokenization utilities (BPE support)
│   └── data.py                   # Dataset and data loading utilities
├── tests/                        # Comprehensive test suite
├── configs/                      # Configuration system
│   ├── transformer_config.py     # Model and training configurations
│   └── base_config.json          # Default base model config
├── docs/                         # Documentation
├── notebooks/                    # Jupyter tutorial notebooks
│   ├── 01_training_tutorial.ipynb    # Step-by-step training guide
│   └── 02_inference_tutorial.ipynb   # Inference and generation guide
├── requirements.txt              # Python dependencies
└── pytest.ini                    # Pytest configuration
```

## Installation

### Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA 12.1+ (for GPU training)

### Quick Setup

```bash
# Clone the repository
git clone https://github.com/shenweihai1/attention_all_you_need.git
cd attention_all_you_need

# Install dependencies
pip install -r requirements.txt

# Verify installation
pytest tests/ -v --tb=short
```

### Target Server Setup (RTX 5090)

For the target server running `pytorch:1.0.2-cu1281-torch280-ubuntu2404`:

```bash
# The server image already includes PyTorch with CUDA support
# Install additional dependencies
pip install -r requirements.txt

# Optional: Install SentencePiece for BPE tokenization
pip install sentencepiece

# Optional: Install HuggingFace datasets for WMT data
pip install datasets
```

## Tutorial Notebooks

Interactive Jupyter notebooks are available to help you get started:

- **[01_training_tutorial.ipynb](notebooks/01_training_tutorial.ipynb)** - Step-by-step guide covering data preparation, tokenization, model creation, and training with validation at each step
- **[02_inference_tutorial.ipynb](notebooks/02_inference_tutorial.ipynb)** - Guide to loading models, greedy decoding, batch inference, and examining model internals

Run them with:
```bash
cd notebooks
jupyter notebook
```

## Quick Start

### Creating a Transformer Model

```python
from src import Transformer

# Create a Transformer model (base configuration from the paper)
model = Transformer(
    src_vocab_size=37000,    # Source vocabulary size
    tgt_vocab_size=37000,    # Target vocabulary size
    d_model=512,             # Model dimension
    n_heads=8,               # Number of attention heads
    n_encoder_layers=6,      # Encoder layers
    n_decoder_layers=6,      # Decoder layers
    d_ff=2048,               # Feed-forward dimension
    dropout=0.1,             # Dropout rate
    max_seq_len=5000,        # Maximum sequence length
    pad_idx=0,               # Padding token index
)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
```

### Forward Pass

```python
import torch

# Example input (batch_size=2, seq_len=10)
src = torch.randint(1, 37000, (2, 10))  # Source sequences
tgt = torch.randint(1, 37000, (2, 8))   # Target sequences

# Forward pass returns logits
logits = model(src, tgt)  # Shape: (2, 8, 37000)
```

### Autoregressive Generation

```python
# Generate translation (greedy decoding)
generated = model.generate(
    src=src,
    max_length=50,
    bos_token_id=2,  # <s>
    eos_token_id=3,  # </s>
)
```

## Training

### Complete Training Example

```python
import torch
from torch.optim import Adam

from src import (
    Transformer,
    TransformerScheduler,
    LabelSmoothingLoss,
    Trainer,
    TrainerConfig,
    SimpleTokenizer,
    create_translation_dataloader,
)

# 1. Prepare data
src_sentences = ["hello world", "how are you", ...]
tgt_sentences = ["hallo welt", "wie geht es dir", ...]

# 2. Create tokenizer
tokenizer = SimpleTokenizer()
tokenizer.build_vocab(src_sentences + tgt_sentences)

# 3. Create dataloader
train_loader = create_translation_dataloader(
    src_data=src_sentences,
    tgt_data=tgt_sentences,
    src_tokenizer=tokenizer,
    tgt_tokenizer=tokenizer,
    batch_size=32,
    shuffle=True,
)

# 4. Create model
model = Transformer(
    src_vocab_size=tokenizer.vocab_size,
    tgt_vocab_size=tokenizer.vocab_size,
    d_model=512,
    n_heads=8,
    n_encoder_layers=6,
    n_decoder_layers=6,
)

# 5. Setup optimizer with paper's settings
optimizer = Adam(
    model.parameters(),
    lr=1.0,  # Will be controlled by scheduler
    betas=(0.9, 0.98),
    eps=1e-9,
)

# 6. Learning rate scheduler (warmup + inverse sqrt decay)
scheduler = TransformerScheduler(
    optimizer,
    d_model=512,
    warmup_steps=4000,
)

# 7. Label smoothing loss (epsilon=0.1)
criterion = LabelSmoothingLoss(
    smoothing=0.1,
    padding_idx=0,
)

# 8. Configure trainer
config = TrainerConfig(
    max_steps=100000,
    gradient_accumulation_steps=4,  # Effective batch size = 32 * 4 = 128
    max_grad_norm=1.0,
    log_steps=100,
    eval_steps=1000,
    save_steps=5000,
    save_dir="checkpoints",
    device="cuda",
)

# 9. Create trainer and train
trainer = Trainer(
    model=model,
    optimizer=optimizer,
    criterion=criterion,
    config=config,
    scheduler=scheduler,
    train_loader=train_loader,
)

history = trainer.train()
```

### Using Dynamic Batching (Token-based)

For more efficient GPU memory utilization:

```python
from src import create_dynamic_dataloader, TranslationDataset

dataset = TranslationDataset(
    src_data=src_sentences,
    tgt_data=tgt_sentences,
    src_tokenizer=tokenizer,
    tgt_tokenizer=tokenizer,
)

# Create dataloader with max tokens per batch instead of fixed batch size
train_loader = create_dynamic_dataloader(
    dataset=dataset,
    max_tokens=4096,      # Max tokens per batch
    max_sentences=128,    # Optional cap on sentences
    shuffle=True,
)
```

### Checkpoint Management

```python
# Save checkpoint
trainer.save_checkpoint("checkpoints/model_step_10000.pt")

# Load checkpoint and resume training
trainer.load_checkpoint("checkpoints/model_step_10000.pt")
trainer.train()  # Continues from saved step
```

## Model Architecture Details

### Base Model Parameters (from Paper)

| Parameter | Value | Description |
|-----------|-------|-------------|
| d_model | 512 | Model/embedding dimension |
| n_heads | 8 | Number of attention heads |
| d_k, d_v | 64 | Key/Value dimensions (d_model / n_heads) |
| n_layers | 6 | Encoder and decoder layers |
| d_ff | 2048 | Feed-forward inner dimension |
| dropout | 0.1 | Dropout probability |
| warmup_steps | 4000 | Learning rate warmup steps |
| label_smoothing | 0.1 | Label smoothing epsilon |

### Components

1. **Scaled Dot-Product Attention**
   ```
   Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
   ```

2. **Multi-Head Attention**
   ```
   MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
   where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
   ```

3. **Position-wise Feed-Forward Network**
   ```
   FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
   ```

4. **Sinusoidal Positional Encoding**
   ```
   PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
   PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
   ```

5. **Learning Rate Schedule**
   ```
   lrate = d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))
   ```

## Using Pre-trained Tokenizers (BPE)

For production use with BPE tokenization:

```python
from src import Tokenizer

# Train a new BPE tokenizer
tokenizer = Tokenizer.train(
    input_files=["train.en", "train.de"],
    model_prefix="wmt_bpe",
    vocab_size=37000,  # Shared vocabulary as in paper
)

# Or load existing model
tokenizer = Tokenizer(model_path="wmt_bpe.model")

# Encode/decode
ids = tokenizer.encode("Hello world", add_bos=True, add_eos=True)
text = tokenizer.decode(ids)
```

## Loading WMT Dataset

```python
from src.data import load_wmt_dataset

# Load WMT14 English-German (requires 'datasets' package)
dataset = load_wmt_dataset(
    name="wmt14",
    language_pair="de-en",
    split="train",
    src_tokenizer=tokenizer,
    tgt_tokenizer=tokenizer,
    max_samples=100000,  # Optional limit
)
```

## Testing

Run the complete test suite:

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_transformer.py -v

# Run with coverage
pytest tests/ -v --cov=src

# Quick smoke test
pytest tests/test_attention.py -v
```

## Configuration System

```python
from configs.transformer_config import TransformerConfig, get_base_config

# Use default base configuration
config = get_base_config()

# Customize configuration
config = TransformerConfig(
    d_model=512,
    n_heads=8,
    n_layers=6,
    d_ff=2048,
    dropout=0.1,
    vocab_size=37000,
    max_seq_len=512,
)

# Save and load
config.save("my_config.json")
config = TransformerConfig.load("my_config.json")
```

## TODOs
- [ ] Serve an open LLM with high throughput + low latency on your GPU, such as: continuous batching, **KV-cache**, Quantization, Rate limit, Metrics, **RAG**
- [ ] Agent frameworks to implement apas
- [ ] Small RAG projects

## Performance Tips

1. **Use gradient accumulation** for larger effective batch sizes without more GPU memory
2. **Use dynamic batching** (`create_dynamic_dataloader`) for efficient memory utilization
3. **Enable mixed precision** training for faster computation on modern GPUs
4. **Use SortedBatchSampler** to minimize padding overhead

## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Vaswani et al., 2017
- [The Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html) - Harvard NLP
- [SentencePiece](https://github.com/google/sentencepiece) - Subword tokenization

## License

MIT License
