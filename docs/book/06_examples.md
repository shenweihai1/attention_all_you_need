# Chapter 6: Practical Examples

## Overview

This chapter provides complete, runnable examples:
1. **Quick Start** - Create and use a Transformer model
2. **Training Example** - Full training pipeline
3. **Generation** - Autoregressive sequence generation
4. **Custom Configuration** - Building your own architecture

## Example 1: Quick Start

### Creating a Model

```python
import torch
from src import Transformer

# Create a Transformer model
model = Transformer(
    src_vocab_size=10000,   # Source vocabulary size
    tgt_vocab_size=10000,   # Target vocabulary size
    d_model=512,            # Model dimension
    n_heads=8,              # Number of attention heads
    n_encoder_layers=6,     # Number of encoder layers
    n_decoder_layers=6,     # Number of decoder layers
    d_ff=2048,              # Feed-forward dimension
    dropout=0.1,            # Dropout probability
    max_seq_len=512,        # Maximum sequence length
    pad_idx=0,              # Padding token index
)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
# Model parameters: ~44,000,000
```

### Forward Pass

```python
# Create sample inputs
batch_size = 2
src_len = 20
tgt_len = 15

src = torch.randint(1, 10000, (batch_size, src_len))  # Source tokens
tgt = torch.randint(1, 10000, (batch_size, tgt_len))  # Target tokens

# Forward pass
logits = model(src, tgt)
print(f"Output shape: {logits.shape}")
# Output shape: torch.Size([2, 15, 10000])

# Get predictions
predictions = logits.argmax(dim=-1)
print(f"Predictions shape: {predictions.shape}")
# Predictions shape: torch.Size([2, 15])
```

### With Padding

```python
# Create sequences with padding (pad_idx=0)
src = torch.tensor([
    [2, 101, 102, 103, 3, 0, 0],  # "<s> Hello world </s> <pad> <pad>"
    [2, 201, 202, 203, 204, 205, 3],  # "<s> Longer sentence </s>"
])

tgt = torch.tensor([
    [2, 301, 302, 3, 0],  # "<s> Hallo Welt </s> <pad>"
    [2, 401, 402, 403, 3],  # "<s> Längerer Satz </s>"
])

# Model automatically creates padding masks
logits = model(src, tgt)
```

## Example 2: Complete Training Pipeline

### Setup

```python
import torch
from src import Transformer
from src.scheduler import TransformerScheduler
from src.label_smoothing import LabelSmoothingLoss
from src.trainer import Trainer, TrainerConfig
from src.tokenizer import SimpleTokenizer
from src.data import TranslationDataset, create_dynamic_dataloader

# Sample training data
src_sentences = [
    "The cat sat on the mat",
    "Hello world",
    "How are you today",
    "I love machine learning",
    "The weather is nice",
    # ... more sentences ...
]

tgt_sentences = [
    "Die Katze saß auf der Matte",
    "Hallo Welt",
    "Wie geht es dir heute",
    "Ich liebe maschinelles Lernen",
    "Das Wetter ist schön",
    # ... more sentences ...
]
```

### Create Tokenizer

```python
# Build tokenizer from training data
tokenizer = SimpleTokenizer()
tokenizer.build_vocab(src_sentences + tgt_sentences)
print(f"Vocabulary size: {tokenizer.vocab_size}")
```

### Create Dataset and DataLoader

```python
# Create dataset
dataset = TranslationDataset(
    src_data=src_sentences,
    tgt_data=tgt_sentences,
    src_tokenizer=tokenizer,
    tgt_tokenizer=tokenizer,
    add_bos=True,
    add_eos=True,
)

# Create dataloader with dynamic batching
train_loader = create_dynamic_dataloader(
    dataset=dataset,
    max_tokens=1024,  # Max tokens per batch
    shuffle=True,
)
```

### Create Model and Optimizer

```python
# Create model
model = Transformer(
    src_vocab_size=tokenizer.vocab_size,
    tgt_vocab_size=tokenizer.vocab_size,
    d_model=256,          # Smaller for demo
    n_heads=4,
    n_encoder_layers=3,
    n_decoder_layers=3,
    d_ff=512,
    dropout=0.1,
)

# Create optimizer with paper's settings
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1.0,             # Will be overridden by scheduler
    betas=(0.9, 0.98),
    eps=1e-9,
)

# Create learning rate scheduler
scheduler = TransformerScheduler(
    optimizer,
    d_model=256,
    warmup_steps=100,  # Fewer for demo
)

# Create loss function
criterion = LabelSmoothingLoss(
    smoothing=0.1,
    padding_idx=0,
)
```

### Configure and Run Training

```python
# Training configuration
config = TrainerConfig(
    max_steps=1000,
    gradient_accumulation_steps=1,
    max_grad_norm=1.0,
    log_steps=50,
    eval_steps=100,
    save_steps=500,
    save_dir="checkpoints",
    device="cuda" if torch.cuda.is_available() else "cpu",
    padding_idx=0,
)

# Create trainer
trainer = Trainer(
    model=model,
    optimizer=optimizer,
    criterion=criterion,
    config=config,
    scheduler=scheduler,
    train_loader=train_loader,
)

# Add logging callback
def log_callback(metrics):
    step = metrics.get("step", 0)
    loss = metrics.get("train_loss", 0)
    lr = metrics.get("learning_rate", 0)
    print(f"Step {step}: loss={loss:.4f}, lr={lr:.6f}")

trainer.set_log_callback(log_callback)

# Train!
history = trainer.train()
```

### Output Example

```
Step 50: loss=6.2341, lr=0.000220
Step 100: loss=5.8765, lr=0.000440
Step 150: loss=5.4321, lr=0.000660
Step 200: loss=5.0123, lr=0.000880
...
```

## Example 3: Sequence Generation

### Greedy Decoding

```python
# Assuming trained model and tokenizer

# Encode source sentence
src_text = "The cat sat on the mat"
src_ids = tokenizer.encode(src_text, add_bos=True, add_eos=True)
src = torch.tensor([src_ids])  # Add batch dimension

# Generate translation
model.eval()
with torch.no_grad():
    generated = model.generate(
        src=src,
        max_len=50,
        start_token=tokenizer.bos_id,
        end_token=tokenizer.eos_id,
    )

# Decode output
output_ids = generated[0].tolist()
output_text = tokenizer.decode(output_ids)
print(f"Translation: {output_text}")
```

### Step-by-Step Generation

```python
def generate_step_by_step(model, src, tokenizer, max_len=50):
    """Generate with detailed output at each step."""
    model.eval()
    device = next(model.parameters()).device

    # Encode source
    src = src.to(device)
    memory = model.encode(src)

    # Start with BOS token
    tgt = torch.tensor([[tokenizer.bos_id]], device=device)

    print("Generation steps:")
    for step in range(max_len):
        # Decode
        tgt_mask = model._create_tgt_mask(tgt)
        decoder_output = model.decode(tgt, memory, tgt_mask)

        # Get logits for last position
        logits = model.output_projection(decoder_output[:, -1, :])

        # Get probabilities
        probs = torch.softmax(logits, dim=-1)

        # Greedy: pick most likely token
        next_token = probs.argmax(dim=-1, keepdim=True)

        # Get top 3 predictions
        top_probs, top_ids = probs.topk(3, dim=-1)

        print(f"Step {step + 1}:")
        print(f"  Top predictions: ", end="")
        for i in range(3):
            token = tokenizer.decode([top_ids[0, i].item()])
            prob = top_probs[0, i].item()
            print(f"'{token}' ({prob:.2%})", end=" ")
        print()

        # Append to sequence
        tgt = torch.cat([tgt, next_token], dim=1)

        # Check for EOS
        if next_token.item() == tokenizer.eos_id:
            print("  [EOS reached]")
            break

    return tgt

# Usage
src = torch.tensor([[2, 101, 102, 103, 3]])  # Example source
generated = generate_step_by_step(model, src, tokenizer)
```

## Example 4: Custom Configuration

### Small Model (for Testing)

```python
model = Transformer(
    src_vocab_size=5000,
    tgt_vocab_size=5000,
    d_model=128,
    n_heads=4,
    n_encoder_layers=2,
    n_decoder_layers=2,
    d_ff=256,
    dropout=0.1,
)
# ~2M parameters
```

### Base Model (Paper's Base)

```python
model = Transformer(
    src_vocab_size=37000,
    tgt_vocab_size=37000,
    d_model=512,
    n_heads=8,
    n_encoder_layers=6,
    n_decoder_layers=6,
    d_ff=2048,
    dropout=0.1,
)
# ~65M parameters (with 37K vocab)
```

### Big Model (Paper's Big)

```python
model = Transformer(
    src_vocab_size=37000,
    tgt_vocab_size=37000,
    d_model=1024,
    n_heads=16,
    n_encoder_layers=6,
    n_decoder_layers=6,
    d_ff=4096,
    dropout=0.3,  # Higher dropout for big model
)
# ~213M parameters (with 37K vocab)
```

### Shared Embeddings

```python
# Share embeddings between encoder and decoder
# (common when source and target use same vocabulary)
model = Transformer(
    src_vocab_size=37000,
    tgt_vocab_size=37000,
    d_model=512,
    n_heads=8,
    n_encoder_layers=6,
    n_decoder_layers=6,
    d_ff=2048,
    share_embeddings=True,  # Share embeddings
)
```

## Example 5: Inference with Trained Model

### Load Checkpoint

```python
# Load saved checkpoint
checkpoint = torch.load("checkpoints/model_step_10000.pt")

# Create model with same architecture
model = Transformer(
    src_vocab_size=32000,
    tgt_vocab_size=32000,
    d_model=512,
    n_heads=8,
    n_encoder_layers=6,
    n_decoder_layers=6,
)

# Load weights
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
```

### Batch Translation

```python
def translate_batch(model, tokenizer, sentences, max_len=100):
    """Translate a batch of sentences."""
    model.eval()
    device = next(model.parameters()).device

    # Tokenize all sentences
    src_batch = []
    for sent in sentences:
        ids = tokenizer.encode(sent, add_bos=True, add_eos=True)
        src_batch.append(ids)

    # Pad to same length
    max_src_len = max(len(s) for s in src_batch)
    src_padded = torch.zeros(len(sentences), max_src_len, dtype=torch.long)
    for i, ids in enumerate(src_batch):
        src_padded[i, :len(ids)] = torch.tensor(ids)

    src_padded = src_padded.to(device)

    # Generate translations
    with torch.no_grad():
        generated = model.generate(
            src=src_padded,
            max_len=max_len,
            start_token=tokenizer.bos_id,
            end_token=tokenizer.eos_id,
        )

    # Decode all outputs
    translations = []
    for i in range(len(sentences)):
        output_ids = generated[i].tolist()
        text = tokenizer.decode(output_ids, skip_special_tokens=True)
        translations.append(text)

    return translations

# Usage
sentences = [
    "Hello, how are you?",
    "The weather is beautiful today.",
    "I love programming.",
]
translations = translate_batch(model, tokenizer, sentences)
for src, tgt in zip(sentences, translations):
    print(f"{src} → {tgt}")
```

## Example 6: Using Individual Components

### Just the Encoder

```python
from src.encoder import Encoder

encoder = Encoder(
    n_layers=6,
    d_model=512,
    n_heads=8,
    d_ff=2048,
)

# Input: embedded + positional encoded source
x = torch.randn(2, 20, 512)  # (batch, seq_len, d_model)
output = encoder(x)
# output.shape: (2, 20, 512)
```

### Just the Decoder

```python
from src.decoder import Decoder

decoder = Decoder(
    n_layers=6,
    d_model=512,
    n_heads=8,
    d_ff=2048,
)

# Inputs
tgt = torch.randn(2, 15, 512)      # Target embeddings
memory = torch.randn(2, 20, 512)   # Encoder output

output = decoder(tgt, memory)
# output.shape: (2, 15, 512)
```

### Just Multi-Head Attention

```python
from src.attention import MultiHeadAttention

mha = MultiHeadAttention(d_model=512, n_heads=8, dropout=0.1)

# Self-attention
x = torch.randn(2, 10, 512)
output, weights = mha(x, x, x)
# output.shape: (2, 10, 512)
# weights.shape: (2, 8, 10, 10)

# Cross-attention
q = torch.randn(2, 10, 512)  # Query from decoder
kv = torch.randn(2, 20, 512)  # Key/Value from encoder
output, weights = mha(q, kv, kv)
# output.shape: (2, 10, 512)
# weights.shape: (2, 8, 10, 20)
```

## Example 7: Visualization

### Attention Weights

```python
import matplotlib.pyplot as plt

def visualize_attention(model, src, tgt, layer=0, head=0):
    """Visualize attention weights for a specific layer and head."""
    model.eval()

    # Hook to capture attention weights
    attention_weights = {}

    def hook_fn(name):
        def hook(module, input, output):
            attention_weights[name] = output[1]  # (output, weights)
        return hook

    # Register hooks
    handles = []
    for i, layer in enumerate(model.decoder.layers):
        h = layer.cross_attention.register_forward_hook(
            hook_fn(f"decoder_layer_{i}_cross_attn")
        )
        handles.append(h)

    # Forward pass
    with torch.no_grad():
        _ = model(src, tgt)

    # Remove hooks
    for h in handles:
        h.remove()

    # Get weights for specified layer
    key = f"decoder_layer_{layer}_cross_attn"
    weights = attention_weights[key][0, head].cpu().numpy()
    # weights.shape: (tgt_len, src_len)

    # Plot
    plt.figure(figsize=(10, 8))
    plt.imshow(weights, cmap="viridis")
    plt.colorbar()
    plt.xlabel("Source Position")
    plt.ylabel("Target Position")
    plt.title(f"Cross-Attention Weights (Layer {layer}, Head {head})")
    plt.show()

    return weights
```

### Learning Rate Schedule

```python
from src.scheduler import get_lr_at_step
import matplotlib.pyplot as plt

steps = range(1, 50001)
lrs = [get_lr_at_step(s, d_model=512, warmup_steps=4000) for s in steps]

plt.figure(figsize=(12, 4))
plt.plot(steps, lrs)
plt.xlabel("Training Step")
plt.ylabel("Learning Rate")
plt.title("Transformer Learning Rate Schedule")
plt.axvline(x=4000, color="r", linestyle="--", label="Warmup ends")
plt.legend()
plt.show()
```

## Summary

| Example | Purpose | Key Components |
|---------|---------|----------------|
| Quick Start | Basic model usage | `Transformer`, forward pass |
| Training | Full pipeline | `Trainer`, `Scheduler`, `LabelSmoothing` |
| Generation | Autoregressive decoding | `model.generate()` |
| Custom Config | Architecture variants | Model hyperparameters |
| Checkpoint | Save/load models | `save_checkpoint()`, `load_checkpoint()` |
| Components | Using parts individually | `Encoder`, `Decoder`, `MultiHeadAttention` |
| Visualization | Understanding behavior | Attention weights, LR schedule |

---

*This concludes the Transformer book. For more details, refer to:*
- *Original paper: "Attention Is All You Need" (Vaswani et al., 2017)*
- *Source code in the `src/` directory*
- *Tests in the `tests/` directory*
