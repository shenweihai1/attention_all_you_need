# Chapter 4: Training Infrastructure

## Overview

Training a Transformer well requires careful attention to:
1. **Weight Initialization** - Starting from good initial values
2. **Learning Rate Scheduling** - The warmup + decay schedule from the paper
3. **Label Smoothing** - Regularization technique that improves generalization
4. **Gradient Accumulation** - Simulating larger batch sizes
5. **The Training Loop** - Putting it all together

This chapter covers the training infrastructure in this codebase.

## Weight Initialization

### Why Initialization Matters

Poor initialization can cause:
- **Vanishing gradients**: Weights too small → signals die out
- **Exploding gradients**: Weights too large → signals blow up
- **Slow convergence**: Starting far from optimum

### Initialization Strategy

From `src/init.py:15-46`:

```python
def init_transformer_weights(module: nn.Module, d_model: int = 512) -> None:
    """
    Initialize weights for Transformer modules.

    - Linear layers: Xavier uniform initialization
    - Embeddings: Normal distribution with std = d_model^(-0.5)
    - LayerNorm: weight=1, bias=0
    - Biases: zeros
    """
    if isinstance(module, nn.Linear):
        # Xavier uniform: Var(W) = 2 / (fan_in + fan_out)
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

    elif isinstance(module, nn.Embedding):
        # Normal with std = 1/sqrt(d_model) ≈ 0.044 for d_model=512
        nn.init.normal_(module.weight, mean=0.0, std=d_model ** -0.5)
        if module.padding_idx is not None:
            nn.init.zeros_(module.weight[module.padding_idx])

    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
```

### Why Xavier Initialization?

Xavier (Glorot) uniform initialization maintains variance across layers:

```
For a linear layer with fan_in inputs and fan_out outputs:
W ~ Uniform(-a, a) where a = sqrt(6 / (fan_in + fan_out))

This ensures:
- Var(input) ≈ Var(output)
- Gradients don't vanish or explode
```

### Usage

```python
from src import Transformer
from src.init import init_transformer_weights, count_parameters

model = Transformer(
    src_vocab_size=10000,
    tgt_vocab_size=10000,
    d_model=512,
    n_heads=8,
    n_encoder_layers=6,
    n_decoder_layers=6,
)

# Initialize weights (automatically done in Transformer.__init__)
model.apply(lambda m: init_transformer_weights(m, d_model=512))

# Check parameter count
print(f"Parameters: {count_parameters(model):,}")  # ~44M for base model
```

## Learning Rate Scheduling

### The Transformer Schedule

The paper uses a specific learning rate schedule:

```
lrate = d_model^(-0.5) × min(step^(-0.5), step × warmup_steps^(-1.5))
```

This has two phases:

1. **Warmup** (steps 1 to warmup_steps): Linear increase
   - LR increases from ~0 to peak value
   - Allows model to stabilize before large updates

2. **Decay** (steps > warmup_steps): Inverse square root decrease
   - LR decreases proportionally to 1/√step
   - Gradual reduction as model converges

### Visual: Learning Rate Schedule

```
Learning Rate
     │
     │           Peak at warmup_steps
     │              ╱╲
     │            ╱    ╲
     │          ╱        ╲
     │        ╱            ╲  (inverse sqrt decay)
     │      ╱                ╲
     │    ╱                    ╲___
     │  ╱ (linear warmup)           ╲___
     │╱                                  ╲___
     └─────────────────────────────────────────────
         1000     4000      10000     50000    steps
                    ↑
              warmup_steps
```

### Implementation

From `src/scheduler.py:22-85`:

```python
class TransformerScheduler(_LRScheduler):
    """
    Learning rate scheduler as described in the paper.

    lrate = d_model^(-0.5) × min(step^(-0.5), step × warmup^(-1.5))
    """

    def __init__(
        self,
        optimizer: Optimizer,
        d_model: int = 512,
        warmup_steps: int = 4000,
        scale: float = 1.0,
    ):
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.scale = scale
        self._d_model_factor = d_model ** -0.5
        super().__init__(optimizer)

    def get_lr(self):
        step_num = max(1, self._step_count)

        lr = self._d_model_factor * min(
            step_num ** -0.5,
            step_num * (self.warmup_steps ** -1.5)
        )

        return [lr * self.scale for _ in self.base_lrs]
```

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| d_model | 512 | Model dimension (affects LR scale) |
| warmup_steps | 4000 | Steps for linear warmup |
| scale | 1.0 | Optional multiplier |

### Learning Rate Values

With d_model=512 and warmup_steps=4000:

| Step | Learning Rate |
|------|---------------|
| 1 | 4.4 × 10⁻⁵ |
| 1000 | 1.4 × 10⁻³ |
| 4000 | 1.4 × 10⁻³ (peak) |
| 10000 | 8.8 × 10⁻⁴ |
| 100000 | 2.8 × 10⁻⁴ |

### Usage

```python
from src.scheduler import TransformerScheduler, get_lr_at_step

# Create optimizer with dummy LR (scheduler will override)
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1.0,  # Will be overridden
    betas=(0.9, 0.98),  # Paper's values
    eps=1e-9
)

# Create scheduler
scheduler = TransformerScheduler(
    optimizer,
    d_model=512,
    warmup_steps=4000
)

# Training loop
for step in range(100000):
    # ... forward, backward ...
    optimizer.step()
    scheduler.step()  # Update learning rate

# Check LR at specific step
lr = get_lr_at_step(4000, d_model=512, warmup_steps=4000)
print(f"Peak LR: {lr:.6f}")  # ~0.001399
```

## Label Smoothing

### What is Label Smoothing?

Standard cross-entropy uses **hard targets**:
- True class: probability = 1
- Other classes: probability = 0

Label smoothing uses **soft targets**:
- True class: probability = 1 - ε + ε/K
- Other classes: probability = ε/K

Where ε = 0.1 (smoothing) and K = vocabulary size.

### Why Label Smoothing?

1. **Prevents overconfidence**: Model doesn't learn to output extreme probabilities
2. **Regularization**: Acts as a form of regularization
3. **Better generalization**: Improves BLEU scores in practice

### Visual: Label Smoothing

```
Hard targets (no smoothing):        Soft targets (ε = 0.1):
┌─────────────────────┐            ┌─────────────────────┐
│ 1.0 ████████████████│            │ 0.9 ███████████████ │  ← true class
│ 0.0                 │            │ 0.001 ░             │
│ 0.0                 │            │ 0.001 ░             │
│ 0.0                 │            │ 0.001 ░             │
│ ...                 │            │ ...                 │
└─────────────────────┘            └─────────────────────┘

Model learns to output:            Model learns to output:
[0.99, 0.001, 0.001, ...]         [0.85, 0.05, 0.05, ...]
(overconfident)                    (calibrated)
```

### Implementation

From `src/label_smoothing.py:21-108`:

```python
class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing with KL divergence.

    loss = (1 - ε) × CE(logits, target) + ε × mean(-log_softmax(logits))
    """

    def __init__(
        self,
        smoothing: float = 0.1,
        padding_idx: Optional[int] = None,
        reduction: str = "mean",
    ):
        super().__init__()
        self.smoothing = smoothing
        self.padding_idx = padding_idx
        self.reduction = reduction
        self.confidence = 1.0 - smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        vocab_size = logits.size(-1)

        # Create smoothed target distribution
        smooth_target = torch.full_like(logits, self.smoothing / vocab_size)

        # True class gets higher probability
        smooth_target.scatter_(
            dim=-1,
            index=target.unsqueeze(-1),
            value=self.confidence + self.smoothing / vocab_size
        )

        # Zero out padding positions
        if self.padding_idx is not None:
            padding_mask = target.eq(self.padding_idx)
            smooth_target[padding_mask] = 0.0

        # KL divergence loss
        log_probs = F.log_softmax(logits, dim=-1)
        loss = -(smooth_target * log_probs).sum(dim=-1)

        # Reduction
        if self.reduction == "mean":
            if self.padding_idx is not None:
                non_padding = target.ne(self.padding_idx)
                return loss.sum() / non_padding.sum().clamp(min=1)
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss
```

### Usage

```python
from src.label_smoothing import LabelSmoothingLoss

# Create loss function
criterion = LabelSmoothingLoss(
    smoothing=0.1,      # ε = 0.1 from paper
    padding_idx=0,      # Ignore padding tokens
    reduction="mean"
)

# In training
logits = model(src, tgt_input)  # (batch, seq_len, vocab_size)
loss = criterion(
    logits.view(-1, vocab_size),  # Flatten: (batch*seq_len, vocab_size)
    tgt_output.view(-1)           # Flatten: (batch*seq_len,)
)
loss.backward()
```

## Gradient Accumulation

### Why Gradient Accumulation?

The paper uses batch size of 25,000 tokens per batch. This may not fit in GPU memory. Gradient accumulation simulates larger batches:

```
Without accumulation:
  Batch 1 (4096 tokens) → backward → optimizer step
  Batch 2 (4096 tokens) → backward → optimizer step
  ...

With accumulation (steps=6):
  Batch 1 (4096 tokens) → backward (accumulate)
  Batch 2 (4096 tokens) → backward (accumulate)
  Batch 3 (4096 tokens) → backward (accumulate)
  Batch 4 (4096 tokens) → backward (accumulate)
  Batch 5 (4096 tokens) → backward (accumulate)
  Batch 6 (4096 tokens) → backward → optimizer step
  → Effective batch: 24,576 tokens
```

### Implementation

From `src/trainer.py:184-256`:

```python
def train_step(self, src, tgt, src_mask=None, tgt_mask=None):
    """Perform a single forward/backward pass."""
    self.model.train()

    # Move to device
    src = src.to(self.device)
    tgt = tgt.to(self.device)

    # Teacher forcing: input is tgt[:-1], target is tgt[1:]
    tgt_input = tgt[:, :-1]
    tgt_output = tgt[:, 1:]

    # Forward
    logits = self.model(src, tgt_input, src_mask, tgt_mask)

    # Loss
    loss = self.criterion(
        logits.view(-1, logits.size(-1)),
        tgt_output.view(-1)
    )

    # Scale loss for accumulation
    scaled_loss = loss / self.config.gradient_accumulation_steps

    # Backward (gradients accumulate)
    scaled_loss.backward()

    return loss.item(), n_tokens

def optimizer_step(self):
    """Perform optimizer update after accumulation."""
    # Gradient clipping
    if self.config.max_grad_norm is not None:
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config.max_grad_norm
        )

    # Update weights
    self.optimizer.step()
    self.optimizer.zero_grad()

    # Update learning rate
    if self.scheduler:
        self.scheduler.step()

    self.global_step += 1
```

### Key Points

1. **Loss scaling**: Divide by accumulation steps so final gradient has correct scale
2. **Gradient clipping**: Prevents exploding gradients (max_norm=1.0)
3. **Optimizer step**: Only after accumulating specified number of batches

## The Complete Training Loop

### Trainer Configuration

From `src/trainer.py:67-93`:

```python
@dataclass
class TrainerConfig:
    # Training
    max_steps: int = 100000
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0

    # Validation
    eval_steps: int = 1000

    # Logging and checkpoints
    log_steps: int = 100
    save_steps: int = 5000
    save_dir: Optional[str] = None

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    padding_idx: int = 0
```

### Training Loop Overview

```python
# Main training loop (simplified)
while global_step < max_steps:
    batch = next(data_iterator)

    # Forward + backward (accumulates gradients)
    loss, n_tokens = train_step(batch)

    accumulation_count += 1

    if accumulation_count >= gradient_accumulation_steps:
        # Clip gradients and update weights
        optimizer_step()
        accumulation_count = 0

        # Logging
        if global_step % log_steps == 0:
            log_metrics()

        # Evaluation
        if global_step % eval_steps == 0:
            evaluate()

        # Save checkpoint
        if global_step % save_steps == 0:
            save_checkpoint()
```

### Complete Training Example

```python
import torch
from src import Transformer
from src.scheduler import TransformerScheduler
from src.label_smoothing import LabelSmoothingLoss
from src.trainer import Trainer, TrainerConfig

# 1. Create model
model = Transformer(
    src_vocab_size=32000,
    tgt_vocab_size=32000,
    d_model=512,
    n_heads=8,
    n_encoder_layers=6,
    n_decoder_layers=6,
    d_ff=2048,
    dropout=0.1,
)

# 2. Create optimizer (Adam with paper's parameters)
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1.0,  # Scheduler will override
    betas=(0.9, 0.98),
    eps=1e-9
)

# 3. Create scheduler
scheduler = TransformerScheduler(
    optimizer,
    d_model=512,
    warmup_steps=4000
)

# 4. Create loss function
criterion = LabelSmoothingLoss(
    smoothing=0.1,
    padding_idx=0
)

# 5. Configure trainer
config = TrainerConfig(
    max_steps=100000,
    gradient_accumulation_steps=8,  # Effective batch = 8 × actual batch
    max_grad_norm=1.0,
    eval_steps=1000,
    log_steps=100,
    save_steps=5000,
    save_dir="checkpoints",
)

# 6. Create trainer
trainer = Trainer(
    model=model,
    optimizer=optimizer,
    criterion=criterion,
    config=config,
    scheduler=scheduler,
    train_loader=train_loader,
    eval_loader=eval_loader,
)

# 7. Optional: Add logging callback
def log_fn(metrics):
    print(f"Step {metrics['step']}: loss={metrics.get('train_loss', 0):.4f}")

trainer.set_log_callback(log_fn)

# 8. Train!
history = trainer.train()
```

### Checkpoint Management

```python
# Save checkpoint
trainer.save_checkpoint("checkpoints/model_step_10000.pt")

# Load checkpoint
trainer.load_checkpoint("checkpoints/model_step_10000.pt")

# Resume training
trainer.train()  # Continues from loaded step
```

### Checkpoint Contents

```python
checkpoint = {
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "scheduler_state_dict": scheduler.state_dict(),
    "global_step": 10000,
    "epoch": 5,
    "best_eval_loss": 2.34,
    "config": trainer_config,
}
```

## Paper's Training Details

From "Attention Is All You Need":

| Hyperparameter | Base Model | Big Model |
|----------------|------------|-----------|
| d_model | 512 | 1024 |
| d_ff | 2048 | 4096 |
| n_heads | 8 | 16 |
| n_layers | 6 | 6 |
| d_k = d_v | 64 | 64 |
| Dropout | 0.1 | 0.3 |
| Label smoothing | 0.1 | 0.1 |
| Warmup steps | 4000 | 4000 |
| Batch size | ~25K tokens | ~25K tokens |
| Training steps | 100K | 300K |
| Hardware | 8 P100 GPUs | 8 P100 GPUs |
| Training time | 12 hours | 3.5 days |

## Summary

| Component | Purpose | Key Parameters |
|-----------|---------|----------------|
| Weight Init | Stable starting point | Xavier uniform, std=1/√d_model |
| LR Scheduler | Warmup + decay | warmup=4000, d_model=512 |
| Label Smoothing | Regularization | ε=0.1 |
| Gradient Accumulation | Larger effective batch | accumulation_steps |
| Gradient Clipping | Prevent explosion | max_norm=1.0 |

---

*Next: [Chapter 5: Data Processing](05_data.md)*
