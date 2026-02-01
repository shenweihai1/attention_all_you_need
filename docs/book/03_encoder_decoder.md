# Chapter 3: Encoder and Decoder

## Overview

The Transformer uses an **encoder-decoder architecture**:
- **Encoder**: Processes the source sequence (e.g., English sentence)
- **Decoder**: Generates the target sequence (e.g., French translation)

This chapter covers:
1. Positional Encoding - How positions are represented
2. Position-wise Feed-Forward Networks - The FFN sublayer
3. Encoder Layer and Stack - Processing source sequences
4. Decoder Layer and Stack - Generating target sequences
5. Residual Connections and Layer Normalization

## Positional Encoding

### Why Positional Encoding?

Unlike RNNs which process tokens sequentially, attention is **permutation-invariant** - it doesn't know the order of tokens. We must explicitly add position information.

The paper uses **sinusoidal positional encoding**:

```
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

Where:
- `pos` = position in the sequence (0, 1, 2, ...)
- `i` = dimension index (0, 1, 2, ..., d_model/2)
- `d_model` = 512 (embedding dimension)

### Why Sinusoidal?

1. **Unique encoding for each position**: Different positions have different patterns
2. **Relative positions can be learned**: PE(pos+k) can be represented as a linear function of PE(pos)
3. **Generalizes to longer sequences**: Works for sequences longer than those seen during training

### Implementation

From `src/positional_encoding.py:59-95`:

```python
def _create_positional_encoding(self, max_seq_len: int, d_model: int) -> torch.Tensor:
    # Create position indices: [0, 1, 2, ..., max_seq_len-1]
    position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)

    # Compute division term: 10000^(2i/d_model)
    # Using exp(2i * -log(10000) / d_model) for numerical stability
    div_term = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float)
        * (-math.log(10000.0) / d_model)
    )

    # Create encoding matrix
    pe = torch.zeros(max_seq_len, d_model)

    # Even dimensions get sin
    pe[:, 0::2] = torch.sin(position * div_term)

    # Odd dimensions get cos
    pe[:, 1::2] = torch.cos(position * div_term)

    return pe
```

### Visual: Positional Encoding Pattern

```
Position 0:  [sin(0), cos(0), sin(0), cos(0), ...]
             = [0.0,   1.0,   0.0,   1.0,   ...]

Position 1:  [sin(1/1), cos(1/1), sin(1/w), cos(1/w), ...]
             where w = 10000^(2/512) ≈ 1.036

Position 2:  [sin(2/1), cos(2/1), sin(2/w), cos(2/w), ...]

The wavelengths form a geometric progression from 2π to 10000·2π
```

### Usage

```python
from src.positional_encoding import PositionalEncoding

pe = PositionalEncoding(d_model=512, max_seq_len=5000, dropout=0.1)

# Add positional encoding to embeddings
embeddings = torch.randn(2, 100, 512)  # (batch, seq_len, d_model)
encoded = pe(embeddings)  # Same shape, with positions added
```

## Position-wise Feed-Forward Network

### The Formula

After attention, each layer applies a feed-forward network **independently to each position**:

```
FFN(x) = max(0, xW₁ + b₁)W₂ + b₂
```

This is equivalent to two 1x1 convolutions:
- First layer: `d_model → d_ff` (512 → 2048)
- ReLU activation
- Second layer: `d_ff → d_model` (2048 → 512)

### Why "Position-wise"?

The same FFN is applied at each position, but each position is processed independently:

```
Input:  [x₁, x₂, x₃, x₄]  (sequence of positions)
         ↓   ↓   ↓   ↓
FFN:   [FFN(x₁), FFN(x₂), FFN(x₃), FFN(x₄)]

Same weights, no cross-position interaction
```

### Implementation

From `src/feedforward.py:35-76`:

```python
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int = 512, d_ff: int = 2048, dropout: float = 0.0):
        super().__init__()

        # d_model → d_ff (512 → 2048)
        self.linear1 = nn.Linear(d_model, d_ff)

        # d_ff → d_model (2048 → 512)
        self.linear2 = nn.Linear(d_ff, d_model)

        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # FFN(x) = max(0, xW1 + b1)W2 + b2
        hidden = F.relu(self.linear1(x))  # (batch, seq, d_ff)

        if self.dropout is not None:
            hidden = self.dropout(hidden)

        output = self.linear2(hidden)     # (batch, seq, d_model)
        return output
```

### Why d_ff = 4 × d_model?

The inner dimension (2048) is 4× the model dimension (512). This gives the network more capacity to learn complex functions while keeping the input/output dimensions fixed.

## Encoder

### Encoder Architecture

The encoder processes the source sequence and produces representations:

```
Input Embeddings + Positional Encoding
            ↓
    ┌───────────────────┐
    │   Encoder Layer   │ ×N (N=6)
    │  ┌─────────────┐  │
    │  │ Self-Attn   │──┤
    │  └─────────────┘  │
    │  ┌─────────────┐  │
    │  │    FFN      │──┤
    │  └─────────────┘  │
    └───────────────────┘
            ↓
    Final Layer Norm
            ↓
    Encoder Output (Memory)
```

### Encoder Layer

Each encoder layer has two sublayers:
1. Multi-head self-attention
2. Position-wise feed-forward network

From `src/encoder.py:17-109`:

```python
class EncoderLayer(nn.Module):
    def __init__(self, d_model=512, n_heads=8, d_ff=2048, dropout=0.1):
        super().__init__()

        # Sublayer 1: Multi-head self-attention
        self.self_attention = MultiHeadAttention(
            d_model=d_model, n_heads=n_heads, dropout=dropout
        )

        # Sublayer 2: Position-wise FFN
        self.feed_forward = PositionwiseFeedForward(
            d_model=d_model, d_ff=d_ff, dropout=dropout
        )

        # Layer normalization for each sublayer
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Dropout for residual connections
        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)

    def forward(self, x, src_mask=None):
        # Sublayer 1: Self-attention + residual + norm
        attn_output, _ = self.self_attention(x, x, x, mask=src_mask)
        x = self.norm1(x + self.dropout1(attn_output))

        # Sublayer 2: FFN + residual + norm
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_output))

        return x
```

### Residual Connection Pattern

Each sublayer uses the pattern: `LayerNorm(x + Sublayer(x))`

```
        x (input)
        │
        ├──────────────────────┐
        ↓                      │
   ┌─────────┐                 │ (residual)
   │ Sublayer│                 │
   └────┬────┘                 │
        │                      │
        ↓                      │
    Dropout                    │
        │                      │
        ├──────────────────────┘
        ↓
   ┌─────────┐
   │LayerNorm│
   └────┬────┘
        ↓
     Output
```

### Encoder Stack

The full encoder stacks N=6 identical layers:

From `src/encoder.py:112-188`:

```python
class Encoder(nn.Module):
    def __init__(self, n_layers=6, d_model=512, n_heads=8, d_ff=2048, dropout=0.1):
        super().__init__()

        # Stack of N encoder layers
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # Final layer normalization
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, src_mask=None):
        # Pass through each layer sequentially
        for layer in self.layers:
            x = layer(x, src_mask=src_mask)

        # Final layer norm
        x = self.norm(x)
        return x
```

### Usage

```python
from src.encoder import Encoder

encoder = Encoder(n_layers=6, d_model=512, n_heads=8, d_ff=2048)

# Input: embedded + positional encoded source
x = torch.randn(2, 20, 512)  # (batch=2, seq_len=20, d_model=512)

# Encode
memory = encoder(x)  # (2, 20, 512)
```

## Decoder

### Decoder Architecture

The decoder generates target tokens one at a time, attending to both previous outputs and encoder memory:

```
Target Embeddings + Positional Encoding
            ↓
    ┌───────────────────────┐
    │   Decoder Layer       │ ×N (N=6)
    │  ┌─────────────────┐  │
    │  │ Masked Self-Attn│──┤ (causal mask)
    │  └─────────────────┘  │
    │  ┌─────────────────┐  │
    │  │ Cross-Attention │──┤ (attend to encoder)
    │  └─────────────────┘  │
    │  ┌─────────────────┐  │
    │  │      FFN        │──┤
    │  └─────────────────┘  │
    └───────────────────────┘
            ↓
    Final Layer Norm
            ↓
    Decoder Output
```

### Decoder Layer

Each decoder layer has **three** sublayers:
1. **Masked multi-head self-attention** (causal - can't see future)
2. **Multi-head cross-attention** (attends to encoder output)
3. **Position-wise feed-forward network**

From `src/decoder.py:17-129`:

```python
class DecoderLayer(nn.Module):
    def __init__(self, d_model=512, n_heads=8, d_ff=2048, dropout=0.1):
        super().__init__()

        # Sublayer 1: Masked self-attention
        self.self_attention = MultiHeadAttention(
            d_model=d_model, n_heads=n_heads, dropout=dropout
        )

        # Sublayer 2: Cross-attention (encoder-decoder attention)
        self.cross_attention = MultiHeadAttention(
            d_model=d_model, n_heads=n_heads, dropout=dropout
        )

        # Sublayer 3: FFN
        self.feed_forward = PositionwiseFeedForward(
            d_model=d_model, d_ff=d_ff, dropout=dropout
        )

        # Layer norms for each sublayer
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        # Dropout for residual connections
        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)
        self.dropout3 = nn.Dropout(p=dropout)

    def forward(self, x, memory, tgt_mask=None, memory_mask=None):
        # Sublayer 1: Masked self-attention
        # Q, K, V all from decoder (self-attention)
        self_attn_output, _ = self.self_attention(x, x, x, mask=tgt_mask)
        x = self.norm1(x + self.dropout1(self_attn_output))

        # Sublayer 2: Cross-attention
        # Q from decoder, K and V from encoder memory
        cross_attn_output, _ = self.cross_attention(x, memory, memory, mask=memory_mask)
        x = self.norm2(x + self.dropout2(cross_attn_output))

        # Sublayer 3: FFN
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout3(ff_output))

        return x
```

### Understanding Cross-Attention

In cross-attention, the decoder uses its own representation as the **query** to look up relevant information from the encoder:

```
Decoder:  "What French word should I produce?"  (Query)
          ↓
Encoder:  "Here's what the English sentence means"  (Key, Value)
          ↓
Output:   Weighted combination of encoder representations
```

Example: Translating "The cat sat"
```
When generating "Le" (French "The"):
  - Query: decoder state for position 0
  - Keys/Values: encoder representations of ["The", "cat", "sat"]
  - Result: attention focuses on "The" → outputs French equivalent
```

### Decoder Stack

From `src/decoder.py:132-223`:

```python
class Decoder(nn.Module):
    def __init__(self, n_layers=6, d_model=512, n_heads=8, d_ff=2048, dropout=0.1):
        super().__init__()

        # Stack of N decoder layers
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # Final layer normalization
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, memory, tgt_mask=None, memory_mask=None):
        for layer in self.layers:
            x = layer(x, memory, tgt_mask=tgt_mask, memory_mask=memory_mask)

        x = self.norm(x)
        return x
```

### Usage

```python
from src.decoder import Decoder

decoder = Decoder(n_layers=6, d_model=512, n_heads=8, d_ff=2048)

# Inputs
tgt = torch.randn(2, 15, 512)    # Target embeddings (batch=2, tgt_len=15)
memory = torch.randn(2, 20, 512) # Encoder output (batch=2, src_len=20)

# Decode
output = decoder(tgt, memory)  # (2, 15, 512)
```

## Complete Data Flow

Let's trace a complete forward pass for translation:

### Step 1: Source Encoding
```python
# Source: "Hello world" → token IDs → embeddings
src_tokens = [101, 7592, 2088, 102]  # Example token IDs
src_embed = embedding(src_tokens) * sqrt(d_model)  # (1, 4, 512)
src_encoded = positional_encoding(src_embed)       # (1, 4, 512)

# Encode
memory = encoder(src_encoded, src_mask)            # (1, 4, 512)
```

### Step 2: Target Decoding (Training)
```python
# Target: "<BOS> Bonjour monde" (shifted right for teacher forcing)
tgt_tokens = [2, 7534, 8765]  # BOS + first two target tokens
tgt_embed = embedding(tgt_tokens) * sqrt(d_model)  # (1, 3, 512)
tgt_encoded = positional_encoding(tgt_embed)       # (1, 3, 512)

# Create causal mask (can't see future tokens during training)
tgt_mask = create_causal_mask(3)  # (3, 3)

# Decode
decoder_output = decoder(tgt_encoded, memory, tgt_mask)  # (1, 3, 512)
```

### Step 3: Output Projection
```python
# Project to vocabulary
logits = output_projection(decoder_output)  # (1, 3, vocab_size)

# Each position predicts the next token:
# Position 0 (BOS) → predicts "Bonjour"
# Position 1 (Bonjour) → predicts "monde"
# Position 2 (monde) → predicts "</s>" (EOS)
```

## Why This Architecture?

### Encoder Benefits
- **Bidirectional**: Every position can attend to every other position
- **Deep contextualization**: 6 layers refine representations
- **Parallel processing**: All positions processed simultaneously

### Decoder Benefits
- **Causal masking**: Maintains autoregressive property for generation
- **Cross-attention**: Effectively uses source information
- **Same structure**: Reuses attention/FFN building blocks

### Layer Normalization
- **Training stability**: Normalizes activations, prevents gradient issues
- **Pre-norm vs post-norm**: This implementation uses post-norm (norm after residual)

### Residual Connections
- **Gradient flow**: Gradients flow directly through residual paths
- **Feature preservation**: Original features remain accessible in later layers

## Summary

| Component | Purpose | Key Parameters |
|-----------|---------|----------------|
| Positional Encoding | Add position information | d_model=512, max_seq_len=5000 |
| Feed-Forward Network | Non-linear transformation | d_model=512, d_ff=2048 |
| Encoder Layer | Self-attention + FFN | n_heads=8, dropout=0.1 |
| Encoder Stack | N encoder layers | n_layers=6 |
| Decoder Layer | Masked self-attn + cross-attn + FFN | n_heads=8, dropout=0.1 |
| Decoder Stack | N decoder layers | n_layers=6 |

---

*Next: [Chapter 4: Training Infrastructure](04_training.md)*
