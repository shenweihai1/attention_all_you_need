# Chapter 2: The Attention Mechanism

## Overview

The attention mechanism is the core innovation that makes Transformers work. This chapter explains:
1. **Scaled Dot-Product Attention** - The fundamental operation
2. **Multi-Head Attention** - Running multiple attention operations in parallel
3. **Masking** - Controlling what positions can attend to each other

## Why Attention?

Before Transformers, sequence models used RNNs which process tokens one at a time. The attention mechanism allows the model to look at **all positions simultaneously** and learn which positions are relevant to each other.

For example, in the sentence "The cat sat on the mat because it was tired":
- When processing "it", attention can directly look back at "cat" to understand the reference
- No information needs to be passed step-by-step through hidden states

## Scaled Dot-Product Attention

### The Formula

The attention formula from the paper:

```
Attention(Q, K, V) = softmax(QK^T / √d_k) × V
```

Where:
- **Q** (Query): What we're looking for
- **K** (Key): What each position offers
- **V** (Value): The actual content at each position
- **d_k**: Dimension of keys (used for scaling)

### Why Scale by √d_k?

The paper explains: "For large values of d_k, the dot products grow large in magnitude, pushing the softmax into regions where it has extremely small gradients."

Without scaling:
- If d_k = 64 and Q, K have unit variance
- QK^T has variance ≈ d_k = 64
- Large values → softmax outputs near 0 or 1 → vanishing gradients

With scaling by √d_k:
- QK^T / √64 = QK^T / 8
- Variance ≈ 1, softmax operates in its useful range

### Implementation

From `src/attention.py:16-73`:

```python
def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    dropout: Optional[nn.Dropout] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Scaled Dot-Product Attention.

    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
    """
    d_k = query.size(-1)

    # Step 1: Compute attention scores
    # QK^T / √d_k
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    # Step 2: Apply mask (optional)
    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))

    # Step 3: Softmax to get attention weights
    attention_weights = F.softmax(scores, dim=-1)

    # Step 4: Apply dropout (optional)
    if dropout is not None:
        attention_weights = dropout(attention_weights)

    # Step 5: Weighted sum of values
    output = torch.matmul(attention_weights, value)

    return output, attention_weights
```

### Step-by-Step Breakdown

Let's trace through with concrete shapes:

```python
# Input tensors (batch=2, heads=8, seq_len=10, d_k=64)
Q = torch.randn(2, 8, 10, 64)
K = torch.randn(2, 8, 10, 64)
V = torch.randn(2, 8, 10, 64)

# Step 1: Compute scores
# Q @ K^T: (2, 8, 10, 64) @ (2, 8, 64, 10) = (2, 8, 10, 10)
scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(64)
# scores[b, h, i, j] = how much position i attends to position j

# Step 2: Apply mask (if masking position j)
# mask[..., i, j] = True means position i cannot attend to j
scores = scores.masked_fill(mask, float("-inf"))

# Step 3: Softmax over last dimension (key positions)
attention_weights = F.softmax(scores, dim=-1)
# attention_weights[b, h, i, :] sums to 1
# -inf positions become 0 after softmax

# Step 4: Weighted combination of values
output = torch.matmul(attention_weights, V)
# (2, 8, 10, 10) @ (2, 8, 10, 64) = (2, 8, 10, 64)
```

### Visual Representation

```
Query (what I'm looking for)     Key (what each position offers)
     ┌───┐                            ┌───┬───┬───┬───┐
     │ Q │                            │K_0│K_1│K_2│K_3│
     └───┘                            └───┴───┴───┴───┘
       │                                      │
       └──────────────┬───────────────────────┘
                      │
                      ▼
              QK^T / √d_k
                      │
                      ▼
                  softmax
                      │
         ┌───────────────────────┐
         │ 0.1  0.5  0.3  0.1    │  (attention weights)
         └───────────────────────┘
                      │
                      ▼
         Weighted sum of Values
                      │
                      ▼
              ┌───────────┐
              │  Output   │
              └───────────┘
```

## Multi-Head Attention

### The Concept

Instead of computing attention once, we compute it **h times** (h heads) with different learned projections. This allows the model to jointly attend to information from different representation subspaces.

From the paper:
> "Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions."

### The Formula

```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O

where head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)
```

Parameters for base model:
- d_model = 512
- h = 8 heads
- d_k = d_v = d_model / h = 64

### Implementation

From `src/attention.py:182-297`:

```python
class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism.

    MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
    where head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()

        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
            )

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # 512 // 8 = 64

        # Linear projections (no bias, as per paper)
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else None
```

### The Forward Pass

```python
def forward(self, query, key, value, mask=None):
    batch_size = query.size(0)
    seq_len_q = query.size(1)
    seq_len_k = key.size(1)

    # Step 1: Linear projections
    q = self.w_q(query)  # (batch, seq_len, d_model)
    k = self.w_k(key)
    v = self.w_v(value)

    # Step 2: Reshape to separate heads
    # (batch, seq_len, d_model) → (batch, n_heads, seq_len, d_k)
    q = q.view(batch_size, seq_len_q, self.n_heads, self.d_k).transpose(1, 2)
    k = k.view(batch_size, seq_len_k, self.n_heads, self.d_k).transpose(1, 2)
    v = v.view(batch_size, seq_len_k, self.n_heads, self.d_k).transpose(1, 2)

    # Step 3: Scaled dot-product attention (all heads in parallel)
    attn_output, attn_weights = scaled_dot_product_attention(
        q, k, v, mask=mask, dropout=self.dropout
    )

    # Step 4: Concatenate heads
    # (batch, n_heads, seq_len, d_k) → (batch, seq_len, d_model)
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.view(batch_size, seq_len_q, self.d_model)

    # Step 5: Final output projection
    output = self.w_o(attn_output)

    return output, attn_weights
```

### Why Multiple Heads?

Each head can learn to focus on different patterns:

```
Head 1: Focuses on syntactic relationships
        "The cat sat" → attention to subject-verb

Head 2: Focuses on semantic relationships
        "cat ... it" → attention to coreference

Head 3: Focuses on positional patterns
        Adjacent tokens might attend to each other

Head 4-8: Other learned patterns
```

### Visual: Multi-Head Attention

```
Input: x (batch, seq_len, d_model=512)
          │
          ▼
    ┌─────┴─────┐
    │  Linear   │  W_Q, W_K, W_V
    └─────┬─────┘
          │
          ▼
    ┌─────────────────────────────────────┐
    │        Split into 8 heads           │
    │  (batch, 8, seq_len, 64)            │
    └─────────────────────────────────────┘
          │
    ┌─────┼─────┬─────┬─────┬─────┬─────┬─────┬─────┐
    ▼     ▼     ▼     ▼     ▼     ▼     ▼     ▼     ▼
  Head1 Head2 Head3 Head4 Head5 Head6 Head7 Head8
    │     │     │     │     │     │     │     │
    │  (Each head does scaled dot-product attention)
    │     │     │     │     │     │     │     │
    └─────┴─────┴─────┴─────┴─────┴─────┴─────┘
                      │
                      ▼
            ┌─────────────────┐
            │   Concatenate   │
            │ (batch, seq, 512)│
            └────────┬────────┘
                     │
                     ▼
               ┌───────────┐
               │  Linear   │  W_O
               │   (512)   │
               └─────┬─────┘
                     │
                     ▼
              Output (batch, seq_len, 512)
```

## Masking

Masks control which positions can attend to which. There are two main types:

### 1. Padding Mask

Prevents attention to padding tokens (usually index 0).

From `src/attention.py:157-179`:

```python
def create_padding_mask(seq: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
    """
    Create a padding mask from a sequence tensor.

    Returns shape (batch, 1, 1, seq_len) for broadcasting.
    """
    # (batch, seq_len) → (batch, 1, 1, seq_len)
    mask = (seq == pad_idx).unsqueeze(1).unsqueeze(2)
    return mask
```

Example:
```python
seq = torch.tensor([[5, 3, 2, 0, 0],   # Last 2 are padding
                    [4, 1, 0, 0, 0]])  # Last 3 are padding

mask = create_padding_mask(seq, pad_idx=0)
# mask[0] = [[[[False, False, False, True, True]]]]
# mask[1] = [[[[False, False, True, True, True]]]]

# When applied to attention scores:
# Position i attending to padded position j gets score -inf
# After softmax: attention weight becomes 0
```

### 2. Causal (Look-Ahead) Mask

Prevents positions from attending to future positions. Used in the decoder.

From `src/attention.py:129-154`:

```python
def create_causal_mask(seq_len: int, device=None) -> torch.Tensor:
    """
    Create a causal mask for decoder self-attention.

    Returns upper triangular matrix where True = "do not attend".
    """
    mask = torch.triu(
        torch.ones(seq_len, seq_len, device=device),
        diagonal=1
    ).bool()
    return mask
```

Example:
```python
mask = create_causal_mask(4)
# tensor([[False,  True,  True,  True],
#         [False, False,  True,  True],
#         [False, False, False,  True],
#         [False, False, False, False]])

# Position 0 can only attend to position 0
# Position 1 can attend to positions 0, 1
# Position 2 can attend to positions 0, 1, 2
# Position 3 can attend to all positions
```

### Why Causal Masking?

During training, the decoder sees the entire target sequence at once (teacher forcing). Without causal masking, position 3 could "cheat" by looking at positions 4, 5, ... to predict the next token.

```
Target: "The cat sat on the mat"

Without mask: Position "sat" sees "on the mat" → trivial prediction
With mask:    Position "sat" only sees "The cat sat" → must learn patterns
```

## Three Types of Attention in Transformers

### 1. Encoder Self-Attention

- Query, Key, Value all come from encoder input
- Uses padding mask only
- Every position can attend to every other position

```python
# In encoder layer
self_attn_output = self.self_attention(x, x, x, mask=src_padding_mask)
```

### 2. Decoder Self-Attention (Masked)

- Query, Key, Value all come from decoder input
- Uses causal mask + padding mask
- Position i can only attend to positions 0...i

```python
# In decoder layer
self_attn_output = self.self_attention(x, x, x, mask=tgt_mask)
# tgt_mask combines causal mask and padding mask
```

### 3. Encoder-Decoder (Cross) Attention

- Query from decoder, Key/Value from encoder output
- Uses source padding mask only
- Decoder positions attend to encoder outputs

```python
# In decoder layer
cross_attn_output = self.cross_attention(
    x,              # Query from decoder
    encoder_output, # Key from encoder
    encoder_output, # Value from encoder
    mask=src_padding_mask
)
```

### Visual: Attention Types

```
ENCODER SELF-ATTENTION          DECODER MASKED SELF-ATTENTION
(all-to-all)                    (causal)

  Attend to:                      Attend to:
  1 2 3 4                         1 2 3 4
1 ✓ ✓ ✓ ✓                       1 ✓ ✗ ✗ ✗
2 ✓ ✓ ✓ ✓                       2 ✓ ✓ ✗ ✗
3 ✓ ✓ ✓ ✓                       3 ✓ ✓ ✓ ✗
4 ✓ ✓ ✓ ✓                       4 ✓ ✓ ✓ ✓


CROSS-ATTENTION
(decoder queries attend to all encoder positions)

Decoder Query    Encoder Keys
    ↓             1 2 3 4 5
    1           → ✓ ✓ ✓ ✓ ✓
    2           → ✓ ✓ ✓ ✓ ✓
    3           → ✓ ✓ ✓ ✓ ✓
```

## Usage Examples

### Basic Scaled Dot-Product Attention

```python
from src.attention import scaled_dot_product_attention

# Create sample tensors
batch_size, seq_len, d_k = 2, 10, 64
Q = torch.randn(batch_size, seq_len, d_k)
K = torch.randn(batch_size, seq_len, d_k)
V = torch.randn(batch_size, seq_len, d_k)

# Compute attention
output, weights = scaled_dot_product_attention(Q, K, V)
print(f"Output shape: {output.shape}")     # (2, 10, 64)
print(f"Weights shape: {weights.shape}")   # (2, 10, 10)
```

### Multi-Head Attention

```python
from src.attention import MultiHeadAttention

# Create multi-head attention module
mha = MultiHeadAttention(d_model=512, n_heads=8, dropout=0.1)

# Input tensor
x = torch.randn(2, 20, 512)  # (batch, seq_len, d_model)

# Self-attention (Q=K=V=x)
output, weights = mha(x, x, x)
print(f"Output shape: {output.shape}")     # (2, 20, 512)
print(f"Weights shape: {weights.shape}")   # (2, 8, 20, 20)
```

### With Masking

```python
from src.attention import (
    MultiHeadAttention,
    create_causal_mask,
    create_padding_mask
)

mha = MultiHeadAttention(d_model=512, n_heads=8)

# Create masks
seq = torch.tensor([[1, 2, 3, 0, 0],
                    [1, 2, 3, 4, 0]])  # 0 is padding
padding_mask = create_padding_mask(seq)  # (2, 1, 1, 5)
causal_mask = create_causal_mask(5)      # (5, 5)

# Combine masks for decoder
combined_mask = padding_mask | causal_mask.unsqueeze(0).unsqueeze(0)

# Apply attention with mask
x = torch.randn(2, 5, 512)
output, weights = mha(x, x, x, mask=combined_mask)
```

## Key Takeaways

1. **Scaled dot-product attention** computes relevance scores between queries and keys, then uses these to weight values.

2. **Scaling by √d_k** prevents gradient issues when d_k is large.

3. **Multi-head attention** runs multiple attention operations in parallel, allowing the model to capture different types of relationships.

4. **Masking** controls attention patterns:
   - Padding mask: ignore pad tokens
   - Causal mask: prevent looking at future tokens

5. **Three attention types** in Transformers:
   - Encoder self-attention (bidirectional)
   - Decoder self-attention (causal/unidirectional)
   - Cross-attention (decoder attends to encoder)

---

*Next: [Chapter 3: Encoder and Decoder](03_encoder_decoder.md)*
