
## Reproduce Transformer
Goal: Reproduce Attention is all your need paper.
  - Just evaluate Base model (big model is not necessary).
  - You should use PyTorch to implement this framework, but you can not use `torch.nn.Transformer` and `torch.nn.MultiheadAttention`. You should implement a `Transformer` using PyTorch's other existing functions.
  - During iterating code, you don't need to validate all your code, because I don't have a target server. The running server does not have GPU.

Deliverables:
  - Add a README.md to include all necessary important information about this project.
  - Include instructions and installations etc to show how to train the model on target server RTX 5090 (server version is `pytorch:1.0.2-cu1281-torch280-ubuntu2404`).
  - Make code clear and readable.

### Task Breakdown

#### 1. Project Setup [COMPLETED]
- [x] 1.1 Create project structure (folders: src/, tests/, docs/, configs/)
- [x] 1.2 Create requirements.txt with dependencies
- [x] 1.3 Create base configuration system for model hyperparameters

#### 2. Core Transformer Components [NOT STARTED]
- [ ] 2.1 Implement Scaled Dot-Product Attention
  - Implement attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
  - Handle optional masking for decoder self-attention
- [ ] 2.2 Implement Multi-Head Attention (without using torch.nn.MultiheadAttention)
  - Split into h heads, apply attention, concatenate, project
  - Base model: h=8 heads, d_model=512, d_k=d_v=64
- [ ] 2.3 Implement Position-wise Feed-Forward Network
  - FFN(x) = max(0, xW1 + b1)W2 + b2
  - Inner layer dimension d_ff=2048
- [ ] 2.4 Implement Positional Encoding
  - Sinusoidal positional encoding as per paper
  - PE(pos, 2i) = sin(pos/10000^(2i/d_model))
  - PE(pos, 2i+1) = cos(pos/10000^(2i/d_model))

#### 3. Encoder and Decoder [NOT STARTED]
- [ ] 3.1 Implement Encoder Layer (self-attention + FFN + residual + layer norm)
- [ ] 3.2 Implement Encoder Stack (N=6 layers for base model)
- [ ] 3.3 Implement Decoder Layer (masked self-attention + cross-attention + FFN)
- [ ] 3.4 Implement Decoder Stack (N=6 layers for base model)

#### 4. Full Transformer Model [NOT STARTED]
- [ ] 4.1 Implement Input Embedding with scaling (multiply by sqrt(d_model))
- [ ] 4.2 Implement Full Transformer (encoder + decoder + final linear + softmax)
- [ ] 4.3 Implement weight initialization as per paper

#### 5. Training Infrastructure [NOT STARTED]
- [ ] 5.1 Implement learning rate scheduler (warmup + inverse sqrt decay)
- [ ] 5.2 Implement label smoothing (epsilon=0.1)
- [ ] 5.3 Create training loop with gradient accumulation support

#### 6. Data Processing [NOT STARTED]
- [ ] 6.1 Implement tokenization utilities (BPE or similar)
- [ ] 6.2 Implement data loading for WMT translation dataset
- [ ] 6.3 Implement batching with dynamic padding

#### 7. Documentation and README [NOT STARTED]
- [ ] 7.1 Write comprehensive README.md with project overview
- [ ] 7.2 Add installation and setup instructions for RTX 5090 target server
- [ ] 7.3 Add training and inference usage examples

