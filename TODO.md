
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

#### 2. Core Transformer Components [COMPLETED]
- [ ] *high* Make a comprehensive markdown book to explain the codebase to let beginners to understand what and why in the code.
- [x] 2.1 Implement Scaled Dot-Product Attention
  - Implemented attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
  - Added optional masking support for decoder self-attention
  - Includes helper functions: create_causal_mask, create_padding_mask
- [x] 2.2 Implement Multi-Head Attention (without using torch.nn.MultiheadAttention)
  - Implemented MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
  - Uses linear projections W_Q, W_K, W_V for each head
  - Supports self-attention and cross-attention patterns
  - Base model: h=8 heads, d_model=512, d_k=d_v=64
- [x] 2.3 Implement Position-wise Feed-Forward Network
  - Implemented FFN(x) = max(0, xW1 + b1)W2 + b2
  - Two linear transformations with ReLU activation
  - Inner layer dimension d_ff=2048, applied position-wise
- [x] 2.4 Implement Positional Encoding
  - Implemented sinusoidal positional encoding as per paper
  - PE(pos, 2i) = sin(pos/10000^(2i/d_model))
  - PE(pos, 2i+1) = cos(pos/10000^(2i/d_model))
  - Pre-computed and registered as buffer for efficiency

#### 3. Encoder and Decoder [COMPLETED]
- [x] 3.1 Implement Encoder Layer (self-attention + FFN + residual + layer norm)
  - Multi-head self-attention with residual connection and layer norm
  - Position-wise FFN with residual connection and layer norm
  - Dropout on sublayer outputs before residual addition
- [x] 3.2 Implement Encoder Stack (N=6 layers for base model)
  - Stack of N identical EncoderLayer modules
  - Final layer normalization after all layers
  - Supports optional source mask for padding
- [x] 3.3 Implement Decoder Layer (masked self-attention + cross-attention + FFN)
  - Masked multi-head self-attention (prevents attending to future positions)
  - Multi-head cross-attention (attends to encoder output)
  - Position-wise FFN with residual connections and layer normalization
- [x] 3.4 Implement Decoder Stack (N=6 layers for base model)
  - Stack of N identical DecoderLayer modules
  - Final layer normalization after all layers
  - Supports target mask (causal) and memory mask (padding)

#### 4. Full Transformer Model [COMPLETED]
- [x] 4.1 Implement Input Embedding with scaling (multiply by sqrt(d_model))
  - TransformerEmbedding class wrapping nn.Embedding
  - Scales embeddings by sqrt(d_model) as per paper
  - Supports padding_idx for proper gradient masking
- [x] 4.2 Implement Full Transformer (encoder + decoder + final linear + softmax)
  - Complete encoder-decoder architecture with embeddings and positional encoding
  - Final linear projection to target vocabulary size
  - Automatic mask generation for padding and causal attention
  - Autoregressive generation with greedy decoding support
- [x] 4.3 Implement weight initialization as per paper
  - Xavier uniform initialization for Linear layers
  - Normal distribution (std=d_model^-0.5) for Embeddings
  - LayerNorm initialized to weight=1, bias=0
  - BERT-style alternative initialization (std=0.02)
  - Auto-initialization on model creation

#### 5. Training Infrastructure [COMPLETED]
- [x] 5.1 Implement learning rate scheduler (warmup + inverse sqrt decay)
  - TransformerScheduler: implements lrate = d_model^(-0.5) * min(step^(-0.5), step * warmup^(-1.5))
  - Linear warmup during first warmup_steps (default 4000)
  - Inverse square root decay after warmup
  - Also includes WarmupScheduler and InverseSquareRootScheduler variants
  - Convenience functions: get_transformer_scheduler, get_lr_at_step
- [x] 5.2 Implement label smoothing (epsilon=0.1)
  - LabelSmoothingLoss: uses KL divergence with smoothed targets
  - LabelSmoothingCrossEntropy: alternative implementation using CE formulation
  - Functional interface: label_smoothing_loss()
  - Supports padding token masking and different reduction modes
  - Default smoothing epsilon=0.1 as per paper
- [x] 5.3 Create training loop with gradient accumulation support
  - Trainer class with complete training loop
  - Gradient accumulation for effective larger batch sizes
  - Gradient clipping with configurable max norm
  - Checkpoint save/load functionality
  - Validation loop with metrics
  - Log callback support for custom logging
  - TrainerConfig dataclass for configuration
  - create_trainer convenience function

#### 6. Data Processing [COMPLETED]
- [x] 6.1 Implement tokenization utilities (BPE or similar)
  - Tokenizer class wrapping SentencePiece for BPE tokenization
  - SimpleTokenizer for whitespace-based tokenization (testing/simple cases)
  - Support for training from text files or text lists
  - Special tokens: PAD, UNK, BOS, EOS
  - pad_sequences function for batch padding
  - create_padding_mask_from_lengths utility
- [x] 6.2 Implement data loading for WMT translation dataset
  - TranslationDataset: loads parallel source-target pairs from lists or files
  - TranslationCollator: batches with padding and mask creation
  - create_translation_dataloader: convenience function for DataLoader creation
  - SortedBatchSampler: groups similar-length sequences to reduce padding
  - BucketIterator: sophisticated length-based batching
  - load_wmt_dataset: HuggingFace datasets integration (optional)
- [x] 6.3 Implement batching with dynamic padding
  - DynamicBatchSampler: creates batches based on max_tokens constraint
  - create_dynamic_dataloader: convenience function for token-based batching
  - Batches adapt padding to actual sequence lengths in each batch
  - Supports max_sentences cap and length-based sorting

#### 7. Documentation and README [COMPLETED]
- [x] 7.1 Write comprehensive README.md with project overview
  - Complete project overview with key features
  - Project structure documentation
  - Model architecture details with formulas from paper
  - Testing and configuration instructions
- [x] 7.2 Add installation and setup instructions for RTX 5090 target server
  - Requirements section (Python 3.8+, PyTorch 2.0+, CUDA 12.1+)
  - Quick setup instructions
  - Target server specific instructions for pytorch:1.0.2-cu1281-torch280-ubuntu2404
  - Optional dependencies (SentencePiece, HuggingFace datasets)
- [x] 7.3 Add training and inference usage examples
  - Quick start with model creation examples
  - Forward pass and autoregressive generation examples
  - Complete training example with all components
  - Dynamic batching usage
  - Checkpoint management
  - BPE tokenization and WMT dataset loading

