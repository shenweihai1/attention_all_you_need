# Chapter 5: Data Processing

## Overview

Efficient data processing is critical for training Transformers. This chapter covers:
1. **Tokenization** - Converting text to numerical tokens (BPE)
2. **Datasets** - Loading and managing parallel translation data
3. **Batching** - Creating efficient batches with minimal padding
4. **Dynamic Batching** - Token-based batching for memory efficiency

## Tokenization

### Why Subword Tokenization?

Word-level tokenization has problems:
- Large vocabularies (100K+ words)
- Cannot handle rare/unknown words (OOV)
- Doesn't share information between related words ("run", "running", "runner")

**Byte Pair Encoding (BPE)** solves these by breaking words into subword units:

```
"unbelievable" → ["un", "believ", "able"]
"running"      → ["run", "ning"]
"GPT2"         → ["G", "PT", "2"]  (handles rare tokens)
```

### BPE Algorithm

1. Start with character-level vocabulary
2. Count all adjacent character pairs
3. Merge most frequent pair into new token
4. Repeat until vocabulary size reached

```
Vocabulary: {a, b, c, d, ...}
Text: "aaabdaaabac"

Iteration 1: Most frequent pair is "aa" → merge
Vocabulary: {a, b, c, d, ..., aa}
Text: "aa ab d aa ab a c"

Iteration 2: Most frequent pair is "aa ab" → merge
...
```

### The Tokenizer Class

This codebase uses SentencePiece for BPE tokenization.

From `src/tokenizer.py:44-179`:

```python
class Tokenizer:
    """
    BPE tokenizer wrapper using SentencePiece.
    """

    def __init__(self, model_path=None, vocab_size=32000, model_type="bpe"):
        if not HAS_SENTENCEPIECE:
            raise ImportError("sentencepiece required. pip install sentencepiece")

        self.vocab_size = vocab_size
        self.model_type = model_type
        self._sp = None

        if model_path is not None:
            self.load(model_path)

    def encode(self, text, add_bos=False, add_eos=False):
        """Encode text to token IDs."""
        ids = self._sp.EncodeAsIds(text)

        if add_bos:
            ids = [BOS_ID] + ids
        if add_eos:
            ids = ids + [EOS_ID]

        return ids

    def decode(self, ids, skip_special_tokens=True):
        """Decode token IDs to text."""
        if skip_special_tokens:
            ids = [i for i in ids if i not in (PAD_ID, BOS_ID, EOS_ID)]
        return self._sp.DecodeIds(ids)
```

### Special Tokens

| Token | ID | Purpose |
|-------|-----|---------|
| `<pad>` | 0 | Padding (fill shorter sequences) |
| `<unk>` | 1 | Unknown token (rare words) |
| `<s>` | 2 | Beginning of sequence (BOS) |
| `</s>` | 3 | End of sequence (EOS) |

### Training a Tokenizer

```python
from src.tokenizer import Tokenizer

# Train from text files
tokenizer = Tokenizer.train(
    input_files=["train.en", "train.de"],  # Both languages together
    model_prefix="wmt_bpe",
    vocab_size=37000,  # Paper uses 37K shared vocabulary
    model_type="bpe"
)

# Train from list of strings
tokenizer = Tokenizer.train_from_texts(
    texts=["Hello world", "Bonjour monde", ...],
    model_prefix="my_tokenizer",
    vocab_size=8000
)
```

### Using the Tokenizer

```python
# Load existing model
tokenizer = Tokenizer(model_path="wmt_bpe.model")

# Encode text
ids = tokenizer.encode("Hello world", add_bos=True, add_eos=True)
# [2, 1234, 5678, 3]  # <s> Hello world </s>

# Decode back
text = tokenizer.decode(ids)
# "Hello world"

# Batch encoding
texts = ["Hello", "World"]
batch_ids = tokenizer.encode_batch(texts, add_bos=True, add_eos=True)
# [[2, 1234, 3], [2, 5678, 3]]
```

### SimpleTokenizer (for Testing)

For quick tests without SentencePiece:

```python
from src.tokenizer import SimpleTokenizer

tokenizer = SimpleTokenizer()
tokenizer.build_vocab(["hello world", "hello there"])

ids = tokenizer.encode("hello world")
# [4, 5]  (vocabulary: hello=4, world=5)

tokenizer.vocab_size  # 6 (4 special + 2 words)
```

## Translation Dataset

### Dataset Structure

The `TranslationDataset` class handles parallel source-target pairs.

From `src/data.py:40-144`:

```python
class TranslationDataset(Dataset):
    """
    Dataset for parallel translation data.
    """

    def __init__(
        self,
        src_data,       # List of strings or file path
        tgt_data,       # List of strings or file path
        src_tokenizer=None,
        tgt_tokenizer=None,
        max_length=None,
        add_bos=True,
        add_eos=True,
    ):
        self.src_sentences = self._load_data(src_data)
        self.tgt_sentences = self._load_data(tgt_data)

        if len(self.src_sentences) != len(self.tgt_sentences):
            raise ValueError("Source and target have different lengths")

    def __getitem__(self, idx):
        src_text = self.src_sentences[idx]
        tgt_text = self.tgt_sentences[idx]

        result = {"src_text": src_text, "tgt_text": tgt_text}

        # Tokenize if tokenizers provided
        if self.src_tokenizer:
            src_ids = self.src_tokenizer.encode(src_text, add_bos, add_eos)
            result["src"] = torch.tensor(src_ids, dtype=torch.long)

        if self.tgt_tokenizer:
            tgt_ids = self.tgt_tokenizer.encode(tgt_text, add_bos, add_eos)
            result["tgt"] = torch.tensor(tgt_ids, dtype=torch.long)

        return result
```

### Loading Data

```python
from src.data import TranslationDataset

# From files
dataset = TranslationDataset(
    src_data="train.en",
    tgt_data="train.de",
    src_tokenizer=tokenizer,
    tgt_tokenizer=tokenizer,  # Shared tokenizer
    max_length=512,
)

# From lists
dataset = TranslationDataset(
    src_data=["Hello world", "Good morning"],
    tgt_data=["Hallo Welt", "Guten Morgen"],
    src_tokenizer=tokenizer,
    tgt_tokenizer=tokenizer,
)

# Access examples
print(len(dataset))  # Number of sentence pairs
example = dataset[0]
print(example["src"])  # Source token IDs
print(example["tgt"])  # Target token IDs
```

### Loading WMT Dataset (HuggingFace)

```python
from src.data import load_wmt_dataset

# Load WMT14 German-English
dataset = load_wmt_dataset(
    name="wmt14",
    language_pair="de-en",
    split="train",
    src_tokenizer=tokenizer,
    tgt_tokenizer=tokenizer,
    max_samples=100000,  # Limit for testing
)
```

## Batching and Collation

### The Padding Problem

Sequences have different lengths, but batches need uniform shape:

```
Sequence 1: [2, 101, 102, 103, 3]           (length 5)
Sequence 2: [2, 201, 202, 3]                (length 4)
Sequence 3: [2, 301, 302, 303, 304, 305, 3] (length 7)

After padding to max_length=7:
[2, 101, 102, 103, 3,   0,   0]
[2, 201, 202, 3,   0,   0,   0]
[2, 301, 302, 303, 304, 305, 3]
```

### TranslationCollator

From `src/data.py:147-209`:

```python
class TranslationCollator:
    """
    Collator for batching translation examples with padding.
    """

    def __init__(self, pad_id=PAD_ID, padding_side="right"):
        self.pad_id = pad_id
        self.padding_side = padding_side

    def __call__(self, batch):
        result = {}

        # Pad source sequences
        if "src" in batch[0]:
            src_sequences = [item["src"].tolist() for item in batch]
            result["src"] = pad_sequences(
                src_sequences,
                padding_value=self.pad_id,
            )
            # Create padding mask (True where padded)
            result["src_mask"] = result["src"] == self.pad_id

        # Pad target sequences
        if "tgt" in batch[0]:
            tgt_sequences = [item["tgt"].tolist() for item in batch]
            result["tgt"] = pad_sequences(
                tgt_sequences,
                padding_value=self.pad_id,
            )
            result["tgt_mask"] = result["tgt"] == self.pad_id

        return result
```

### Creating a DataLoader

```python
from src.data import create_translation_dataloader

loader = create_translation_dataloader(
    src_data="train.en",
    tgt_data="train.de",
    src_tokenizer=tokenizer,
    tgt_tokenizer=tokenizer,
    batch_size=32,
    max_length=512,
    shuffle=True,
)

for batch in loader:
    src = batch["src"]       # (batch_size, max_src_len)
    tgt = batch["tgt"]       # (batch_size, max_tgt_len)
    src_mask = batch["src_mask"]  # (batch_size, max_src_len)
    # ... training ...
```

## Efficient Batching Strategies

### Problem: Wasted Computation

With fixed batch sizes, short sequences waste computation:

```
Batch with batch_size=4:
[2, 101, 3,   0,   0,   0,   0,   0]  ← 62% padding!
[2, 201, 202, 3,   0,   0,   0,   0]  ← 50% padding!
[2, 301, 302, 303, 304, 305, 306, 3]  ← 0% padding
[2, 401, 402, 403, 3,   0,   0,   0]  ← 37% padding!
```

### Solution 1: Sorted Batch Sampler

Group similar-length sequences together:

From `src/data.py:342-395`:

```python
class SortedBatchSampler:
    """
    Batch sampler that groups similar-length sequences.
    """

    def __init__(self, lengths, batch_size, shuffle=True, drop_last=False):
        self.lengths = lengths
        self.batch_size = batch_size

        # Sort indices by length
        self.sorted_indices = sorted(
            range(len(lengths)),
            key=lambda i: lengths[i]
        )

    def __iter__(self):
        # Create batches from sorted sequences
        batches = []
        for i in range(0, len(self.sorted_indices), self.batch_size):
            batch = self.sorted_indices[i : i + self.batch_size]
            batches.append(batch)

        # Shuffle batches (not sequences within batches)
        if self.shuffle:
            random.shuffle(batches)

        yield from batches
```

Result:
```
Batch 1 (short sequences):
[2, 101, 3]              ← 0% padding
[2, 201, 3]              ← 0% padding
[2, 301, 302, 3]         ← 0% padding
[2, 401, 3]              ← 0% padding

Batch 2 (long sequences):
[2, 501, 502, 503, 504, 505, 506, 3]
[2, 601, 602, 603, 604, 605, 3,   0]  ← 12% padding
[2, 701, 702, 703, 704, 705, 706, 707, 3]
...
```

### Solution 2: Dynamic Batch Sampler (Token-Based)

Instead of fixed sentence count, use fixed **token count**:

From `src/data.py:484-578`:

```python
class DynamicBatchSampler:
    """
    Batch sampler that creates batches based on maximum tokens.
    """

    def __init__(
        self,
        lengths,
        max_tokens=4096,
        max_sentences=None,
        shuffle=True,
        sort_by_length=True,
    ):
        self.lengths = lengths
        self.max_tokens = max_tokens
        self.max_sentences = max_sentences

        # Pre-compute batches
        self._batches = self._create_batches()

    def _create_batches(self):
        # Sort by length
        if self.sort_by_length:
            indices = sorted(range(len(self.lengths)), key=lambda i: self.lengths[i])
        else:
            indices = list(range(len(self.lengths)))

        batches = []
        current_batch = []
        current_max_len = 0

        for idx in indices:
            sample_len = self.lengths[idx]

            # Calculate tokens if we add this sample
            new_max_len = max(current_max_len, sample_len)
            new_tokens = new_max_len * (len(current_batch) + 1)

            # Check if adding this sample exceeds limits
            if current_batch and new_tokens > self.max_tokens:
                batches.append(current_batch)
                current_batch = [idx]
                current_max_len = sample_len
            else:
                current_batch.append(idx)
                current_max_len = new_max_len

        if current_batch:
            batches.append(current_batch)

        return batches
```

### Visual: Dynamic Batching

```
max_tokens = 4096

Short sequences (len ≈ 50):
  Batch can fit ~80 sentences (80 × 50 = 4000 tokens)

Long sequences (len ≈ 200):
  Batch can fit ~20 sentences (20 × 200 = 4000 tokens)

Result: Consistent memory usage regardless of sequence length!
```

### Using Dynamic Batching

```python
from src.data import create_dynamic_dataloader

loader = create_dynamic_dataloader(
    dataset=dataset,
    max_tokens=4096,     # ~4K tokens per batch
    max_sentences=128,   # Cap at 128 sentences
    shuffle=True,
)

for batch in loader:
    # Batch sizes vary, but token count ≈ 4096
    print(f"Batch size: {batch['src'].size(0)}, "
          f"Tokens: {batch['src'].numel()}")
```

## Padding Utilities

### pad_sequences Function

From `src/tokenizer.py:517-554`:

```python
def pad_sequences(
    sequences,
    padding_value=PAD_ID,
    max_length=None,
    padding_side="right",
):
    """Pad sequences to the same length."""
    if max_length is None:
        max_length = max(len(seq) for seq in sequences)

    batch_size = len(sequences)
    padded = torch.full((batch_size, max_length), padding_value, dtype=torch.long)

    for i, seq in enumerate(sequences):
        length = min(len(seq), max_length)
        if padding_side == "right":
            padded[i, :length] = torch.tensor(seq[:length])
        else:  # left
            padded[i, -length:] = torch.tensor(seq[:length])

    return padded
```

### Creating Masks from Lengths

```python
from src.tokenizer import create_padding_mask_from_lengths

lengths = [5, 3, 7]
mask = create_padding_mask_from_lengths(lengths, max_length=7)
# tensor([[False, False, False, False, False,  True,  True],
#         [False, False, False,  True,  True,  True,  True],
#         [False, False, False, False, False, False, False]])
```

## Complete Data Pipeline Example

```python
import torch
from src.tokenizer import Tokenizer, SimpleTokenizer
from src.data import (
    TranslationDataset,
    create_dynamic_dataloader,
)

# 1. Create or load tokenizer
tokenizer = SimpleTokenizer()
tokenizer.build_vocab([
    "The cat sat on the mat",
    "Die Katze saß auf der Matte",
    "Hello world",
    "Hallo Welt",
])

# 2. Create dataset
dataset = TranslationDataset(
    src_data=["The cat sat on the mat", "Hello world"],
    tgt_data=["Die Katze saß auf der Matte", "Hallo Welt"],
    src_tokenizer=tokenizer,
    tgt_tokenizer=tokenizer,
    add_bos=True,
    add_eos=True,
)

# 3. Create dataloader with dynamic batching
loader = create_dynamic_dataloader(
    dataset=dataset,
    max_tokens=1024,
    shuffle=True,
)

# 4. Training loop
for batch in loader:
    src = batch["src"]        # (batch_size, src_len)
    tgt = batch["tgt"]        # (batch_size, tgt_len)
    src_mask = batch["src_mask"]
    tgt_mask = batch["tgt_mask"]

    # Teacher forcing
    tgt_input = tgt[:, :-1]   # Input: all but last
    tgt_output = tgt[:, 1:]   # Target: all but first

    # Forward pass
    logits = model(src, tgt_input)
    # ...
```

## Summary

| Component | Purpose | Key Methods |
|-----------|---------|-------------|
| Tokenizer | BPE tokenization | `encode()`, `decode()`, `train()` |
| SimpleTokenizer | Whitespace tokenization (testing) | `build_vocab()`, `encode()` |
| TranslationDataset | Load parallel data | `__getitem__()` |
| TranslationCollator | Batch with padding | `__call__()` |
| SortedBatchSampler | Group similar lengths | `__iter__()` |
| DynamicBatchSampler | Token-based batching | `__iter__()` |

### Paper's Data Settings

| Setting | Value |
|---------|-------|
| Vocabulary | 37,000 (shared BPE) |
| Batch size | ~25,000 tokens |
| Max sequence length | 512 tokens |
| Dataset | WMT14 English-German |

---

*Next: [Chapter 6: Practical Examples](06_examples.md)*
