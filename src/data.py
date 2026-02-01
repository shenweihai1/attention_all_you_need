"""
Data loading utilities for Transformer training.

This module provides dataset classes and data loading utilities for
machine translation, specifically designed for the Transformer model
as described in "Attention Is All You Need".

Supports:
- Parallel text file loading (source and target files)
- Integration with tokenizers
- HuggingFace datasets (when available)
- Collation with padding for batching
"""

import os
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Tuple,
    Union,
)

import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset

from src.tokenizer import (
    SimpleTokenizer,
    pad_sequences,
    PAD_ID,
    BOS_ID,
    EOS_ID,
)


class TranslationDataset(Dataset):
    """
    Dataset for parallel translation data.

    Loads source-target sentence pairs from parallel text files or lists.
    Optionally tokenizes the data using a provided tokenizer.

    Args:
        src_data: Source sentences (list of strings or path to file)
        tgt_data: Target sentences (list of strings or path to file)
        src_tokenizer: Optional tokenizer for source language
        tgt_tokenizer: Optional tokenizer for target language
        max_length: Maximum sequence length (truncates longer sequences)
        add_bos: Add beginning-of-sequence token
        add_eos: Add end-of-sequence token

    Example:
        >>> dataset = TranslationDataset(
        ...     src_data="train.en",
        ...     tgt_data="train.de",
        ...     src_tokenizer=tokenizer,
        ...     tgt_tokenizer=tokenizer,
        ... )
        >>> src, tgt = dataset[0]
    """

    def __init__(
        self,
        src_data: Union[str, List[str]],
        tgt_data: Union[str, List[str]],
        src_tokenizer: Optional[Any] = None,
        tgt_tokenizer: Optional[Any] = None,
        max_length: Optional[int] = None,
        add_bos: bool = True,
        add_eos: bool = True,
    ):
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.max_length = max_length
        self.add_bos = add_bos
        self.add_eos = add_eos

        # Load data
        self.src_sentences = self._load_data(src_data)
        self.tgt_sentences = self._load_data(tgt_data)

        if len(self.src_sentences) != len(self.tgt_sentences):
            raise ValueError(
                f"Source and target have different lengths: "
                f"{len(self.src_sentences)} vs {len(self.tgt_sentences)}"
            )

    def _load_data(self, data: Union[str, List[str]]) -> List[str]:
        """Load data from file path or return list directly."""
        if isinstance(data, str):
            # Treat as file path
            path = Path(data)
            if not path.exists():
                raise FileNotFoundError(f"Data file not found: {path}")
            with open(path, "r", encoding="utf-8") as f:
                return [line.strip() for line in f if line.strip()]
        else:
            return list(data)

    def __len__(self) -> int:
        return len(self.src_sentences)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single example.

        Returns:
            Dictionary with 'src' and 'tgt' keys, containing either
            token IDs (if tokenizer provided) or raw strings.
        """
        src_text = self.src_sentences[idx]
        tgt_text = self.tgt_sentences[idx]

        result = {
            "src_text": src_text,
            "tgt_text": tgt_text,
        }

        # Tokenize if tokenizers provided
        if self.src_tokenizer is not None:
            src_ids = self.src_tokenizer.encode(
                src_text,
                add_bos=self.add_bos,
                add_eos=self.add_eos,
            )
            if self.max_length:
                src_ids = src_ids[: self.max_length]
            result["src"] = torch.tensor(src_ids, dtype=torch.long)

        if self.tgt_tokenizer is not None:
            tgt_ids = self.tgt_tokenizer.encode(
                tgt_text,
                add_bos=self.add_bos,
                add_eos=self.add_eos,
            )
            if self.max_length:
                tgt_ids = tgt_ids[: self.max_length]
            result["tgt"] = torch.tensor(tgt_ids, dtype=torch.long)

        return result


class TranslationCollator:
    """
    Collator for batching translation examples with padding.

    Pads source and target sequences to the maximum length in the batch.

    Args:
        pad_id: Padding token ID (default: 0)
        padding_side: 'right' or 'left' padding

    Example:
        >>> collator = TranslationCollator(pad_id=0)
        >>> dataloader = DataLoader(dataset, batch_size=32, collate_fn=collator)
    """

    def __init__(
        self,
        pad_id: int = PAD_ID,
        padding_side: str = "right",
    ):
        self.pad_id = pad_id
        self.padding_side = padding_side

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Collate a batch of examples.

        Args:
            batch: List of dictionaries from TranslationDataset

        Returns:
            Dictionary with padded tensors
        """
        result = {}

        # Collect text if present
        if "src_text" in batch[0]:
            result["src_text"] = [item["src_text"] for item in batch]
        if "tgt_text" in batch[0]:
            result["tgt_text"] = [item["tgt_text"] for item in batch]

        # Pad token sequences
        if "src" in batch[0]:
            src_sequences = [item["src"].tolist() for item in batch]
            result["src"] = pad_sequences(
                src_sequences,
                padding_value=self.pad_id,
                padding_side=self.padding_side,
            )
            # Create source padding mask (True where padded)
            result["src_mask"] = result["src"] == self.pad_id

        if "tgt" in batch[0]:
            tgt_sequences = [item["tgt"].tolist() for item in batch]
            result["tgt"] = pad_sequences(
                tgt_sequences,
                padding_value=self.pad_id,
                padding_side=self.padding_side,
            )
            # Create target padding mask
            result["tgt_mask"] = result["tgt"] == self.pad_id

        return result


def create_translation_dataloader(
    src_data: Union[str, List[str]],
    tgt_data: Union[str, List[str]],
    src_tokenizer: Optional[Any] = None,
    tgt_tokenizer: Optional[Any] = None,
    batch_size: int = 32,
    max_length: Optional[int] = None,
    shuffle: bool = True,
    num_workers: int = 0,
    pad_id: int = PAD_ID,
    add_bos: bool = True,
    add_eos: bool = True,
) -> DataLoader:
    """
    Create a DataLoader for translation data.

    Convenience function that creates dataset, collator, and dataloader.

    Args:
        src_data: Source sentences (list or file path)
        tgt_data: Target sentences (list or file path)
        src_tokenizer: Tokenizer for source language
        tgt_tokenizer: Tokenizer for target language
        batch_size: Batch size
        max_length: Maximum sequence length
        shuffle: Whether to shuffle data
        num_workers: Number of data loading workers
        pad_id: Padding token ID
        add_bos: Add BOS token
        add_eos: Add EOS token

    Returns:
        Configured DataLoader

    Example:
        >>> loader = create_translation_dataloader(
        ...     src_data="train.en",
        ...     tgt_data="train.de",
        ...     src_tokenizer=tokenizer,
        ...     tgt_tokenizer=tokenizer,
        ...     batch_size=64,
        ... )
    """
    dataset = TranslationDataset(
        src_data=src_data,
        tgt_data=tgt_data,
        src_tokenizer=src_tokenizer,
        tgt_tokenizer=tgt_tokenizer,
        max_length=max_length,
        add_bos=add_bos,
        add_eos=add_eos,
    )

    collator = TranslationCollator(pad_id=pad_id)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collator,
    )


# Optional: HuggingFace datasets integration
def load_wmt_dataset(
    name: str = "wmt14",
    language_pair: str = "de-en",
    split: str = "train",
    src_tokenizer: Optional[Any] = None,
    tgt_tokenizer: Optional[Any] = None,
    max_samples: Optional[int] = None,
) -> TranslationDataset:
    """
    Load WMT dataset from HuggingFace datasets.

    Requires the 'datasets' package to be installed.

    Args:
        name: Dataset name (e.g., "wmt14", "wmt16", "wmt17")
        language_pair: Language pair (e.g., "de-en", "fr-en")
        split: Data split ("train", "validation", "test")
        src_tokenizer: Source language tokenizer
        tgt_tokenizer: Target language tokenizer
        max_samples: Maximum number of samples to load

    Returns:
        TranslationDataset instance

    Example:
        >>> dataset = load_wmt_dataset(
        ...     name="wmt14",
        ...     language_pair="de-en",
        ...     split="train",
        ...     src_tokenizer=tokenizer,
        ...     tgt_tokenizer=tokenizer,
        ... )
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "HuggingFace datasets is required. Install with: pip install datasets"
        )

    # Parse language pair
    src_lang, tgt_lang = language_pair.split("-")

    # Load dataset
    dataset = load_dataset(name, language_pair, split=split)

    # Extract sentences
    src_sentences = []
    tgt_sentences = []

    for i, example in enumerate(dataset):
        if max_samples and i >= max_samples:
            break
        translation = example["translation"]
        src_sentences.append(translation[src_lang])
        tgt_sentences.append(translation[tgt_lang])

    return TranslationDataset(
        src_data=src_sentences,
        tgt_data=tgt_sentences,
        src_tokenizer=src_tokenizer,
        tgt_tokenizer=tgt_tokenizer,
    )


class SortedBatchSampler:
    """
    Batch sampler that groups similar-length sequences.

    This reduces padding overhead by ensuring sequences in a batch
    have similar lengths.

    Args:
        lengths: List of sequence lengths
        batch_size: Batch size
        shuffle: Whether to shuffle batches (not within batches)
        drop_last: Drop last incomplete batch

    Example:
        >>> lengths = [len(tokenizer.encode(s)) for s in sentences]
        >>> sampler = SortedBatchSampler(lengths, batch_size=32)
        >>> loader = DataLoader(dataset, batch_sampler=sampler)
    """

    def __init__(
        self,
        lengths: List[int],
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
    ):
        self.lengths = lengths
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

        # Sort indices by length
        self.sorted_indices = sorted(range(len(lengths)), key=lambda i: lengths[i])

    def __iter__(self) -> Iterator[List[int]]:
        # Create batches of similar-length sequences
        batches = []
        for i in range(0, len(self.sorted_indices), self.batch_size):
            batch = self.sorted_indices[i : i + self.batch_size]
            if len(batch) == self.batch_size or not self.drop_last:
                batches.append(batch)

        # Shuffle batches if requested
        if self.shuffle:
            import random

            random.shuffle(batches)

        yield from batches

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.sorted_indices) // self.batch_size
        return (len(self.sorted_indices) + self.batch_size - 1) // self.batch_size


class BucketIterator:
    """
    Iterator that creates batches with similar sequence lengths.

    More sophisticated version of SortedBatchSampler that works with
    the full dataset and creates token-based batches.

    Args:
        dataset: Translation dataset
        batch_size: Target batch size (in sentences)
        max_tokens: Maximum tokens per batch (overrides batch_size if set)
        shuffle: Shuffle data each epoch
        sort_within_batch: Sort sequences within each batch by length

    Example:
        >>> iterator = BucketIterator(dataset, batch_size=32)
        >>> for batch in iterator:
        ...     train_step(batch)
    """

    def __init__(
        self,
        dataset: TranslationDataset,
        batch_size: int = 32,
        max_tokens: Optional[int] = None,
        shuffle: bool = True,
        sort_within_batch: bool = True,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.max_tokens = max_tokens
        self.shuffle = shuffle
        self.sort_within_batch = sort_within_batch

        # Precompute lengths
        self.lengths = []
        for i in range(len(dataset)):
            item = dataset[i]
            src_len = len(item["src"]) if "src" in item else 0
            tgt_len = len(item["tgt"]) if "tgt" in item else 0
            self.lengths.append(max(src_len, tgt_len))

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        # Get indices sorted by length
        indices = sorted(range(len(self.dataset)), key=lambda i: self.lengths[i])

        if self.shuffle:
            # Shuffle within length buckets
            import random

            bucket_size = self.batch_size * 10
            for i in range(0, len(indices), bucket_size):
                bucket = indices[i : i + bucket_size]
                random.shuffle(bucket)
                indices[i : i + len(bucket)] = bucket

        # Create batches
        collator = TranslationCollator()
        batch = []

        for idx in indices:
            batch.append(self.dataset[idx])

            # Check if batch is full
            if len(batch) >= self.batch_size:
                if self.sort_within_batch:
                    batch.sort(
                        key=lambda x: len(x.get("src", [])),
                        reverse=True,
                    )
                yield collator(batch)
                batch = []

        # Yield remaining samples
        if batch:
            if self.sort_within_batch:
                batch.sort(
                    key=lambda x: len(x.get("src", [])),
                    reverse=True,
                )
            yield collator(batch)

    def __len__(self) -> int:
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size
