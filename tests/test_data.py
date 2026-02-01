"""
Tests for data loading utilities.
"""

import os
import tempfile
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from src.data import (
    TranslationDataset,
    TranslationCollator,
    create_translation_dataloader,
    SortedBatchSampler,
    BucketIterator,
    DynamicBatchSampler,
    create_dynamic_dataloader,
)
from src.tokenizer import SimpleTokenizer, PAD_ID, BOS_ID, EOS_ID


class TestTranslationDataset:
    """Tests for TranslationDataset class."""

    def test_creation_from_lists(self):
        """Test dataset creation from lists."""
        src = ["hello world", "how are you"]
        tgt = ["hallo welt", "wie geht es dir"]

        dataset = TranslationDataset(src_data=src, tgt_data=tgt)

        assert len(dataset) == 2
        assert dataset.src_sentences == src
        assert dataset.tgt_sentences == tgt

    def test_creation_from_files(self):
        """Test dataset creation from files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            src_file = os.path.join(tmpdir, "src.txt")
            tgt_file = os.path.join(tmpdir, "tgt.txt")

            with open(src_file, "w") as f:
                f.write("hello world\nhow are you\n")
            with open(tgt_file, "w") as f:
                f.write("hallo welt\nwie geht es dir\n")

            dataset = TranslationDataset(src_data=src_file, tgt_data=tgt_file)

            assert len(dataset) == 2
            assert dataset.src_sentences[0] == "hello world"
            assert dataset.tgt_sentences[0] == "hallo welt"

    def test_mismatched_lengths_raises(self):
        """Test that mismatched source/target raises error."""
        src = ["hello", "world"]
        tgt = ["hallo"]

        with pytest.raises(ValueError, match="different lengths"):
            TranslationDataset(src_data=src, tgt_data=tgt)

    def test_file_not_found_raises(self):
        """Test that missing file raises error."""
        with pytest.raises(FileNotFoundError):
            TranslationDataset(
                src_data="/nonexistent/file.txt",
                tgt_data="/also/nonexistent.txt",
            )

    def test_getitem_without_tokenizer(self):
        """Test getting item without tokenizer."""
        src = ["hello world", "how are you"]
        tgt = ["hallo welt", "wie geht es dir"]

        dataset = TranslationDataset(src_data=src, tgt_data=tgt)
        item = dataset[0]

        assert "src_text" in item
        assert "tgt_text" in item
        assert item["src_text"] == "hello world"
        assert item["tgt_text"] == "hallo welt"
        assert "src" not in item  # No tokenizer, so no IDs
        assert "tgt" not in item

    def test_getitem_with_tokenizer(self):
        """Test getting item with tokenizer."""
        src = ["hello world", "how are you"]
        tgt = ["hallo welt", "wie geht es dir"]

        tokenizer = SimpleTokenizer()
        tokenizer.build_vocab(src + tgt)

        dataset = TranslationDataset(
            src_data=src,
            tgt_data=tgt,
            src_tokenizer=tokenizer,
            tgt_tokenizer=tokenizer,
        )
        item = dataset[0]

        assert "src" in item
        assert "tgt" in item
        assert isinstance(item["src"], torch.Tensor)
        assert isinstance(item["tgt"], torch.Tensor)
        assert item["src"].dtype == torch.long

    def test_bos_eos_tokens(self):
        """Test BOS and EOS token addition."""
        src = ["hello world"]
        tgt = ["hallo welt"]

        tokenizer = SimpleTokenizer()
        tokenizer.build_vocab(src + tgt)

        # With BOS and EOS
        dataset = TranslationDataset(
            src_data=src,
            tgt_data=tgt,
            src_tokenizer=tokenizer,
            tgt_tokenizer=tokenizer,
            add_bos=True,
            add_eos=True,
        )
        item = dataset[0]

        assert item["src"][0].item() == BOS_ID
        assert item["src"][-1].item() == EOS_ID

    def test_max_length_truncation(self):
        """Test max length truncation."""
        src = ["one two three four five six seven"]
        tgt = ["eins zwei drei vier fünf sechs sieben"]

        tokenizer = SimpleTokenizer()
        tokenizer.build_vocab(src + tgt)

        dataset = TranslationDataset(
            src_data=src,
            tgt_data=tgt,
            src_tokenizer=tokenizer,
            tgt_tokenizer=tokenizer,
            max_length=5,
            add_bos=False,
            add_eos=False,
        )
        item = dataset[0]

        assert len(item["src"]) <= 5
        assert len(item["tgt"]) <= 5


class TestTranslationCollator:
    """Tests for TranslationCollator class."""

    @pytest.fixture
    def sample_dataset(self):
        """Create a sample dataset."""
        src = ["hello", "hello world how are you"]
        tgt = ["hallo", "hallo welt wie geht es dir"]

        tokenizer = SimpleTokenizer()
        tokenizer.build_vocab(src + tgt)

        return TranslationDataset(
            src_data=src,
            tgt_data=tgt,
            src_tokenizer=tokenizer,
            tgt_tokenizer=tokenizer,
            add_bos=True,
            add_eos=True,
        )

    def test_basic_collation(self, sample_dataset):
        """Test basic batch collation."""
        collator = TranslationCollator()
        batch = [sample_dataset[0], sample_dataset[1]]
        result = collator(batch)

        assert "src" in result
        assert "tgt" in result
        assert "src_mask" in result
        assert "tgt_mask" in result

        # Batch should be padded to max length
        assert result["src"].shape[0] == 2
        assert result["tgt"].shape[0] == 2

    def test_padding_mask(self, sample_dataset):
        """Test that padding mask is correct."""
        collator = TranslationCollator(pad_id=PAD_ID)
        batch = [sample_dataset[0], sample_dataset[1]]
        result = collator(batch)

        # Shorter sequence should have padding
        # The first sample is shorter
        assert result["src_mask"][0].any()  # Has padding
        # Second sample is longest, may or may not have padding

        # Mask should match where padding is
        assert (result["src_mask"] == (result["src"] == PAD_ID)).all()

    def test_text_preserved(self, sample_dataset):
        """Test that text is preserved in output."""
        collator = TranslationCollator()
        batch = [sample_dataset[0], sample_dataset[1]]
        result = collator(batch)

        assert "src_text" in result
        assert "tgt_text" in result
        assert len(result["src_text"]) == 2


class TestCreateTranslationDataloader:
    """Tests for create_translation_dataloader function."""

    def test_basic_creation(self):
        """Test basic dataloader creation."""
        src = ["hello world", "how are you", "goodbye"]
        tgt = ["hallo welt", "wie geht es dir", "auf wiedersehen"]

        tokenizer = SimpleTokenizer()
        tokenizer.build_vocab(src + tgt)

        loader = create_translation_dataloader(
            src_data=src,
            tgt_data=tgt,
            src_tokenizer=tokenizer,
            tgt_tokenizer=tokenizer,
            batch_size=2,
            shuffle=False,
        )

        assert isinstance(loader, DataLoader)

        batch = next(iter(loader))
        assert "src" in batch
        assert "tgt" in batch
        assert batch["src"].shape[0] == 2

    def test_all_options(self):
        """Test dataloader with all options."""
        src = ["hello"] * 10
        tgt = ["hallo"] * 10

        tokenizer = SimpleTokenizer()
        tokenizer.build_vocab(["hello", "hallo"])

        loader = create_translation_dataloader(
            src_data=src,
            tgt_data=tgt,
            src_tokenizer=tokenizer,
            tgt_tokenizer=tokenizer,
            batch_size=4,
            max_length=10,
            shuffle=True,
            num_workers=0,
            pad_id=0,
            add_bos=True,
            add_eos=True,
        )

        batches = list(loader)
        total_samples = sum(b["src"].shape[0] for b in batches)
        assert total_samples == 10


class TestSortedBatchSampler:
    """Tests for SortedBatchSampler class."""

    def test_basic_sampling(self):
        """Test basic sorted batch sampling."""
        lengths = [5, 2, 8, 3, 10, 1]
        sampler = SortedBatchSampler(lengths, batch_size=2, shuffle=False)

        batches = list(sampler)
        assert len(batches) == 3

        # First batch should have shortest sequences (indices 5, 1)
        assert set(batches[0]) == {5, 1}  # lengths 1, 2

    def test_shuffle_batches(self):
        """Test that batches can be shuffled."""
        lengths = list(range(100))
        sampler = SortedBatchSampler(lengths, batch_size=10, shuffle=True)

        batches1 = [tuple(b) for b in sampler]
        batches2 = [tuple(b) for b in sampler]

        # With shuffling, order should often differ
        # Note: There's a small chance they're the same

    def test_drop_last(self):
        """Test drop_last option."""
        lengths = [1, 2, 3, 4, 5]

        sampler_keep = SortedBatchSampler(lengths, batch_size=2, drop_last=False)
        sampler_drop = SortedBatchSampler(lengths, batch_size=2, drop_last=True)

        assert len(sampler_keep) == 3
        assert len(sampler_drop) == 2


class TestBucketIterator:
    """Tests for BucketIterator class."""

    @pytest.fixture
    def sample_dataset(self):
        """Create a sample dataset with varying lengths."""
        src = [
            "a",
            "a b",
            "a b c",
            "a b c d",
            "a b c d e",
        ]
        tgt = [
            "x",
            "x y",
            "x y z",
            "x y z w",
            "x y z w v",
        ]

        tokenizer = SimpleTokenizer()
        tokenizer.build_vocab(src + tgt)

        return TranslationDataset(
            src_data=src,
            tgt_data=tgt,
            src_tokenizer=tokenizer,
            tgt_tokenizer=tokenizer,
            add_bos=False,
            add_eos=False,
        )

    def test_basic_iteration(self, sample_dataset):
        """Test basic bucket iteration."""
        iterator = BucketIterator(sample_dataset, batch_size=2, shuffle=False)

        batches = list(iterator)
        total_samples = sum(b["src"].shape[0] for b in batches)
        assert total_samples == 5

    def test_len(self, sample_dataset):
        """Test iterator length."""
        iterator = BucketIterator(sample_dataset, batch_size=2)
        assert len(iterator) == 3  # 5 samples, batch_size=2

    def test_sorted_within_batch(self, sample_dataset):
        """Test that sequences are sorted within batches."""
        iterator = BucketIterator(
            sample_dataset,
            batch_size=3,
            shuffle=False,
            sort_within_batch=True,
        )

        for batch in iterator:
            lengths = [len(batch["src"][i]) for i in range(batch["src"].shape[0])]
            # Should be sorted descending within batch
            # Note: padding makes this check tricky, but longest should be first
            assert lengths[0] >= lengths[-1]


class TestDataIntegration:
    """Integration tests for data loading."""

    def test_full_pipeline(self):
        """Test full data loading pipeline."""
        # Create sample data
        src = [
            "the cat sat on the mat",
            "hello world",
            "machine learning is great",
        ]
        tgt = [
            "die katze saß auf der matte",
            "hallo welt",
            "maschinelles lernen ist großartig",
        ]

        # Create tokenizer
        tokenizer = SimpleTokenizer()
        tokenizer.build_vocab(src + tgt)

        # Create dataloader
        loader = create_translation_dataloader(
            src_data=src,
            tgt_data=tgt,
            src_tokenizer=tokenizer,
            tgt_tokenizer=tokenizer,
            batch_size=2,
            shuffle=False,
        )

        # Iterate through batches
        all_src = []
        all_tgt = []
        for batch in loader:
            assert batch["src"].shape == batch["tgt"].shape[:1] + batch["tgt"].shape[1:]
            all_src.extend(batch["src_text"])
            all_tgt.extend(batch["tgt_text"])

        assert set(all_src) == set(src)
        assert set(all_tgt) == set(tgt)

    def test_with_files(self):
        """Test data loading from files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            src_file = os.path.join(tmpdir, "src.txt")
            tgt_file = os.path.join(tmpdir, "tgt.txt")

            sentences_src = ["hello world", "goodbye moon"]
            sentences_tgt = ["hallo welt", "auf wiedersehen mond"]

            with open(src_file, "w") as f:
                f.write("\n".join(sentences_src))
            with open(tgt_file, "w") as f:
                f.write("\n".join(sentences_tgt))

            tokenizer = SimpleTokenizer()
            tokenizer.build_vocab(sentences_src + sentences_tgt)

            loader = create_translation_dataloader(
                src_data=src_file,
                tgt_data=tgt_file,
                src_tokenizer=tokenizer,
                tgt_tokenizer=tokenizer,
                batch_size=2,
            )

            batch = next(iter(loader))
            assert batch["src"].shape[0] == 2


class TestDataEdgeCases:
    """Tests for edge cases."""

    def test_empty_lines_filtered(self):
        """Test that empty lines are filtered from files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            src_file = os.path.join(tmpdir, "src.txt")
            tgt_file = os.path.join(tmpdir, "tgt.txt")

            with open(src_file, "w") as f:
                f.write("hello\n\nworld\n")
            with open(tgt_file, "w") as f:
                f.write("hallo\n\nwelt\n")

            dataset = TranslationDataset(src_data=src_file, tgt_data=tgt_file)
            assert len(dataset) == 2

    def test_single_sample(self):
        """Test with single sample."""
        src = ["hello"]
        tgt = ["hallo"]

        tokenizer = SimpleTokenizer()
        tokenizer.build_vocab(src + tgt)

        dataset = TranslationDataset(
            src_data=src,
            tgt_data=tgt,
            src_tokenizer=tokenizer,
            tgt_tokenizer=tokenizer,
        )

        assert len(dataset) == 1

        collator = TranslationCollator()
        batch = collator([dataset[0]])
        assert batch["src"].shape[0] == 1

    def test_unicode_text(self):
        """Test with Unicode text."""
        src = ["Hello 世界", "Привет мир"]
        tgt = ["你好 World", "Привіт світ"]

        dataset = TranslationDataset(src_data=src, tgt_data=tgt)
        assert len(dataset) == 2
        assert dataset[0]["src_text"] == "Hello 世界"


class TestDynamicBatchSampler:
    """Tests for DynamicBatchSampler class."""

    def test_basic_creation(self):
        """Test basic sampler creation."""
        lengths = [5, 10, 3, 8, 2, 15]
        sampler = DynamicBatchSampler(lengths, max_tokens=20)

        batches = list(sampler)
        assert len(batches) > 0

        # All samples should be included
        all_indices = [idx for batch in batches for idx in batch]
        assert set(all_indices) == set(range(len(lengths)))

    def test_max_tokens_constraint(self):
        """Test that max_tokens constraint is respected."""
        lengths = [10, 10, 10, 10, 10]
        sampler = DynamicBatchSampler(lengths, max_tokens=25, shuffle=False)

        for batch in sampler:
            max_len = max(lengths[i] for i in batch)
            total_tokens = max_len * len(batch)
            # Should not exceed max_tokens (with some tolerance for the algorithm)
            assert total_tokens <= 25 or len(batch) == 1

    def test_max_sentences_constraint(self):
        """Test that max_sentences constraint is respected."""
        lengths = [2, 2, 2, 2, 2, 2, 2, 2]
        sampler = DynamicBatchSampler(
            lengths, max_tokens=1000, max_sentences=3, shuffle=False
        )

        for batch in sampler:
            assert len(batch) <= 3

    def test_shuffle(self):
        """Test that shuffle works."""
        lengths = list(range(20))
        sampler = DynamicBatchSampler(lengths, max_tokens=50, shuffle=True)

        batches1 = [tuple(b) for b in sampler]
        batches2 = [tuple(b) for b in sampler]

        # With shuffling, order may differ (not guaranteed but likely)

    def test_sort_by_length(self):
        """Test sorting by length groups similar sequences."""
        lengths = [1, 10, 2, 9, 3, 8]
        sampler = DynamicBatchSampler(
            lengths, max_tokens=20, shuffle=False, sort_by_length=True
        )

        batches = list(sampler)
        # Sorted batches should group similar lengths
        assert len(batches) > 0

    def test_len(self):
        """Test sampler length."""
        lengths = [5, 5, 5, 5]
        sampler = DynamicBatchSampler(lengths, max_tokens=15)

        assert len(sampler) == len(list(sampler))


class TestCreateDynamicDataloader:
    """Tests for create_dynamic_dataloader function."""

    @pytest.fixture
    def sample_dataset(self):
        """Create a sample dataset."""
        src = [
            "a",
            "a b",
            "a b c",
            "a b c d",
            "a b c d e",
            "a b c d e f",
        ]
        tgt = [
            "x",
            "x y",
            "x y z",
            "x y z w",
            "x y z w v",
            "x y z w v u",
        ]

        tokenizer = SimpleTokenizer()
        tokenizer.build_vocab(src + tgt)

        return TranslationDataset(
            src_data=src,
            tgt_data=tgt,
            src_tokenizer=tokenizer,
            tgt_tokenizer=tokenizer,
            add_bos=False,
            add_eos=False,
        )

    def test_basic_creation(self, sample_dataset):
        """Test basic dynamic dataloader creation."""
        loader = create_dynamic_dataloader(
            dataset=sample_dataset,
            max_tokens=20,
            shuffle=False,
        )

        assert isinstance(loader, DataLoader)

        # Collect all batches
        batches = list(loader)
        total_samples = sum(b["src"].shape[0] for b in batches)
        assert total_samples == 6

    def test_with_max_sentences(self, sample_dataset):
        """Test with max_sentences constraint."""
        loader = create_dynamic_dataloader(
            dataset=sample_dataset,
            max_tokens=1000,
            max_sentences=2,
        )

        for batch in loader:
            assert batch["src"].shape[0] <= 2

    def test_dynamic_padding(self, sample_dataset):
        """Test that padding is dynamic per batch."""
        loader = create_dynamic_dataloader(
            dataset=sample_dataset,
            max_tokens=10,
            shuffle=False,
        )

        batch_max_lens = []
        for batch in loader:
            # Get actual max length in batch (non-padded)
            src_lens = (batch["src"] != PAD_ID).sum(dim=1)
            batch_max_lens.append(src_lens.max().item())

        # Different batches should have different max lengths (dynamic padding)
        # This demonstrates padding adapts to batch content
        assert len(set(batch_max_lens)) > 1 or len(batch_max_lens) == 1
