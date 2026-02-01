"""
Tests for tokenization utilities.
"""

import os
import tempfile
from pathlib import Path

import pytest
import torch

from src.tokenizer import (
    Tokenizer,
    SimpleTokenizer,
    pad_sequences,
    create_padding_mask_from_lengths,
    PAD_ID,
    UNK_ID,
    BOS_ID,
    EOS_ID,
    PAD_TOKEN,
    UNK_TOKEN,
    BOS_TOKEN,
    EOS_TOKEN,
    HAS_SENTENCEPIECE,
)


class TestSpecialTokenConstants:
    """Tests for special token constants."""

    def test_special_token_ids(self):
        """Test that special token IDs are defined correctly."""
        assert PAD_ID == 0
        assert UNK_ID == 1
        assert BOS_ID == 2
        assert EOS_ID == 3

    def test_special_token_strings(self):
        """Test that special token strings are defined correctly."""
        assert PAD_TOKEN == "<pad>"
        assert UNK_TOKEN == "<unk>"
        assert BOS_TOKEN == "<s>"
        assert EOS_TOKEN == "</s>"


class TestSimpleTokenizer:
    """Tests for SimpleTokenizer class."""

    def test_creation(self):
        """Test tokenizer creation."""
        tokenizer = SimpleTokenizer()
        assert tokenizer.vocab_size == 4  # Special tokens only
        assert tokenizer.pad_id == PAD_ID
        assert tokenizer.unk_id == UNK_ID
        assert tokenizer.bos_id == BOS_ID
        assert tokenizer.eos_id == EOS_ID

    def test_build_vocab(self):
        """Test vocabulary building."""
        tokenizer = SimpleTokenizer()
        texts = ["hello world", "hello there", "world hello"]
        tokenizer.build_vocab(texts)

        # Should have special tokens + "hello", "world", "there"
        assert tokenizer.vocab_size == 7
        assert len(tokenizer) == 7

    def test_build_vocab_with_max_size(self):
        """Test vocabulary building with max size."""
        tokenizer = SimpleTokenizer()
        texts = ["a b c d e f g h i j"]
        tokenizer.build_vocab(texts, max_vocab_size=8)

        # 4 special tokens + 4 words = 8
        assert tokenizer.vocab_size == 8

    def test_build_vocab_with_min_freq(self):
        """Test vocabulary building with minimum frequency."""
        tokenizer = SimpleTokenizer(min_freq=2)
        texts = ["hello world", "hello there", "goodbye"]
        tokenizer.build_vocab(texts)

        # Only "hello" appears twice
        assert "hello" in tokenizer._word2id
        # "world", "there", "goodbye" appear only once
        assert tokenizer.vocab_size == 5  # 4 special + "hello"

    def test_encode_basic(self):
        """Test basic encoding."""
        tokenizer = SimpleTokenizer()
        tokenizer.build_vocab(["hello world"])

        ids = tokenizer.encode("hello world")
        assert len(ids) == 2
        assert all(isinstance(id, int) for id in ids)

    def test_encode_with_special_tokens(self):
        """Test encoding with BOS and EOS tokens."""
        tokenizer = SimpleTokenizer()
        tokenizer.build_vocab(["hello world"])

        ids = tokenizer.encode("hello world", add_bos=True, add_eos=True)
        assert ids[0] == BOS_ID
        assert ids[-1] == EOS_ID
        assert len(ids) == 4

    def test_encode_unknown_words(self):
        """Test encoding of unknown words."""
        tokenizer = SimpleTokenizer()
        tokenizer.build_vocab(["hello"])

        ids = tokenizer.encode("hello unknown")
        assert ids[1] == UNK_ID

    def test_decode_basic(self):
        """Test basic decoding."""
        tokenizer = SimpleTokenizer()
        tokenizer.build_vocab(["hello world"])

        ids = tokenizer.encode("hello world")
        text = tokenizer.decode(ids)
        assert text == "hello world"

    def test_decode_skip_special_tokens(self):
        """Test decoding with special token skipping."""
        tokenizer = SimpleTokenizer()
        tokenizer.build_vocab(["hello world"])

        ids = tokenizer.encode("hello world", add_bos=True, add_eos=True)

        # With skip_special_tokens=True (default)
        text = tokenizer.decode(ids)
        assert text == "hello world"

        # With skip_special_tokens=False
        text = tokenizer.decode(ids, skip_special_tokens=False)
        assert BOS_TOKEN in text or EOS_TOKEN in text

    def test_encode_batch(self):
        """Test batch encoding."""
        tokenizer = SimpleTokenizer()
        tokenizer.build_vocab(["hello world", "foo bar"])

        texts = ["hello world", "foo bar"]
        batch = tokenizer.encode_batch(texts)

        assert len(batch) == 2
        assert all(isinstance(ids, list) for ids in batch)

    def test_decode_batch(self):
        """Test batch decoding."""
        tokenizer = SimpleTokenizer()
        tokenizer.build_vocab(["hello world", "foo bar"])

        texts = ["hello world", "foo bar"]
        batch = tokenizer.encode_batch(texts)
        decoded = tokenizer.decode_batch(batch)

        assert decoded == texts

    def test_roundtrip(self):
        """Test encode-decode roundtrip."""
        tokenizer = SimpleTokenizer()
        texts = ["the quick brown fox", "jumps over the lazy dog"]
        tokenizer.build_vocab(texts)

        for text in texts:
            ids = tokenizer.encode(text)
            decoded = tokenizer.decode(ids)
            assert decoded == text

    def test_custom_vocab(self):
        """Test initialization with custom vocabulary."""
        vocab = {"hello": 4, "world": 5}
        tokenizer = SimpleTokenizer(vocab=vocab)

        assert tokenizer.vocab_size == 6  # 4 special + 2 custom
        assert tokenizer._word2id["hello"] == 4
        assert tokenizer._word2id["world"] == 5


@pytest.mark.skipif(not HAS_SENTENCEPIECE, reason="sentencepiece not installed")
class TestTokenizer:
    """Tests for Tokenizer class (requires sentencepiece)."""

    @pytest.fixture
    def sample_texts(self):
        """Sample texts for training."""
        return [
            "Hello, how are you?",
            "I am fine, thank you.",
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is interesting.",
            "Natural language processing with transformers.",
        ] * 100  # Repeat for enough training data

    @pytest.fixture
    def trained_tokenizer(self, sample_texts):
        """Create a trained tokenizer."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_prefix = os.path.join(tmpdir, "test_tokenizer")
            tokenizer = Tokenizer.train_from_texts(
                texts=sample_texts,
                model_prefix=model_prefix,
                vocab_size=100,
            )
            yield tokenizer

    def test_creation_without_model(self):
        """Test tokenizer creation without loading a model."""
        tokenizer = Tokenizer()
        assert tokenizer._sp is None

    def test_load_nonexistent_raises(self):
        """Test that loading nonexistent model raises error."""
        tokenizer = Tokenizer()
        with pytest.raises(FileNotFoundError):
            tokenizer.load("/nonexistent/path.model")

    def test_train_from_texts(self, sample_texts):
        """Test training from text list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_prefix = os.path.join(tmpdir, "test_tokenizer")
            tokenizer = Tokenizer.train_from_texts(
                texts=sample_texts,
                model_prefix=model_prefix,
                vocab_size=100,
            )

            assert tokenizer._sp is not None
            assert len(tokenizer) <= 100

    def test_train_from_file(self, sample_texts):
        """Test training from file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Write sample texts to file
            text_file = os.path.join(tmpdir, "train.txt")
            with open(text_file, "w") as f:
                for text in sample_texts:
                    f.write(text + "\n")

            model_prefix = os.path.join(tmpdir, "test_tokenizer")
            tokenizer = Tokenizer.train(
                input_files=text_file,
                model_prefix=model_prefix,
                vocab_size=100,
            )

            assert tokenizer._sp is not None

    def test_encode_decode(self, trained_tokenizer):
        """Test encoding and decoding."""
        text = "Hello, how are you?"
        ids = trained_tokenizer.encode(text)

        assert isinstance(ids, list)
        assert all(isinstance(id, int) for id in ids)

        decoded = trained_tokenizer.decode(ids)
        # Note: SentencePiece may normalize whitespace/punctuation
        assert len(decoded) > 0

    def test_encode_with_special_tokens(self, trained_tokenizer):
        """Test encoding with special tokens."""
        text = "Hello world"
        ids = trained_tokenizer.encode(text, add_bos=True, add_eos=True)

        assert ids[0] == BOS_ID
        assert ids[-1] == EOS_ID

    def test_encode_as_pieces(self, trained_tokenizer):
        """Test encoding to subword pieces."""
        text = "Hello world"
        pieces = trained_tokenizer.encode_as_pieces(text)

        assert isinstance(pieces, list)
        assert all(isinstance(p, str) for p in pieces)

    def test_id_piece_conversion(self, trained_tokenizer):
        """Test ID to piece and piece to ID conversion."""
        # Get a valid ID
        text = "Hello"
        ids = trained_tokenizer.encode(text)
        if ids:
            id = ids[0]
            piece = trained_tokenizer.id_to_piece(id)
            back_id = trained_tokenizer.piece_to_id(piece)
            assert back_id == id

    def test_special_token_properties(self, trained_tokenizer):
        """Test special token properties."""
        assert trained_tokenizer.pad_id == PAD_ID
        assert trained_tokenizer.unk_id == UNK_ID
        assert trained_tokenizer.bos_id == BOS_ID
        assert trained_tokenizer.eos_id == EOS_ID

    def test_batch_encode_decode(self, trained_tokenizer):
        """Test batch encoding and decoding."""
        texts = ["Hello world", "How are you?"]
        batch = trained_tokenizer.encode_batch(texts)

        assert len(batch) == 2
        assert all(isinstance(ids, list) for ids in batch)

        decoded = trained_tokenizer.decode_batch(batch)
        assert len(decoded) == 2

    def test_operations_without_model_raises(self):
        """Test that operations without loaded model raise error."""
        tokenizer = Tokenizer()

        with pytest.raises(RuntimeError, match="No model loaded"):
            tokenizer.encode("test")

        with pytest.raises(RuntimeError, match="No model loaded"):
            tokenizer.decode([1, 2, 3])


class TestPadSequences:
    """Tests for pad_sequences function."""

    def test_basic_padding(self):
        """Test basic sequence padding."""
        sequences = [[1, 2, 3], [4, 5], [6]]
        padded = pad_sequences(sequences)

        assert padded.shape == (3, 3)
        assert padded[0].tolist() == [1, 2, 3]
        assert padded[1].tolist() == [4, 5, 0]
        assert padded[2].tolist() == [6, 0, 0]

    def test_custom_padding_value(self):
        """Test padding with custom value."""
        sequences = [[1, 2], [3]]
        padded = pad_sequences(sequences, padding_value=99)

        assert padded[1, 1] == 99

    def test_max_length(self):
        """Test padding with max_length."""
        sequences = [[1, 2, 3, 4, 5], [6, 7]]
        padded = pad_sequences(sequences, max_length=3)

        assert padded.shape == (2, 3)
        assert padded[0].tolist() == [1, 2, 3]  # Truncated
        assert padded[1].tolist() == [6, 7, 0]

    def test_left_padding(self):
        """Test left-side padding."""
        sequences = [[1, 2, 3], [4, 5]]
        padded = pad_sequences(sequences, padding_side="left")

        assert padded[1].tolist() == [0, 4, 5]

    def test_empty_sequences(self):
        """Test with empty input."""
        padded = pad_sequences([])
        assert padded.shape == (0,)

    def test_single_sequence(self):
        """Test with single sequence."""
        sequences = [[1, 2, 3]]
        padded = pad_sequences(sequences)

        assert padded.shape == (1, 3)
        assert padded[0].tolist() == [1, 2, 3]

    def test_dtype(self):
        """Test output dtype."""
        sequences = [[1, 2], [3, 4]]
        padded = pad_sequences(sequences)

        assert padded.dtype == torch.long


class TestCreatePaddingMaskFromLengths:
    """Tests for create_padding_mask_from_lengths function."""

    def test_basic_mask(self):
        """Test basic mask creation."""
        lengths = [3, 2, 1]
        mask = create_padding_mask_from_lengths(lengths, max_length=4)

        assert mask.shape == (3, 4)
        # First sequence: 3 valid, 1 padding
        assert mask[0].tolist() == [False, False, False, True]
        # Second sequence: 2 valid, 2 padding
        assert mask[1].tolist() == [False, False, True, True]
        # Third sequence: 1 valid, 3 padding
        assert mask[2].tolist() == [False, True, True, True]

    def test_auto_max_length(self):
        """Test automatic max_length determination."""
        lengths = [3, 5, 2]
        mask = create_padding_mask_from_lengths(lengths)

        assert mask.shape == (3, 5)  # Max length is 5

    def test_no_padding(self):
        """Test when all sequences are max length."""
        lengths = [4, 4, 4]
        mask = create_padding_mask_from_lengths(lengths, max_length=4)

        # No padding needed
        assert not mask.any()

    def test_dtype(self):
        """Test output dtype."""
        lengths = [2, 3]
        mask = create_padding_mask_from_lengths(lengths)

        assert mask.dtype == torch.bool


class TestTokenizerEdgeCases:
    """Tests for edge cases."""

    def test_empty_text(self):
        """Test encoding empty text."""
        tokenizer = SimpleTokenizer()
        tokenizer.build_vocab(["hello"])

        ids = tokenizer.encode("")
        assert ids == []

    def test_whitespace_only(self):
        """Test encoding whitespace-only text."""
        tokenizer = SimpleTokenizer()
        tokenizer.build_vocab(["hello"])

        ids = tokenizer.encode("   ")
        assert ids == []

    def test_decode_empty(self):
        """Test decoding empty list."""
        tokenizer = SimpleTokenizer()
        tokenizer.build_vocab(["hello"])

        text = tokenizer.decode([])
        assert text == ""

    def test_decode_only_special(self):
        """Test decoding only special tokens."""
        tokenizer = SimpleTokenizer()
        tokenizer.build_vocab(["hello"])

        text = tokenizer.decode([PAD_ID, BOS_ID, EOS_ID])
        assert text == ""

    def test_special_chars_in_text(self):
        """Test text with special characters."""
        tokenizer = SimpleTokenizer()
        tokenizer.build_vocab(["hello!", "world?"])

        # Note: SimpleTokenizer splits on whitespace, so punctuation stays with words
        ids = tokenizer.encode("hello! world?")
        decoded = tokenizer.decode(ids)
        assert decoded == "hello! world?"


class TestTokenizerIntegration:
    """Integration tests."""

    def test_with_pad_sequences(self):
        """Test tokenizer output with pad_sequences."""
        tokenizer = SimpleTokenizer()
        tokenizer.build_vocab(["hello world", "hi there everyone"])

        texts = ["hello world", "hi"]
        batch = tokenizer.encode_batch(texts, add_eos=True)
        padded = pad_sequences(batch, padding_value=tokenizer.pad_id)

        assert padded.shape[0] == 2
        # Shorter sequence should have padding
        assert (padded[1] == tokenizer.pad_id).any()

    def test_with_mask_creation(self):
        """Test tokenizer output with mask creation."""
        tokenizer = SimpleTokenizer()
        tokenizer.build_vocab(["hello world", "hi there everyone"])

        texts = ["hello world", "hi"]
        batch = tokenizer.encode_batch(texts)
        lengths = [len(ids) for ids in batch]
        mask = create_padding_mask_from_lengths(lengths)

        # Mask shape should match batch structure
        assert mask.shape[0] == len(texts)
