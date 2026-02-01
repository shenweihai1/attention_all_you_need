"""
Tokenization utilities for Transformer training.

This module provides tokenization functionality for the Transformer model,
wrapping SentencePiece for BPE (Byte Pair Encoding) tokenization as used
in the original "Attention Is All You Need" paper.

The tokenizer supports:
- BPE tokenization via SentencePiece
- Special tokens (PAD, UNK, BOS, EOS)
- Shared vocabulary for source and target languages
- Training from text files
- Encoding/decoding text sequences
"""

import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch

# Try to import sentencepiece, provide helpful error if not available
try:
    import sentencepiece as spm
    HAS_SENTENCEPIECE = True
except ImportError:
    HAS_SENTENCEPIECE = False


# Special token definitions
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<s>"
EOS_TOKEN = "</s>"

# Default special token IDs (SentencePiece convention)
PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3


class Tokenizer:
    """
    BPE tokenizer wrapper using SentencePiece.

    This tokenizer implements BPE (Byte Pair Encoding) tokenization as used
    in the original Transformer paper. It wraps SentencePiece for the actual
    tokenization logic.

    Args:
        model_path: Path to a trained SentencePiece model file (.model)
        vocab_size: Vocabulary size (only used when training a new model)
        model_type: SentencePiece model type ('bpe', 'unigram', 'word', 'char')

    Example:
        >>> # Load existing model
        >>> tokenizer = Tokenizer(model_path="tokenizer.model")
        >>> tokens = tokenizer.encode("Hello world")
        >>> text = tokenizer.decode(tokens)

        >>> # Train new model
        >>> tokenizer = Tokenizer.train(
        ...     input_files=["train.txt"],
        ...     model_prefix="my_tokenizer",
        ...     vocab_size=32000
        ... )
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        vocab_size: int = 32000,
        model_type: str = "bpe",
    ):
        if not HAS_SENTENCEPIECE:
            raise ImportError(
                "sentencepiece is required for tokenization. "
                "Install it with: pip install sentencepiece"
            )

        self.vocab_size = vocab_size
        self.model_type = model_type
        self._sp: Optional[spm.SentencePieceProcessor] = None

        if model_path is not None:
            self.load(model_path)

    def load(self, model_path: str) -> None:
        """
        Load a trained SentencePiece model.

        Args:
            model_path: Path to the .model file
        """
        model_path = str(model_path)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        self._sp = spm.SentencePieceProcessor()
        self._sp.Load(model_path)
        self.vocab_size = self._sp.GetPieceSize()

    @classmethod
    def train(
        cls,
        input_files: Union[str, List[str]],
        model_prefix: str,
        vocab_size: int = 32000,
        model_type: str = "bpe",
        character_coverage: float = 1.0,
        num_threads: int = 4,
        **kwargs,
    ) -> "Tokenizer":
        """
        Train a new SentencePiece model from text files.

        Args:
            input_files: Path(s) to training text file(s)
            model_prefix: Prefix for output model files (.model and .vocab)
            vocab_size: Target vocabulary size
            model_type: Model type ('bpe', 'unigram', 'word', 'char')
            character_coverage: Character coverage for CJK languages (default 1.0)
            num_threads: Number of training threads
            **kwargs: Additional SentencePiece training arguments

        Returns:
            Trained Tokenizer instance

        Example:
            >>> tokenizer = Tokenizer.train(
            ...     input_files=["train.en", "train.de"],
            ...     model_prefix="wmt_bpe",
            ...     vocab_size=37000,  # As used in the paper
            ... )
        """
        if not HAS_SENTENCEPIECE:
            raise ImportError(
                "sentencepiece is required for training. "
                "Install it with: pip install sentencepiece"
            )

        # Handle single file or list of files
        if isinstance(input_files, str):
            input_files = [input_files]

        # Validate input files exist
        for f in input_files:
            if not os.path.exists(f):
                raise FileNotFoundError(f"Input file not found: {f}")

        # Join files for training
        input_arg = ",".join(input_files)

        # Build training command
        train_args = {
            "input": input_arg,
            "model_prefix": model_prefix,
            "vocab_size": vocab_size,
            "model_type": model_type,
            "character_coverage": character_coverage,
            "num_threads": num_threads,
            "pad_id": PAD_ID,
            "unk_id": UNK_ID,
            "bos_id": BOS_ID,
            "eos_id": EOS_ID,
            "pad_piece": PAD_TOKEN,
            "unk_piece": UNK_TOKEN,
            "bos_piece": BOS_TOKEN,
            "eos_piece": EOS_TOKEN,
        }
        train_args.update(kwargs)

        # Train the model
        spm.SentencePieceTrainer.Train(**train_args)

        # Load and return the trained tokenizer
        model_path = f"{model_prefix}.model"
        return cls(model_path=model_path, vocab_size=vocab_size, model_type=model_type)

    @classmethod
    def train_from_texts(
        cls,
        texts: List[str],
        model_prefix: str,
        vocab_size: int = 32000,
        model_type: str = "bpe",
        **kwargs,
    ) -> "Tokenizer":
        """
        Train a tokenizer from a list of text strings.

        This is a convenience method that writes texts to a temporary file
        and trains a SentencePiece model.

        Args:
            texts: List of text strings to train on
            model_prefix: Prefix for output model files
            vocab_size: Target vocabulary size
            model_type: Model type
            **kwargs: Additional training arguments

        Returns:
            Trained Tokenizer instance
        """
        # Write texts to temporary file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            for text in texts:
                f.write(text.strip() + "\n")
            temp_path = f.name

        try:
            return cls.train(
                input_files=temp_path,
                model_prefix=model_prefix,
                vocab_size=vocab_size,
                model_type=model_type,
                **kwargs,
            )
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def _check_loaded(self) -> None:
        """Check that a model is loaded."""
        if self._sp is None:
            raise RuntimeError("No model loaded. Call load() or train() first.")

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> List[int]:
        """
        Encode text to token IDs.

        Args:
            text: Input text string
            add_bos: Add beginning-of-sequence token
            add_eos: Add end-of-sequence token

        Returns:
            List of token IDs
        """
        self._check_loaded()

        ids = self._sp.EncodeAsIds(text)

        if add_bos:
            ids = [BOS_ID] + ids
        if add_eos:
            ids = ids + [EOS_ID]

        return ids

    def decode(
        self,
        ids: List[int],
        skip_special_tokens: bool = True,
    ) -> str:
        """
        Decode token IDs to text.

        Args:
            ids: List of token IDs
            skip_special_tokens: Remove special tokens from output

        Returns:
            Decoded text string
        """
        self._check_loaded()

        if skip_special_tokens:
            ids = [i for i in ids if i not in (PAD_ID, BOS_ID, EOS_ID)]

        return self._sp.DecodeIds(ids)

    def encode_batch(
        self,
        texts: List[str],
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> List[List[int]]:
        """
        Encode multiple texts to token IDs.

        Args:
            texts: List of input text strings
            add_bos: Add beginning-of-sequence token
            add_eos: Add end-of-sequence token

        Returns:
            List of token ID lists
        """
        return [self.encode(text, add_bos=add_bos, add_eos=add_eos) for text in texts]

    def decode_batch(
        self,
        ids_batch: List[List[int]],
        skip_special_tokens: bool = True,
    ) -> List[str]:
        """
        Decode multiple token ID sequences to text.

        Args:
            ids_batch: List of token ID lists
            skip_special_tokens: Remove special tokens from output

        Returns:
            List of decoded text strings
        """
        return [self.decode(ids, skip_special_tokens=skip_special_tokens) for ids in ids_batch]

    def encode_as_pieces(self, text: str) -> List[str]:
        """
        Encode text to subword pieces (strings).

        Args:
            text: Input text string

        Returns:
            List of subword piece strings
        """
        self._check_loaded()
        return self._sp.EncodeAsPieces(text)

    def id_to_piece(self, id: int) -> str:
        """Convert token ID to piece string."""
        self._check_loaded()
        return self._sp.IdToPiece(id)

    def piece_to_id(self, piece: str) -> int:
        """Convert piece string to token ID."""
        self._check_loaded()
        return self._sp.PieceToId(piece)

    @property
    def pad_id(self) -> int:
        """Padding token ID."""
        return PAD_ID

    @property
    def unk_id(self) -> int:
        """Unknown token ID."""
        return UNK_ID

    @property
    def bos_id(self) -> int:
        """Beginning-of-sequence token ID."""
        return BOS_ID

    @property
    def eos_id(self) -> int:
        """End-of-sequence token ID."""
        return EOS_ID

    def __len__(self) -> int:
        """Return vocabulary size."""
        if self._sp is None:
            return self.vocab_size
        return self._sp.GetPieceSize()


class SimpleTokenizer:
    """
    Simple whitespace tokenizer for testing and simple use cases.

    This tokenizer splits text on whitespace and maintains a vocabulary
    mapping. It doesn't perform subword tokenization like BPE.

    Args:
        vocab: Optional vocabulary dictionary (word -> id)
        min_freq: Minimum frequency for a word to be included in vocabulary

    Example:
        >>> tokenizer = SimpleTokenizer()
        >>> tokenizer.build_vocab(["hello world", "hello there"])
        >>> tokens = tokenizer.encode("hello world")
    """

    def __init__(
        self,
        vocab: Optional[Dict[str, int]] = None,
        min_freq: int = 1,
    ):
        self.min_freq = min_freq

        # Initialize with special tokens
        self._word2id: Dict[str, int] = {
            PAD_TOKEN: PAD_ID,
            UNK_TOKEN: UNK_ID,
            BOS_TOKEN: BOS_ID,
            EOS_TOKEN: EOS_ID,
        }
        self._id2word: Dict[int, str] = {v: k for k, v in self._word2id.items()}
        self._next_id = 4

        if vocab is not None:
            for word, id in vocab.items():
                if word not in self._word2id:
                    self._word2id[word] = id
                    self._id2word[id] = word
                    self._next_id = max(self._next_id, id + 1)

    def build_vocab(
        self,
        texts: List[str],
        max_vocab_size: Optional[int] = None,
    ) -> None:
        """
        Build vocabulary from texts.

        Args:
            texts: List of text strings
            max_vocab_size: Maximum vocabulary size (including special tokens)
        """
        # Count word frequencies
        freq: Dict[str, int] = {}
        for text in texts:
            for word in text.strip().split():
                freq[word] = freq.get(word, 0) + 1

        # Filter by minimum frequency and sort by frequency
        words = [(w, f) for w, f in freq.items() if f >= self.min_freq]
        words.sort(key=lambda x: (-x[1], x[0]))

        # Apply max vocab size
        if max_vocab_size is not None:
            # Reserve 4 slots for special tokens
            words = words[: max_vocab_size - 4]

        # Add words to vocabulary
        for word, _ in words:
            if word not in self._word2id:
                self._word2id[word] = self._next_id
                self._id2word[self._next_id] = word
                self._next_id += 1

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> List[int]:
        """Encode text to token IDs."""
        ids = []
        if add_bos:
            ids.append(BOS_ID)

        for word in text.strip().split():
            ids.append(self._word2id.get(word, UNK_ID))

        if add_eos:
            ids.append(EOS_ID)

        return ids

    def decode(
        self,
        ids: List[int],
        skip_special_tokens: bool = True,
    ) -> str:
        """Decode token IDs to text."""
        words = []
        for id in ids:
            if skip_special_tokens and id in (PAD_ID, BOS_ID, EOS_ID):
                continue
            word = self._id2word.get(id, UNK_TOKEN)
            words.append(word)
        return " ".join(words)

    def encode_batch(
        self,
        texts: List[str],
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> List[List[int]]:
        """Encode multiple texts."""
        return [self.encode(text, add_bos=add_bos, add_eos=add_eos) for text in texts]

    def decode_batch(
        self,
        ids_batch: List[List[int]],
        skip_special_tokens: bool = True,
    ) -> List[str]:
        """Decode multiple token ID sequences."""
        return [self.decode(ids, skip_special_tokens=skip_special_tokens) for ids in ids_batch]

    @property
    def vocab_size(self) -> int:
        """Return vocabulary size."""
        return len(self._word2id)

    @property
    def pad_id(self) -> int:
        return PAD_ID

    @property
    def unk_id(self) -> int:
        return UNK_ID

    @property
    def bos_id(self) -> int:
        return BOS_ID

    @property
    def eos_id(self) -> int:
        return EOS_ID

    def __len__(self) -> int:
        return self.vocab_size


def pad_sequences(
    sequences: List[List[int]],
    padding_value: int = PAD_ID,
    max_length: Optional[int] = None,
    padding_side: str = "right",
) -> torch.Tensor:
    """
    Pad sequences to the same length.

    Args:
        sequences: List of token ID sequences
        padding_value: Value to use for padding
        max_length: Maximum length (uses longest sequence if None)
        padding_side: 'right' or 'left' padding

    Returns:
        Padded tensor of shape (batch_size, max_length)
    """
    if not sequences:
        return torch.tensor([], dtype=torch.long)

    # Determine max length
    lengths = [len(seq) for seq in sequences]
    if max_length is None:
        max_length = max(lengths)

    # Create padded tensor
    batch_size = len(sequences)
    padded = torch.full((batch_size, max_length), padding_value, dtype=torch.long)

    for i, seq in enumerate(sequences):
        length = min(len(seq), max_length)
        if padding_side == "right":
            padded[i, :length] = torch.tensor(seq[:length], dtype=torch.long)
        else:  # left
            padded[i, -length:] = torch.tensor(seq[:length], dtype=torch.long)

    return padded


def create_padding_mask_from_lengths(
    lengths: List[int],
    max_length: Optional[int] = None,
) -> torch.Tensor:
    """
    Create padding mask from sequence lengths.

    Args:
        lengths: List of sequence lengths
        max_length: Maximum sequence length

    Returns:
        Boolean mask of shape (batch_size, max_length) where True = padding
    """
    if max_length is None:
        max_length = max(lengths)

    batch_size = len(lengths)
    mask = torch.ones(batch_size, max_length, dtype=torch.bool)

    for i, length in enumerate(lengths):
        mask[i, :length] = False

    return mask
