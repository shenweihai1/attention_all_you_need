"""
Weight initialization utilities for the Transformer model.

This module implements weight initialization as described in
"Attention Is All You Need" (Vaswani et al., 2017) and common practices
from the reference implementation.
"""

import math

import torch
import torch.nn as nn


def init_transformer_weights(module: nn.Module, d_model: int = 512) -> None:
    """
    Initialize weights for Transformer modules.

    Follows the initialization strategy from the original Transformer implementation:
    - Linear layers: Xavier uniform initialization
    - Embeddings: Normal distribution with std = d_model^(-0.5)
    - LayerNorm: weight=1, bias=0 (PyTorch default)
    - Biases: zeros

    Args:
        module: The module to initialize
        d_model: Model dimension, used for embedding initialization scale
    """
    if isinstance(module, nn.Linear):
        # Xavier uniform initialization for linear layers
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

    elif isinstance(module, nn.Embedding):
        # Normal initialization with std = d_model^(-0.5)
        # This is the standard approach from tensor2tensor and fairseq
        nn.init.normal_(module.weight, mean=0.0, std=d_model ** -0.5)
        if module.padding_idx is not None:
            # Ensure padding embedding stays zero
            nn.init.zeros_(module.weight[module.padding_idx])

    elif isinstance(module, nn.LayerNorm):
        # LayerNorm: weight=1, bias=0 (PyTorch default, but explicit)
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


def init_bert_weights(module: nn.Module, std: float = 0.02) -> None:
    """
    Alternative initialization following BERT-style approach.

    Uses normal distribution with small std for all weights.

    Args:
        module: The module to initialize
        std: Standard deviation for normal initialization
    """
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=std)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=std)
        if module.padding_idx is not None:
            nn.init.zeros_(module.weight[module.padding_idx])

    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


class TransformerInitMixin:
    """
    Mixin class providing weight initialization for Transformer models.

    This mixin should be used with nn.Module subclasses that have
    a `d_model` attribute.
    """

    def init_weights(self, method: str = "xavier") -> None:
        """
        Initialize all weights in the model.

        Args:
            method: Initialization method. Options:
                - "xavier": Xavier uniform (default, from original paper)
                - "bert": Normal with std=0.02 (BERT-style)
        """
        d_model = getattr(self, "d_model", 512)

        if method == "xavier":
            self.apply(lambda m: init_transformer_weights(m, d_model=d_model))
        elif method == "bert":
            self.apply(init_bert_weights)
        else:
            raise ValueError(f"Unknown initialization method: {method}")

    def _reset_parameters(self) -> None:
        """
        Reset parameters using default Xavier initialization.

        This method can be called in __init__ for automatic initialization.
        """
        self.init_weights(method="xavier")


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """
    Count the number of parameters in a model.

    Args:
        model: The model to count parameters for
        trainable_only: If True, only count trainable parameters

    Returns:
        Number of parameters
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def get_parameter_stats(model: nn.Module) -> dict:
    """
    Get statistics about model parameters.

    Args:
        model: The model to analyze

    Returns:
        Dictionary with parameter statistics
    """
    stats = {
        "total_params": 0,
        "trainable_params": 0,
        "non_trainable_params": 0,
        "layer_stats": {},
    }

    for name, param in model.named_parameters():
        numel = param.numel()
        stats["total_params"] += numel

        if param.requires_grad:
            stats["trainable_params"] += numel
        else:
            stats["non_trainable_params"] += numel

        # Get layer type from name
        layer_type = name.split(".")[0]
        if layer_type not in stats["layer_stats"]:
            stats["layer_stats"][layer_type] = 0
        stats["layer_stats"][layer_type] += numel

    return stats
