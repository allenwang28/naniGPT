"""Dense transformer model package.

Re-exports all public names from model.py so that existing imports
(``from nanigpt.models.dense_transformer import DenseTransformer``)
continue to work unchanged.

Layout follows the torchtitan per-model convention:
    model.py        — architecture, config, presets
    parallelize.py  — sharding specs for distributed training

As the model grows, model.py can be further split (e.g. config.py for
TransformerConfig + presets, components.py for sub-modules) and the
re-exports here updated accordingly.
"""

from nanigpt.models.dense_transformer.model import (
    MODEL_PRESETS,
    DenseTransformer,
    FeedForward,
    ModelOutput,
    MultiHeadAttention,
    TransformerBlock,
    TransformerConfig,
)

__all__ = [
    "MODEL_PRESETS",
    "DenseTransformer",
    "FeedForward",
    "ModelOutput",
    "MultiHeadAttention",
    "TransformerBlock",
    "TransformerConfig",
]
