"""Sharding plan for DenseTransformer.

Defines how DenseTransformer's module types are parallelized. Lives
alongside the model definition (model.py).

This file is imported by the registry at startup to register the
sharding functions. The model code itself has no awareness of parallelism.
"""

from nanigpt.distributed.plan import ParallelPlan
from nanigpt.distributed.registry import ResolvedSharding, register_sharding
from nanigpt.distributed.sharding import colwise, rowwise
from nanigpt.models.dense_transformer.model import FeedForward, MultiHeadAttention


@register_sharding(MultiHeadAttention)
def shard_attention(module: MultiHeadAttention, plan: ParallelPlan) -> ResolvedSharding | None:
    """Column-parallel Q/K/V, row-parallel output projection, halve head count."""
    if plan.tp_size <= 1:
        return None
    return ResolvedSharding(
        children={
            "q_proj": colwise("tp"),
            "k_proj": colwise("tp"),
            "v_proj": colwise("tp"),
            "out_proj": rowwise("tp"),
        },
        adjustments={"n_heads": module.n_heads // plan.tp_size},
    )


@register_sharding(FeedForward)
def shard_ffn(module: FeedForward, plan: ParallelPlan) -> ResolvedSharding | None:
    """Column-parallel up projection, row-parallel down projection."""
    if plan.tp_size <= 1:
        return None
    return ResolvedSharding(
        children={
            "up": colwise("tp"),
            "down": rowwise("tp"),
        },
    )
