"""Tests for the type-based sharding registry (nanigpt/distributed/registry.py).

All tests are CPU-only — uses meta device for model construction, no GPU
or process groups required.
"""

import torch
import torch.nn as nn

from nanigpt.distributed.plan import ParallelPlan
from nanigpt.distributed.registry import (
    ResolvedSharding,
    get_sharding_fn,
    resolve_sharding,
)
from nanigpt.distributed.sharding import Replicate, Shard, colwise, rowwise
from nanigpt.models.dense_transformer import (
    DenseTransformer,
    FeedForward,
    MultiHeadAttention,
    TransformerConfig,
)


# ---- Registry lookup ----


def test_attention_registered():
    assert get_sharding_fn(MultiHeadAttention) is not None


def test_ffn_registered():
    assert get_sharding_fn(FeedForward) is not None


def test_unregistered_type():
    assert get_sharding_fn(nn.LayerNorm) is None


# ---- Sharding functions ----


def test_attention_sharding_tp2():
    with torch.device("meta"):
        config = TransformerConfig(
            vocab_size=256, d_model=64, n_heads=4, n_layers=1, d_ff=128, max_seq_len=32
        )
        attn = MultiHeadAttention(config)

    plan = ParallelPlan(tp_size=2)
    fn = get_sharding_fn(MultiHeadAttention)
    result = fn(attn, plan)

    assert result is not None
    assert set(result.children.keys()) == {"q_proj", "k_proj", "v_proj", "out_proj"}
    assert result.children["q_proj"] == colwise("tp")
    assert result.children["out_proj"] == rowwise("tp")
    assert result.adjustments == {"n_heads": 2}


def test_attention_sharding_tp1_returns_none():
    with torch.device("meta"):
        config = TransformerConfig(
            vocab_size=256, d_model=64, n_heads=4, n_layers=1, d_ff=128, max_seq_len=32
        )
        attn = MultiHeadAttention(config)

    plan = ParallelPlan(tp_size=1)
    fn = get_sharding_fn(MultiHeadAttention)
    result = fn(attn, plan)
    assert result is None


def test_ffn_sharding_tp2():
    with torch.device("meta"):
        config = TransformerConfig(
            vocab_size=256, d_model=64, n_heads=4, n_layers=1, d_ff=128, max_seq_len=32
        )
        ffn = FeedForward(config)

    plan = ParallelPlan(tp_size=2)
    fn = get_sharding_fn(FeedForward)
    result = fn(ffn, plan)

    assert result is not None
    assert set(result.children.keys()) == {"up", "down"}
    assert result.children["up"] == colwise("tp")
    assert result.children["down"] == rowwise("tp")


# ---- resolve_sharding on full model ----


def test_resolve_sharding_full_model_tp2():
    with torch.device("meta"):
        config = TransformerConfig(
            vocab_size=256, d_model=64, n_heads=4, n_layers=2, d_ff=128, max_seq_len=32
        )
        model = DenseTransformer(config)

    plan = ParallelPlan(tp_size=2)
    sharding_plan = resolve_sharding(model, plan)

    # 2 layers × 6 linears per layer = 12 entries
    assert len(sharding_plan.entries) == 12

    # Check that entries cover expected module names
    fqns = {
        f"{e.parent_fqn}.{e.child_name}" if e.parent_fqn else e.child_name
        for e in sharding_plan.entries
    }
    assert "blocks.0.attn.q_proj" in fqns
    assert "blocks.0.attn.out_proj" in fqns
    assert "blocks.0.ffn.up" in fqns
    assert "blocks.0.ffn.down" in fqns
    assert "blocks.1.attn.q_proj" in fqns
    assert "blocks.1.ffn.down" in fqns

    # Check adjustments for attention modules
    assert "blocks.0.attn" in sharding_plan.adjustments
    assert sharding_plan.adjustments["blocks.0.attn"]["n_heads"] == 2
    assert "blocks.1.attn" in sharding_plan.adjustments


def test_resolve_sharding_tp1_returns_empty():
    with torch.device("meta"):
        config = TransformerConfig(
            vocab_size=256, d_model=64, n_heads=4, n_layers=1, d_ff=128, max_seq_len=32
        )
        model = DenseTransformer(config)

    plan = ParallelPlan(tp_size=1)
    sharding_plan = resolve_sharding(model, plan)

    assert len(sharding_plan.entries) == 0
    assert len(sharding_plan.adjustments) == 0
