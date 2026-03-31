"""Tests for tensor parallelism (nanigpt/distributed/tensor_parallel.py).

Unit tests verify the registry-based sharding plan and weight sharding shapes.
Multi-GPU tests verify numerical equivalence between single-GPU and TP=2
forward/backward.
"""

import torch

from nanigpt.distributed.plan import ParallelPlan
from nanigpt.distributed.registry import resolve_sharding
from nanigpt.distributed.sharding import colwise, rowwise
from nanigpt.models.dense_transformer import DenseTransformer, TransformerConfig
from tests.distributed_utils import distributed_test

# ---- Unit tests (no GPU required) ----


def test_tp_plan_covers_all_linear_layers():
    """The registry should produce sharding entries for all expected linear layers."""
    with torch.device("meta"):
        config = TransformerConfig(
            vocab_size=256, d_model=64, n_heads=4, n_layers=1, d_ff=128, max_seq_len=32
        )
        model = DenseTransformer(config)

    plan = ParallelPlan(tp_size=2)
    sharding_plan = resolve_sharding(model, plan)

    fqns = {
        f"{e.parent_fqn}.{e.child_name}" if e.parent_fqn else e.child_name
        for e in sharding_plan.entries
    }
    expected = {
        "blocks.0.attn.q_proj",
        "blocks.0.attn.k_proj",
        "blocks.0.attn.v_proj",
        "blocks.0.attn.out_proj",
        "blocks.0.ffn.up",
        "blocks.0.ffn.down",
    }
    assert expected == fqns


def test_tp_plan_colwise_rowwise_assignment():
    """Colwise for Q/K/V/up, rowwise for out_proj/down."""
    with torch.device("meta"):
        config = TransformerConfig(
            vocab_size=256, d_model=64, n_heads=4, n_layers=1, d_ff=128, max_seq_len=32
        )
        model = DenseTransformer(config)

    plan = ParallelPlan(tp_size=2)
    sharding_plan = resolve_sharding(model, plan)

    by_child = {e.child_name: e.sharding for e in sharding_plan.entries}
    assert by_child["q_proj"] == colwise("tp")
    assert by_child["k_proj"] == colwise("tp")
    assert by_child["v_proj"] == colwise("tp")
    assert by_child["out_proj"] == rowwise("tp")
    assert by_child["up"] == colwise("tp")
    assert by_child["down"] == rowwise("tp")


# ---- Multi-GPU tests ----


@distributed_test(world_size=2)
def test_colwise_shard_shapes(rank, world_size):
    """Column-parallel sharding should split output dim by tp_size."""
    import torch.nn as nn

    from nanigpt.distributed.tensor_parallel import _shard_linear_colwise

    group = __import__("torch").distributed.group.WORLD
    linear = nn.Linear(64, 128, bias=True, device="cuda")

    _shard_linear_colwise(linear, rank, world_size, group)

    assert linear.weight.shape == (64, 64), f"Expected (64, 64), got {linear.weight.shape}"
    assert linear.bias.shape == (64,), f"Expected (64,), got {linear.bias.shape}"
    assert linear.out_features == 64


@distributed_test(world_size=2)
def test_rowwise_shard_shapes(rank, world_size):
    """Row-parallel sharding should split input dim by tp_size."""
    import torch.nn as nn

    from nanigpt.distributed.tensor_parallel import _shard_linear_rowwise

    group = __import__("torch").distributed.group.WORLD
    linear = nn.Linear(128, 64, bias=True, device="cuda")

    _shard_linear_rowwise(linear, rank, world_size, group)

    assert linear.weight.shape == (64, 64), f"Expected (64, 64), got {linear.weight.shape}"
    assert linear.in_features == 64


@distributed_test(world_size=2)
def test_apply_tp_shards_all_modules(rank, world_size):
    """apply_tensor_parallelism should shard all modules in the TP plan."""
    from torch.distributed.device_mesh import init_device_mesh

    from nanigpt.distributed.tensor_parallel import apply_tensor_parallelism
    from nanigpt.models.dense_transformer import DenseTransformer, TransformerConfig

    config = TransformerConfig(
        vocab_size=256,
        d_model=64,
        n_heads=4,
        n_layers=2,
        d_ff=128,
        max_seq_len=32,
        dropout=0.0,
    )
    model = DenseTransformer(config).cuda()

    mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("tp",))
    plan = ParallelPlan(tp_size=world_size)
    apply_tensor_parallelism(model, plan, mesh)

    for block in model.blocks:
        # Colwise: output dim halved
        assert block.attn.q_proj.out_features == 32  # 64 // 2
        assert block.attn.k_proj.out_features == 32
        assert block.attn.v_proj.out_features == 32
        assert block.ffn.up.out_features == 64  # 128 // 2
        # Rowwise: input dim halved
        assert block.attn.out_proj.in_features == 32  # 64 // 2
        assert block.ffn.down.in_features == 64  # 128 // 2
        # Attention heads adjusted
        assert block.attn.n_heads == 2  # 4 // 2


@distributed_test(world_size=2)
def test_tp_numerical_equivalence(rank, world_size):
    """TP=2 should produce the same loss as single-GPU (within tolerance).

    This is the most important test: build a model, run forward+backward on
    a single GPU, then build the same model with TP=2 and compare.
    """
    import torch
    import torch.nn.functional as F
    from torch.distributed.device_mesh import init_device_mesh

    from nanigpt.distributed.tensor_parallel import apply_tensor_parallelism
    from nanigpt.models.dense_transformer import DenseTransformer, TransformerConfig

    config = TransformerConfig(
        vocab_size=256,
        d_model=64,
        n_heads=4,
        n_layers=2,
        d_ff=128,
        max_seq_len=32,
        dropout=0.0,
    )

    # Deterministic input
    torch.manual_seed(123)
    input_ids = torch.randint(0, 256, (2, 16), device="cuda")
    targets = torch.randint(0, 256, (2, 16), device="cuda")

    # ---- Single-GPU reference (all ranks compute the same thing) ----
    torch.manual_seed(42)
    ref_model = DenseTransformer(config).cuda()
    ref_output = ref_model(input_ids)
    ref_loss = F.cross_entropy(
        ref_output.logits[:, :-1].reshape(-1, config.vocab_size),
        targets[:, 1:].reshape(-1),
    )

    # ---- TP model ----
    torch.manual_seed(42)
    tp_model = DenseTransformer(config).cuda()

    mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("tp",))
    plan = ParallelPlan(tp_size=world_size)
    apply_tensor_parallelism(tp_model, plan, mesh)

    tp_output = tp_model(input_ids)
    tp_loss = F.cross_entropy(
        tp_output.logits[:, :-1].reshape(-1, config.vocab_size),
        targets[:, 1:].reshape(-1),
    )

    assert torch.allclose(tp_loss, ref_loss, rtol=1e-4, atol=1e-4), (
        f"TP loss ({tp_loss.item():.6f}) != reference loss ({ref_loss.item():.6f})"
    )


@distributed_test(world_size=2)
def test_tp_backward_runs(rank, world_size):
    """TP model backward should complete without errors and produce gradients."""
    import torch
    import torch.nn.functional as F
    from torch.distributed.device_mesh import init_device_mesh

    from nanigpt.distributed.tensor_parallel import apply_tensor_parallelism
    from nanigpt.models.dense_transformer import DenseTransformer, TransformerConfig

    torch.manual_seed(42)

    config = TransformerConfig(
        vocab_size=256,
        d_model=64,
        n_heads=4,
        n_layers=2,
        d_ff=128,
        max_seq_len=32,
        dropout=0.0,
    )

    model = DenseTransformer(config).cuda()
    mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("tp",))
    plan = ParallelPlan(tp_size=world_size)
    apply_tensor_parallelism(model, plan, mesh)

    torch.manual_seed(123)
    input_ids = torch.randint(0, 256, (2, 16), device="cuda")
    targets = torch.randint(0, 256, (2, 16), device="cuda")

    output = model(input_ids)
    loss = F.cross_entropy(
        output.logits[:, :-1].reshape(-1, config.vocab_size),
        targets[:, 1:].reshape(-1),
    )
    loss.backward()

    for name, param in model.named_parameters():
        assert param.grad is not None, f"Parameter {name} has no gradient"
        assert not torch.isnan(param.grad).any(), f"Parameter {name} has NaN gradient"
