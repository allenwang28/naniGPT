"""Tests for tensor parallelism (nanigpt/distributed/tensor_parallel.py).

Unit tests verify weight sharding shapes and the TP plan. Multi-GPU tests
verify numerical equivalence between single-GPU and TP=2 forward/backward.
"""

from nanigpt.distributed.tensor_parallel import DEFAULT_TP_STYLE
from tests.distributed_utils import distributed_test

# ---- Unit tests (no GPU required) ----


def test_default_tp_style_keys():
    """DEFAULT_TP_STYLE should cover all expected linear layers."""
    expected_keys = {
        "attn.q_proj",
        "attn.k_proj",
        "attn.v_proj",
        "attn.out_proj",
        "ffn.up",
        "ffn.down",
    }
    assert set(DEFAULT_TP_STYLE.keys()) == expected_keys


def test_default_tp_style_values():
    """Colwise for Q/K/V/up, rowwise for out_proj/down."""
    assert DEFAULT_TP_STYLE["attn.q_proj"] == "colwise"
    assert DEFAULT_TP_STYLE["attn.k_proj"] == "colwise"
    assert DEFAULT_TP_STYLE["attn.v_proj"] == "colwise"
    assert DEFAULT_TP_STYLE["attn.out_proj"] == "rowwise"
    assert DEFAULT_TP_STYLE["ffn.up"] == "colwise"
    assert DEFAULT_TP_STYLE["ffn.down"] == "rowwise"


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
    apply_tensor_parallelism(model, mesh)

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
    apply_tensor_parallelism(tp_model, mesh)

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
    apply_tensor_parallelism(model, mesh)

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
