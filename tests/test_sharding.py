"""Tests for sharding descriptors (nanigpt/distributed/sharding.py).

All tests are CPU-only — no GPU or process groups required.
"""

from nanigpt.distributed.sharding import (
    ModuleSharding,
    Replicate,
    Shard,
    colwise,
    compute_local_shape,
    rowwise,
)


# ---- Placement types ----


def test_shard_equality():
    assert Shard(0) == Shard(0)
    assert Shard(0) != Shard(1)


def test_replicate_equality():
    assert Replicate() == Replicate()


def test_shard_is_not_replicate():
    assert Shard(0) != Replicate()


# ---- Factory functions ----


def test_colwise_default_mesh_dim():
    ms = colwise()
    assert ms.params["weight"] == {"tp": Shard(dim=0)}
    assert ms.params["bias"] == {"tp": Shard(dim=0)}
    assert ms.input_placements == {"tp": Replicate()}
    assert ms.output_placements == {"tp": Shard(dim=0)}


def test_colwise_custom_mesh_dim():
    ms = colwise("etp")
    assert "etp" in ms.params["weight"]
    assert "etp" in ms.input_placements


def test_rowwise_default_mesh_dim():
    ms = rowwise()
    assert ms.params["weight"] == {"tp": Shard(dim=1)}
    assert "bias" not in ms.params
    assert ms.input_placements == {"tp": Shard(dim=0)}
    assert ms.output_placements == {"tp": Replicate()}


def test_module_sharding_is_frozen():
    ms = colwise()
    try:
        ms.params = {}
        assert False, "Should have raised FrozenInstanceError"
    except AttributeError:
        pass


# ---- compute_local_shape ----


def test_compute_local_shape_colwise():
    # Weight [128, 64] with Shard(0) on tp=2 → [64, 64]
    local = compute_local_shape(
        global_shape=(128, 64),
        placements={"tp": Shard(dim=0)},
        mesh_degrees={"tp": 2},
    )
    assert local == (64, 64)


def test_compute_local_shape_rowwise():
    # Weight [64, 128] with Shard(1) on tp=2 → [64, 64]
    local = compute_local_shape(
        global_shape=(64, 128),
        placements={"tp": Shard(dim=1)},
        mesh_degrees={"tp": 2},
    )
    assert local == (64, 64)


def test_compute_local_shape_replicated():
    # Replicate leaves shape unchanged
    local = compute_local_shape(
        global_shape=(128, 64),
        placements={"tp": Replicate()},
        mesh_degrees={"tp": 2},
    )
    assert local == (128, 64)


def test_compute_local_shape_degree_one():
    # Degree 1 leaves shape unchanged even with Shard
    local = compute_local_shape(
        global_shape=(128, 64),
        placements={"tp": Shard(dim=0)},
        mesh_degrees={"tp": 1},
    )
    assert local == (128, 64)


def test_compute_local_shape_unknown_mesh_dim():
    # Unknown mesh dim is treated as degree 1
    local = compute_local_shape(
        global_shape=(128, 64),
        placements={"ep": Shard(dim=0)},
        mesh_degrees={"tp": 2},
    )
    assert local == (128, 64)


def test_compute_local_shape_multi_dim():
    # Shard on two mesh dimensions
    local = compute_local_shape(
        global_shape=(128, 64),
        placements={"tp": Shard(dim=0), "ep": Shard(dim=0)},
        mesh_degrees={"tp": 2, "ep": 4},
    )
    assert local == (16, 64)  # 128 // 2 // 4


def test_compute_local_shape_not_divisible():
    import pytest

    with pytest.raises(ValueError, match="not divisible"):
        compute_local_shape(
            global_shape=(127, 64),
            placements={"tp": Shard(dim=0)},
            mesh_degrees={"tp": 2},
        )
