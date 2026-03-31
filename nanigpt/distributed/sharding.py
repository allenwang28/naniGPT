"""Sharding descriptors: pure data types for expressing how modules are parallelized.

This module defines placement types and the ModuleSharding dataclass that
describe how a module's parameters, inputs, and outputs are distributed
across mesh dimensions. These are planning-only data structures — they
describe intent, not execution. No torch.distributed, no DeviceMesh, no
process groups.

See docs/specs/distributed/ for how this fits into the broader parallelism
architecture (description → resolution → execution).

Placement types
---------------
We define our own Shard and Replicate rather than using DTensor's
(torch.distributed._tensor.Shard/Replicate). DTensor's placements are
tied to the DTensor runtime and carry machinery for redistribution,
dispatch, and gradient hooks. Our placements are frozen dataclasses with
no behavior — they describe a sharding, they don't implement one. This
keeps the description layer dependency-free and usable in contexts where
torch.distributed isn't initialized (meta-device dry runs, cost model
sweeps, config validation).

If DTensor's placement types stabilize in a future PyTorch release, the
migration is mechanical: both use Shard(dim=N) and Replicate(), so it's
a swap of imports.

    Shard(dim=0)  — tensor is split along dimension 0 across a mesh dim
    Replicate()   — tensor is identical on all ranks in a mesh dim

ModuleSharding
--------------
Ties placements to a module's parameters, inputs, and outputs on named
mesh dimensions. Factory functions produce the standard TP patterns:

    colwise("tp"):  weight Shard(0), bias Shard(0)  — split output dim
    rowwise("tp"):  weight Shard(1), no bias shard  — split input dim

These are composable across mesh dimensions. A weight sharded on both TP
and EP would have placements {tp: Shard(0), ep: Shard(0)}.

compute_local_shape() applies placements to a global shape to get the
per-rank shape. Used by both the real sharding path and the meta-device
dry run.
"""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Shard:
    """Tensor is split along this dimension across a mesh dimension."""

    dim: int


@dataclass(frozen=True, slots=True)
class Replicate:
    """Tensor is identical on all ranks in a mesh dimension."""

    pass


Placement = Shard | Replicate


@dataclass(frozen=True, slots=True)
class ModuleSharding:
    """Describes how a single nn.Linear is sharded across mesh dimensions.

    params: maps param name (e.g. "weight", "bias") to a dict of
        {mesh_dim_name: Placement}. Only sharded params need entries —
        unmentioned params are implicitly replicated.

    input_placements: {mesh_dim_name: Placement} describing the expected
        input distribution. For colwise, input is Replicate (after
        enter_parallel_region). For rowwise, input is Shard (the output
        of the preceding colwise layer).

    output_placements: {mesh_dim_name: Placement} describing the output
        distribution. For colwise, output is Shard (varying across ranks).
        For rowwise, output is Replicate (after exit_parallel_region).
    """

    params: dict[str, dict[str, Placement]]
    input_placements: dict[str, Placement]
    output_placements: dict[str, Placement]


def colwise(mesh_dim: str = "tp") -> ModuleSharding:
    """Column-parallel sharding: split output dim across mesh_dim.

    Weight [out, in] → Shard(0) splits the output dimension.
    Bias [out] → Shard(0) splits correspondingly.
    Input is Replicate (enter_parallel_region makes it so).
    Output is Shard (each rank has different output features).
    """
    return ModuleSharding(
        params={
            "weight": {mesh_dim: Shard(dim=0)},
            "bias": {mesh_dim: Shard(dim=0)},
        },
        input_placements={mesh_dim: Replicate()},
        output_placements={mesh_dim: Shard(dim=0)},
    )


def rowwise(mesh_dim: str = "tp") -> ModuleSharding:
    """Row-parallel sharding: split input dim across mesh_dim.

    Weight [out, in] → Shard(1) splits the input dimension.
    Bias is not sharded — it's added after all-reduce.
    Input is Shard (receives the varying output of a colwise layer).
    Output is Replicate (exit_parallel_region all-reduces).
    """
    return ModuleSharding(
        params={
            "weight": {mesh_dim: Shard(dim=1)},
        },
        input_placements={mesh_dim: Shard(dim=0)},
        output_placements={mesh_dim: Replicate()},
    )


def compute_local_shape(
    global_shape: tuple[int, ...],
    placements: dict[str, Placement],
    mesh_degrees: dict[str, int],
) -> tuple[int, ...]:
    """Compute the per-rank tensor shape after applying placements.

    For each mesh dimension that has a Shard placement, the corresponding
    tensor dimension is divided by the mesh degree. Replicate placements
    leave the shape unchanged.

    Args:
        global_shape: The full (unsharded) tensor shape.
        placements: {mesh_dim_name: Placement} for this tensor.
        mesh_degrees: {mesh_dim_name: degree} for the mesh.

    Returns:
        The local per-rank shape.

    Raises:
        ValueError: If a sharded dimension is not divisible by the degree.
    """
    local = list(global_shape)
    for mesh_dim, placement in placements.items():
        degree = mesh_degrees.get(mesh_dim, 1)
        if degree == 1:
            continue
        if isinstance(placement, Shard):
            if local[placement.dim] % degree != 0:
                raise ValueError(
                    f"Dimension {placement.dim} (size {local[placement.dim]}) "
                    f"not divisible by {mesh_dim} degree {degree}"
                )
            local[placement.dim] //= degree
    return tuple(local)
