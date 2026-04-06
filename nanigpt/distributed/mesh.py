"""DeviceMesh creation.

Created once in the training loop, passed explicitly everywhere.
Adding a new parallelism dimension means adding a mesh dimension here.

Rank layout (innermost = fastest interconnect):
    [dp_replicate, dp_shard, tp]

For tp=2, dp_shard=2, dp_replicate=2 on 8 GPUs:
    mesh shape: (2, 2, 2)
    dim names: ("dp_replicate", "dp_shard", "tp")
    TP groups (NVLink):        {0,1}, {2,3}, {4,5}, {6,7}
    FSDP shard groups:         {0,2}, {1,3}, {4,6}, {5,7}
    DDP replicate groups:      {0,4}, {1,5}, {2,6}, {3,7}

Only dimensions with degree > 1 are included in the mesh. This keeps
1D and 2D cases simple (no degenerate size-1 dimensions to deal with).

Process groups are created directly as NaniProcessGroups (wrapping raw
NCCL groups from dist.new_group) and passed to DeviceMesh.from_group().
This avoids creating throwaway groups — the same pattern as torchcomms'
integration with torchtitan: substitute process groups below the
DeviceMesh boundary, everything above is unaware.
"""

import logging

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.distributed_c10d import (
    _register_process_group,
    _unregister_process_group,
)

from nanigpt.distributed.process_group import NaniProcessGroup

log = logging.getLogger("distributed.mesh")


def _create_dim_group(
    mesh_tensor: torch.Tensor,
    dim: int,
    dim_name: str,
) -> NaniProcessGroup:
    """Create a NaniProcessGroup for one mesh dimension.

    For a given dimension, computes all rank groups (by moving that dim
    to the last axis and flattening), calls dist.new_group() for each,
    and wraps the current rank's group in NaniProcessGroup.
    """
    # Each row is a group of ranks that communicate along this dimension.
    # E.g. for a (2,2,2) mesh and dim=2 (tp): [[0,1],[2,3],[4,5],[6,7]]
    groups_2d = mesh_tensor.movedim(dim, -1).reshape(-1, mesh_tensor.shape[dim])

    my_rank = dist.get_rank()
    my_group = None

    for row in groups_2d:
        ranks = row.tolist()
        pg = dist.new_group(ranks=ranks)
        if my_rank in ranks:
            my_group = pg

    if my_group is None:
        raise RuntimeError(f"Rank {my_rank} not found in any {dim_name} group")

    wrapper = NaniProcessGroup(my_group, mesh_dim=dim_name)
    # Replace the raw PG in the C++ registry with our wrapper so that
    # _resolve_process_group() returns NaniProcessGroup. This is the same
    # mechanism torchcomms uses to substitute process groups.
    group_name = wrapper.group_name
    _unregister_process_group(group_name)
    _register_process_group(group_name, wrapper)

    # Mirror the raw PG's Python-side registration so FSDP/DDP can find us.
    # torchcomms does the same six-dict mirroring in _create_torchcomm_process_group
    # (torchcomms/device_mesh.py) — this is the established pattern for substituting
    # custom PGs below the DeviceMesh boundary.
    world = dist.distributed_c10d._world
    if my_group in world.pg_map:
        world.pg_map[wrapper] = world.pg_map.pop(my_group)
    if my_group in world.pg_names:
        world.pg_names[wrapper] = world.pg_names.pop(my_group)
    if my_group in world.pg_group_ranks:
        world.pg_group_ranks[wrapper] = world.pg_group_ranks.pop(my_group)
    if my_group in world.pg_backend_config:
        world.pg_backend_config[wrapper] = world.pg_backend_config.pop(my_group)
    pg_tag = f"ptd:{group_name}"
    if pg_tag in world.tags_to_pg:
        world.tags_to_pg[pg_tag] = [
            wrapper if pg is my_group else pg for pg in world.tags_to_pg[pg_tag]
        ]
    if my_group in world.pg_to_tag:
        world.pg_to_tag[wrapper] = world.pg_to_tag.pop(my_group)

    return wrapper


def get_process_group(mesh: DeviceMesh, dim_name: str) -> ProcessGroup:
    """Return the ProcessGroup for a mesh dimension from _pg_registry.

    DeviceMesh.get_group() bypasses _pg_registry at runtime and resolves
    via C++, returning the raw inner PG instead of our NaniProcessGroup
    wrapper. This function reads from _pg_registry directly so that
    NaniProcessGroup overrides (e.g. comm timing) are active.

    Falls back to mesh.get_group() if the registry entry is not available.
    """
    root = mesh._get_root_mesh()
    dim_idx = mesh._get_mesh_dim_by_name(dim_name)
    group_name = mesh._dim_group_names[dim_idx]
    pg = root._pg_registry.get(group_name)
    if pg is not None:
        return pg
    # Fallback for meshes not created via create_device_mesh
    return mesh[dim_name].get_group()


def create_device_mesh(
    world_size: int,
    *,
    dp_replicate: int = 1,
    dp_shard: int = 1,
    tp_size: int = 1,
) -> DeviceMesh:
    """Create a DeviceMesh from parallelism degrees.

    Only active dimensions (degree > 1) are included. If all degrees are 1,
    returns a 1D mesh named "dp_shard" (single-GPU fallback).

    Dimension ordering: dp_replicate (outermost) → dp_shard → tp (innermost).
    TP is innermost so adjacent ranks share NVLink.

    All process groups are NaniProcessGroups for comm profiling and future FT.
    """
    total = dp_replicate * dp_shard * tp_size
    if total != world_size:
        raise ValueError(
            f"dp_replicate ({dp_replicate}) × dp_shard ({dp_shard}) "
            f"× tp_size ({tp_size}) = {total}, but world_size = {world_size}"
        )

    # Build list of (name, size) for active dimensions only
    dims: list[tuple[str, int]] = []
    if dp_replicate > 1:
        dims.append(("dp_replicate", dp_replicate))
    if dp_shard > 1:
        dims.append(("dp_shard", dp_shard))
    if tp_size > 1:
        dims.append(("tp", tp_size))

    # Fallback: single-GPU or pure DP with no sharding
    if not dims:
        dims = [("dp_shard", world_size)]

    names = tuple(name for name, _ in dims)
    shape = tuple(size for _, size in dims)
    mesh_tensor = torch.arange(world_size).reshape(*shape)

    if len(dims) > 1:
        log.info(f"DeviceMesh: shape={shape}, dims={names}")

    # Create wrapped process groups directly — no throwaway DeviceMesh
    wrapped_groups = [
        _create_dim_group(mesh_tensor, dim_idx, name) for dim_idx, name in enumerate(names)
    ]

    return DeviceMesh.from_group(
        wrapped_groups,
        "cuda",
        mesh=mesh_tensor,
        mesh_dim_names=names,
    )
