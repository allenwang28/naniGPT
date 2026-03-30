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
"""

import logging

import torch
from torch.distributed.device_mesh import DeviceMesh

log = logging.getLogger("distributed.mesh")


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
        return DeviceMesh("cuda", list(range(world_size)), mesh_dim_names=("dp_shard",))

    if len(dims) == 1:
        name, size = dims[0]
        return DeviceMesh("cuda", list(range(size)), mesh_dim_names=(name,))

    # Multi-dimensional mesh
    names = tuple(name for name, _ in dims)
    shape = tuple(size for _, size in dims)
    mesh_tensor = torch.arange(world_size).reshape(*shape)
    log.info(f"DeviceMesh: shape={shape}, dims={names}")
    return DeviceMesh("cuda", mesh_tensor, mesh_dim_names=names)
