"""Distributed runtime: process groups, mesh, and model wrapping.

This package owns the entire distributed concern — from process group
lifecycle to parallelism application.

Parallelism composition
-----------------------
Each parallelism dimension is an independent degree. Set them individually
in ParallelConfig and they compose via the DeviceMesh. The product of all
degrees must equal num_workers.

    | Dimension     | What it does                        | Communication        |
    |---------------|-------------------------------------|----------------------|
    | dp_replicate  | Gradient all-reduce (DDP)           | all-reduce           |
    | dp_shard      | Parameter sharding (FSDP)           | all-gather + RS      |
    | tp            | Weight sharding within a layer      | all-reduce per layer |

Data parallelism mode is implicit from the degree values:

    dp_shard>1, dp_replicate=1  → pure FSDP
    dp_replicate>1, dp_shard=1  → pure DDP
    dp_replicate>1, dp_shard>1  → HSDP (FSDP within shard group, DDP across)

dp_shard=-1 (default) auto-fills with remaining ranks after other dimensions.

Examples (8 GPUs):
    ParallelConfig(num_workers=8)                         → FSDP(8)
    ParallelConfig(dp_replicate=8, dp_shard=1, nw=8)      → DDP(8)
    ParallelConfig(dp_replicate=2, nw=8)                  → HSDP: FSDP(4) × DDP(2)
    ParallelConfig(tp_size=2, nw=8)                       → FSDP(4) × TP(2)
    ParallelConfig(dp_replicate=2, tp_size=2, nw=8)       → HSDP: FSDP(2) × DDP(2) × TP(2)

Mesh layout: [dp_replicate, dp_shard, tp] — TP innermost (NVLink), DP outermost.

Application order:
    1. TP (shard weights, replace forwards)
    2. FSDP (parameter sharding — includes HSDP when dp_replicate>1)
    3. DDP (gradient all-reduce)
Each step operates on the model produced by the previous one.
"""

import logging

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh

from nanigpt.config import ParallelConfig
from nanigpt.distributed.data_parallel import apply_ddp, apply_fsdp
from nanigpt.distributed.mesh import create_device_mesh
from nanigpt.distributed.plan import ParallelPlan
from nanigpt.env import MASTER_ADDR, MASTER_PORT

__all__ = [
    "apply_parallelism",
    "cleanup_distributed",
    "create_device_mesh",
    "get_rank",
    "get_world_size",
    "init_distributed",
    "is_rank_zero",
]

log = logging.getLogger("distributed")


# ---- Process group lifecycle ----


def init_distributed(rank: int, world_size: int, backend: str = "nccl") -> None:
    """Initialize the default process group and set CUDA device.

    Expects MASTER_ADDR and MASTER_PORT to already be set in the environment.
    Each rank should see a single GPU via CUDA_VISIBLE_DEVICES (Ray handles this).
    """
    if rank == 0:
        log.info(f"MASTER_ADDR={MASTER_ADDR.get_value()}, MASTER_PORT={MASTER_PORT.get_value()}")

    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(0)  # Each worker sees only its own GPU
    log.info(f"Rank {rank}/{world_size} initialized (backend={backend})")


def cleanup_distributed() -> None:
    """Destroy the default process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_rank_zero() -> bool:
    """Return True if this is rank 0 or if distributed is not initialized."""
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def get_rank() -> int:
    """Return current rank, or 0 if not distributed."""
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_world_size() -> int:
    """Return world size, or 1 if not distributed."""
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


# ---- Parallelism application ----


def apply_parallelism(
    model: torch.nn.Module,
    world_size: int,
    config: ParallelConfig,
) -> torch.nn.Module:
    """Create a DeviceMesh and apply the parallelism plan to a model.

    This is the single entry point for train.py — it handles mesh creation
    and strategy dispatch so the training loop doesn't need to know the details.

    Application order (matches docstring at top of module):
        1. TP — shard weights, replace forwards
        2. FSDP — parameter sharding (includes HSDP when dp_replicate>1)
        3. DDP — gradient all-reduce
    """
    plan = ParallelPlan(
        dp_replicate=config.dp_replicate,
        dp_shard=config.dp_shard,
        tp_size=config.tp_size,
    )

    mesh = create_device_mesh(
        world_size,
        dp_replicate=plan.dp_replicate,
        dp_shard=plan.dp_shard,
        tp_size=plan.tp_size,
    )

    # 1. TP: shard weights, replace forwards
    if plan.tp_size > 1:
        from nanigpt.distributed.tensor_parallel import apply_tensor_parallelism

        apply_tensor_parallelism(model, plan, mesh)

    # 2. FSDP: parameter sharding (2D mesh for HSDP when dp_replicate>1)
    if plan.dp_shard > 1:
        apply_fsdp(model, plan, mesh)

    # 3. DDP: gradient all-reduce (only when no FSDP — HSDP handles replicate via FSDP's 2D mesh)
    if plan.dp_replicate > 1 and plan.dp_shard <= 1:
        if plan.tp_size > 1:
            raise ValueError(
                "DDP + TP is not supported — use FSDP + TP instead (dp_shard > 1). "
                "FSDP gives the same gradient sync with better memory efficiency."
            )
        apply_ddp(model, plan, mesh)

    if config.comm_timing:
        enable_comm_timing(mesh)

    return model


def enable_comm_timing(mesh: DeviceMesh) -> None:
    """Enable CUDA event timing on all NaniProcessGroups in the mesh."""
    from nanigpt.distributed.mesh import get_process_group
    from nanigpt.distributed.process_group import NaniProcessGroup

    for dim_name in mesh._mesh_dim_names:
        pg = get_process_group(mesh, dim_name)
        if isinstance(pg, NaniProcessGroup):
            pg.enable_comm_timing()
            log.info(f"CommTiming: enabled for {dim_name}")
        else:
            log.warning(f"CommTiming: {dim_name} PG is {type(pg).__name__}, cannot enable timing")
