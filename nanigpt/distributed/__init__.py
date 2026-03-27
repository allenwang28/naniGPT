"""Distributed runtime: process groups, mesh, and model wrapping.

This package owns the entire distributed concern — from process group
lifecycle to parallelism application. No globals; the DeviceMesh is
created once and passed explicitly.

Directory structure (current and planned):
    distributed/
    ├── __init__.py           # Process group lifecycle + re-exports
    ├── mesh.py               # DeviceMesh creation
    ├── data_parallel.py      # DDP / FSDP wrapping
    ├── plan.py               # ParallelPlan dataclass
    ├── comm.py               # Autograd.Function conjugate pairs (stub)
    ├── tensor_parallel.py    # (future) TP as functions, not layers
    ├── expert_parallel.py    # (future) EP dispatch/combine strategies
    └── pipeline_parallel.py  # (future) PP schedule IR + execution
"""

import logging
import os

import torch
import torch.distributed as dist

from nanigpt.config import ParallelConfig
from nanigpt.distributed.data_parallel import apply_data_parallelism
from nanigpt.distributed.mesh import create_device_mesh
from nanigpt.distributed.plan import ParallelPlan

__all__ = [
    "apply_parallelism",
    "cleanup_distributed",
    "create_device_mesh",
    "get_rank",
    "get_world_size",
    "init_distributed",
    "is_rank_zero",
    "ParallelPlan",
]

log = logging.getLogger("distributed")


# ---- Process group lifecycle ----


def init_distributed(rank: int, world_size: int, backend: str = "nccl") -> None:
    """Initialize the default process group and set CUDA device.

    Expects MASTER_ADDR and MASTER_PORT to already be set in the environment.
    Each rank should see a single GPU via CUDA_VISIBLE_DEVICES (Ray handles this).
    """
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29500")

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

    Currently supports DDP and FSDP. When TP/PP are added, this function
    will read additional fields from ParallelConfig to build a multi-dimensional
    mesh and apply strategies in the right order (TP first, then DP).
    """
    mesh = create_device_mesh(world_size)
    return apply_data_parallelism(model, mesh, config.plan)
