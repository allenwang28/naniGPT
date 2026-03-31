"""Data parallelism: DDP and FSDP model wrapping.

Applies data parallelism to a model using PyTorch's composable APIs.
The model stays a pure nn.Module — no custom wrapper classes.

FSDP and DDP are applied separately (not dispatched from a single function)
so the caller controls application order explicitly.
"""

import logging

import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh

from nanigpt.distributed.plan import ParallelPlan

log = logging.getLogger("distributed.data_parallel")


def apply_fsdp(model: nn.Module, plan: ParallelPlan, mesh: DeviceMesh) -> nn.Module:
    """Apply FSDP2 via fully_shard.

    Shards each TransformerBlock individually, then the root model.
    token_emb and lm_head stay in the root FSDP unit to preserve weight tying.

    For HSDP: pass a mesh containing both "dp_replicate" and "dp_shard" dims.
    fully_shard on a 2D mesh gives HSDP automatically.
    """
    from torch.distributed._composable.fsdp import fully_shard

    from nanigpt.models.dense_transformer import TransformerBlock

    # Extract the FSDP sub-mesh: [dp_replicate, dp_shard] for HSDP, or just [dp_shard]
    dim_names = mesh.mesh_dim_names
    has_replicate = "dp_replicate" in dim_names
    has_shard = "dp_shard" in dim_names

    if has_replicate and has_shard:
        fsdp_mesh = mesh["dp_replicate", "dp_shard"]
        label = "HSDP"
    elif has_shard:
        fsdp_mesh = mesh["dp_shard"] if mesh.ndim > 1 else mesh
        label = "FSDP2"
    else:
        raise ValueError(f"No dp_shard dimension in mesh {dim_names}")

    for module in model.modules():
        if isinstance(module, TransformerBlock):
            fully_shard(module, mesh=fsdp_mesh)

    fully_shard(model, mesh=fsdp_mesh)
    log.info(f"Applied {label} (fully_shard)")
    return model


def apply_ddp(model: nn.Module, plan: ParallelPlan, mesh: DeviceMesh) -> nn.Module:
    """Apply composable DDP via DTensor-based replicate."""
    from torch.distributed._composable.replicate import replicate

    dp_mesh = mesh["dp_replicate"] if mesh.ndim > 1 else mesh
    replicate(model, device_mesh=dp_mesh)
    log.info("Applied composable DDP (replicate)")
    return model
