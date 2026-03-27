"""Data parallelism: DDP and FSDP model wrapping.

Applies data parallelism to a model using PyTorch's composable APIs.
The model stays a pure nn.Module — no custom wrapper classes.
"""

import logging

import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh

log = logging.getLogger("distributed.data_parallel")


def apply_data_parallelism(
    model: nn.Module,
    mesh: DeviceMesh,
    strategy: str,
) -> nn.Module:
    """Wrap a model for data-parallel training.

    Args:
        model: The model to wrap (should already be on the correct device).
        mesh: DeviceMesh with a "dp" dimension.
        strategy: "ddp" or "fsdp".

    Returns:
        The wrapped model.
    """
    if strategy == "ddp":
        return _apply_ddp(model, mesh)
    elif strategy == "fsdp":
        return _apply_fsdp(model, mesh)
    else:
        raise ValueError(f"Unknown data parallelism strategy: {strategy}")


def _apply_ddp(model: nn.Module, mesh: DeviceMesh) -> nn.Module:
    """Apply composable DDP via DTensor-based replicate."""
    from torch.distributed._composable.replicate import replicate

    replicate(model, device_mesh=mesh)
    log.info("Applied composable DDP (replicate)")
    return model


def _apply_fsdp(model: nn.Module, mesh: DeviceMesh) -> nn.Module:
    """Apply FSDP2 via fully_shard.

    Shards each TransformerBlock individually, then the root model.
    token_emb and lm_head stay in the root FSDP unit to preserve weight tying.
    """
    from torch.distributed._composable.fsdp import fully_shard

    from nanigpt.models.dense_transformer import TransformerBlock

    for module in model.modules():
        if isinstance(module, TransformerBlock):
            fully_shard(module, mesh=mesh)

    fully_shard(model, mesh=mesh)
    log.info("Applied FSDP2 (fully_shard)")
    return model
