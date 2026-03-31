"""Tensor parallelism: TP as functions applied to a pure nn.Module.

The model stays a standard nn.Module with nn.Linear layers. TP is applied
externally by sharding weights and replacing forward methods with closures
that call the comm primitives from comm.py.

This module owns the TP application layer: slicing weights and installing
comm-aware forwards. Resolution (deciding *what* to shard) lives in
registry.py and is TP-agnostic.

Data flow through a TP transformer layer (no sequence parallelism):

    Input x: [B, S, H]                           SPMD: I@tp
    │
    │  enter_parallel_region (identity fwd, all-reduce bwd)   I@tp → R@tp
    │  q = F.linear(x, W_q[:out//tp, :])          R@tp × V@tp → V@tp
    │  k = F.linear(x, W_k[:out//tp, :])          R@tp × V@tp → V@tp
    │  v = F.linear(x, W_v[:out//tp, :])          R@tp × V@tp → V@tp
    │
    │  attn_out = attention(q, k, v)  — local, each rank has its heads   V@tp
    │
    │  proj = F.linear(attn_out, W_out[:, :in//tp])   V@tp × V@tp → P@tp
    │  exit_parallel_region (all-reduce fwd, identity bwd)    P@tp → I@tp
    │
    │  x = x + proj                                   I@tp
    │
    │  enter_parallel_region                           I@tp → R@tp
    │  ffn_up = F.linear(x, W_up[:out//tp, :])        R@tp × V@tp → V@tp
    │  GELU(ffn_up)                                   V@tp (safe: nonlinear on V, not P)
    │
    │  ffn_down = F.linear(act, W_down[:, :in//tp])   V@tp × V@tp → P@tp
    │  exit_parallel_region                            P@tp → I@tp
    │
    │  x = x + ffn_down                               I@tp
    │
    Output x: [B, S, H]                          SPMD: I@tp ✓

    Total communication: 2 all-reduces per layer (in backward: 2 more)
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributed import ProcessGroup
from torch.distributed.device_mesh import DeviceMesh

from nanigpt.distributed.comm import (
    enter_parallel_region,
    exit_parallel_region,
)
from nanigpt.distributed.plan import ParallelPlan
from nanigpt.distributed.registry import ShardingPlan, resolve_sharding
from nanigpt.distributed.sharding import Shard
from nanigpt.distributed.spmd import SPMDType, assert_type, spmd_checks_enabled

log = logging.getLogger(__name__)


def column_parallel_linear(
    input_: Tensor,
    weight: Tensor,
    bias: Tensor | None,
    group: ProcessGroup,
) -> Tensor:
    """Column-parallel linear: shard output dim across TP group.

    SPMD flow: input I@tp → enter → R@tp, matmul R×V → V@tp.
    Forward: identity on input (enter), local matmul with sharded weight.
    Backward: all-reduce on input grad (enter backward), local wgrad.
    """
    input_ = enter_parallel_region(input_, group)  # I@tp → R@tp
    if spmd_checks_enabled():
        assert_type(input_, SPMDType.REPLICATE, group)

    output = F.linear(input_, weight, bias)  # R@tp × V@tp → V@tp

    if spmd_checks_enabled():
        assert_type(output, SPMDType.VARYING, group)
    return output


def row_parallel_linear(
    input_: Tensor,
    weight: Tensor,
    bias: Tensor | None,
    group: ProcessGroup,
) -> Tensor:
    """Row-parallel linear: shard input dim across TP group.

    SPMD flow: input V@tp, matmul V×V → P@tp, exit → I@tp.
    Forward: local matmul with sharded weight, all-reduce output.
    Backward: identity on output grad (exit backward), local wgrad.
    """
    # V@tp × V@tp → P@tp
    output = F.linear(input_, weight)  # no bias yet — add after reduce

    # P@tp → I@tp
    output = exit_parallel_region(output, group)

    if spmd_checks_enabled():
        assert_type(output, SPMDType.INVARIANT, group)

    # Bias is only on rank 0 to avoid double-counting after all-reduce.
    # Alternatively, bias is replicated and added after reduce (same result).
    if bias is not None:
        output = output + bias

    return output


def _shard_linear_colwise(
    linear: nn.Linear,
    tp_rank: int,
    tp_size: int,
    group: ProcessGroup,
) -> None:
    """Shard an nn.Linear for column-parallel: split output dim.

    Weight shape: [out_features, in_features] → [out_features // tp, in_features]
    Bias shape: [out_features] → [out_features // tp]
    """
    out_features = linear.out_features
    shard_size = out_features // tp_size
    start = tp_rank * shard_size
    end = start + shard_size

    # Shard weight in-place
    with torch.no_grad():
        linear.weight = nn.Parameter(linear.weight[start:end, :].contiguous())
        if linear.bias is not None:
            linear.bias = nn.Parameter(linear.bias[start:end].contiguous())

    linear.out_features = shard_size

    # Replace forward with column-parallel version
    def forward(input_: Tensor) -> Tensor:
        return column_parallel_linear(input_, linear.weight, linear.bias, group)

    linear.forward = forward


def _shard_linear_rowwise(
    linear: nn.Linear,
    tp_rank: int,
    tp_size: int,
    group: ProcessGroup,
) -> None:
    """Shard an nn.Linear for row-parallel: split input dim.

    Weight shape: [out_features, in_features] → [out_features, in_features // tp]
    Bias: kept on all ranks (added after all-reduce).
    """
    in_features = linear.in_features
    shard_size = in_features // tp_size
    start = tp_rank * shard_size
    end = start + shard_size

    with torch.no_grad():
        linear.weight = nn.Parameter(linear.weight[:, start:end].contiguous())
        # Bias stays full — it's added after the all-reduce in row_parallel_linear

    linear.in_features = shard_size

    # Replace forward with row-parallel version
    def forward(input_: Tensor) -> Tensor:
        return row_parallel_linear(input_, linear.weight, linear.bias, group)

    linear.forward = forward


def apply_sharding(
    model: nn.Module,
    sharding_plan: ShardingPlan,
    mesh: DeviceMesh,
) -> None:
    """Execute a resolved sharding plan: slice weights and replace forwards.

    This is the execution step — it mutates the model in-place. The resolution
    step (resolve_sharding) has already decided what to shard and how.

    Args:
        model: Pure nn.Module, must already be on device.
        sharding_plan: Output of resolve_sharding().
        mesh: DeviceMesh with a "tp" dimension.
    """
    tp_mesh = mesh["tp"]
    tp_rank = tp_mesh.get_local_rank()
    tp_size = tp_mesh.size()
    group = tp_mesh.get_group()

    module_dict = dict(model.named_modules())

    for entry in sharding_plan.entries:
        fqn = f"{entry.parent_fqn}.{entry.child_name}" if entry.parent_fqn else entry.child_name
        child = module_dict.get(fqn)
        if child is None or not isinstance(child, nn.Linear):
            log.warning(f"Skipping {fqn}: not found or not nn.Linear")
            continue

        weight_placements = entry.sharding.params.get("weight", {})
        tp_placement = weight_placements.get("tp")

        if isinstance(tp_placement, Shard) and tp_placement.dim == 0:
            if child.out_features % tp_size != 0:
                raise ValueError(
                    f"{fqn}: out_features ({child.out_features}) "
                    f"not divisible by tp_size ({tp_size})"
                )
            _shard_linear_colwise(child, tp_rank, tp_size, group)
        elif isinstance(tp_placement, Shard) and tp_placement.dim == 1:
            if child.in_features % tp_size != 0:
                raise ValueError(
                    f"{fqn}: in_features ({child.in_features}) not divisible by tp_size ({tp_size})"
                )
            _shard_linear_rowwise(child, tp_rank, tp_size, group)

    # Apply attribute adjustments (e.g. n_heads for attention)
    for parent_fqn, adjustments in sharding_plan.adjustments.items():
        parent = module_dict.get(parent_fqn)
        if parent is None:
            log.warning(f"Skipping adjustment for {parent_fqn}: not found")
            continue
        for attr, value in adjustments.items():
            setattr(parent, attr, value)


def apply_tensor_parallelism(
    model: nn.Module,
    plan: ParallelPlan,
    mesh: DeviceMesh,
) -> nn.Module:
    """Apply tensor parallelism to a model using the type-based registry.

    Resolution is separated from execution:
    1. resolve_sharding() walks the model and dispatches on module type
       to decide what gets sharded and how (pure, no side effects).
    2. apply_sharding() slices weights and replaces forward methods.

    Args:
        model: Pure nn.Module (e.g. DenseTransformer). Must already be on device.
        plan: ParallelPlan with all parallelism degrees.
        mesh: DeviceMesh with a "tp" dimension.

    Returns:
        The same model, modified in-place with sharded weights and TP forwards.
    """
    tp_mesh = mesh["tp"]
    tp_rank = tp_mesh.get_local_rank()
    tp_size = tp_mesh.size()

    if tp_size == 1:
        log.info("TP size is 1, skipping tensor parallelism")
        return model

    # Resolve: decide what to shard (pure, no side effects)
    sharding_plan = resolve_sharding(model, plan)

    # Log the plan before execution (rank 0 only)
    if tp_rank == 0 and sharding_plan.entries:
        table = sharding_plan.format_table(model, plan)
        n = len(sharding_plan.entries)
        log.info(f"Sharding plan (tp_size={tp_size}, {n} modules):\n{table}")

    # Execute: slice weights, replace forwards, apply adjustments
    apply_sharding(model, sharding_plan, mesh)

    return model
