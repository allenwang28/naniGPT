"""Tensor parallelism: TP as functions applied to a pure nn.Module.

The model stays a standard nn.Module with nn.Linear layers. TP is applied
externally by sharding weights and replacing forward methods with closures
that call the comm primitives from comm.py.

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
from typing import Literal

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
from nanigpt.distributed.spmd import SPMDType, assert_type, spmd_checks_enabled

log = logging.getLogger(__name__)

# Default TP plan for DenseTransformer. Keys are module name suffixes
# matched against named_modules(). Values are "colwise" or "rowwise".
DEFAULT_TP_STYLE: dict[str, Literal["colwise", "rowwise"]] = {
    "attn.q_proj": "colwise",
    "attn.k_proj": "colwise",
    "attn.v_proj": "colwise",
    "attn.out_proj": "rowwise",
    "ffn.up": "colwise",
    "ffn.down": "rowwise",
}


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


def apply_tensor_parallelism(
    model: nn.Module,
    mesh: DeviceMesh,
) -> nn.Module:
    """Apply tensor parallelism to a model using the default TP plan.

    Walks the model's named modules, matches against DEFAULT_TP_STYLE,
    shards weights, and replaces forward methods. Also adjusts attention
    head counts for the local shard.

    Args:
        model: Pure nn.Module (e.g. DenseTransformer). Must already be on device.
        mesh: DeviceMesh with a "tp" dimension.

    Returns:
        The same model, modified in-place with sharded weights and TP forwards.
    """
    tp_mesh = mesh["tp"]
    tp_rank = tp_mesh.get_local_rank()
    tp_size = tp_mesh.size()
    group = tp_mesh.get_group()

    if tp_size == 1:
        log.info("TP size is 1, skipping tensor parallelism")
        return model

    # Validate model dimensions are divisible by tp_size
    from nanigpt.models.dense_transformer import MultiHeadAttention

    # (name, style, shape_before, shape_after) for logging
    sharded: list[tuple[str, str, str, str]] = []

    for name, module in model.named_modules():
        # Check if this module matches any TP style
        for suffix, style in DEFAULT_TP_STYLE.items():
            if name.endswith(suffix) and isinstance(module, nn.Linear):
                shape_before = f"[{module.out_features}, {module.in_features}]"
                if style == "colwise":
                    if module.out_features % tp_size != 0:
                        raise ValueError(
                            f"{name}: out_features ({module.out_features}) "
                            f"not divisible by tp_size ({tp_size})"
                        )
                    _shard_linear_colwise(module, tp_rank, tp_size, group)
                elif style == "rowwise":
                    if module.in_features % tp_size != 0:
                        raise ValueError(
                            f"{name}: in_features ({module.in_features}) "
                            f"not divisible by tp_size ({tp_size})"
                        )
                    _shard_linear_rowwise(module, tp_rank, tp_size, group)
                shape_after = f"[{module.out_features}, {module.in_features}]"
                sharded.append((name, style, shape_before, shape_after))
                break

        # Adjust attention head count for TP
        if isinstance(module, MultiHeadAttention):
            if module.n_heads % tp_size != 0:
                raise ValueError(
                    f"{name}: n_heads ({module.n_heads}) not divisible by tp_size ({tp_size})"
                )
            module.n_heads = module.n_heads // tp_size

    # Pretty-print the sharding plan (rank 0 only to avoid duplicate spam)
    if tp_rank == 0:
        name_w = max(len(n) for n, _, _, _ in sharded)
        style_w = max(len(s) for _, s, _, _ in sharded)
        before_w = max(len(b) for _, _, b, _ in sharded)
        header = f"  {'module':<{name_w}}  {'style':<{style_w}}  {'before':<{before_w}}  after"
        sep = f"  {'-' * name_w}  {'-' * style_w}  {'-' * before_w}  -----"
        rows = "\n".join(
            f"  {name:<{name_w}}  {style:<{style_w}}  {before:<{before_w}}  {after}"
            for name, style, before, after in sharded
        )
        log.info(
            f"Applied tensor parallelism (tp_size={tp_size}, {len(sharded)} modules):\n"
            f"{header}\n{sep}\n{rows}"
        )
    return model

