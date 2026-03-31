# Tensor Parallelism

## Overview

Tensor parallelism shards individual weight matrices across GPUs within a layer, so each GPU computes a slice of each matmul. For a transformer, this means splitting attention heads and FFN hidden dimensions across the TP group. The result is lower per-GPU memory (each rank holds 1/tp of each weight) at the cost of two all-reduces per layer per forward pass.

The core pattern is a column-parallel + row-parallel pair. Column-parallel shards a weight's output dimension — each rank computes a different slice of the output. Row-parallel shards a weight's input dimension — each rank computes a partial sum that must be combined. These always appear in sequence: column-parallel (no communication in forward) followed by row-parallel (all-reduce in forward to recombine partials). The backward communication is the mirror image.

naniGPT takes Megatron's communication primitives (the conjugate autograd.Function pairs) but applies them torchtitan-style: the model stays a pure `nn.Module` with standard `nn.Linear` layers, and TP is applied externally by sharding weights and replacing forward methods. The model code has zero awareness of parallelism.

## Key Files

- **`nanigpt/distributed/comm.py`** — The four autograd.Function conjugate pairs that handle communication. Adapted from Megatron's `mappings.py`, renamed for readability. This is where the actual `dist.all_reduce` / `dist.all_gather` calls live.
- **`nanigpt/distributed/tensor_parallel.py`** — TP execution layer. Defines `column_parallel_linear` / `row_parallel_linear`, the weight sharding functions (`_shard_linear_colwise/rowwise`), and `apply_sharding` which applies a resolved `ShardingPlan` to the model. `apply_tensor_parallelism` is the top-level entry point that calls resolution then execution.
- **`nanigpt/distributed/sharding.py`** — `ModuleSharding`, `Shard`/`Replicate` placement types, `colwise()`/`rowwise()` factories. The description layer — see [distributed overview](overview.md).
- **`nanigpt/distributed/registry.py`** — Type-based dispatch registry. `resolve_sharding()` walks the model and returns a `ShardingPlan` by dispatching on module type (`MultiHeadAttention` → q/k/v colwise + out rowwise, `FeedForward` → up colwise + down rowwise).
- **`nanigpt/distributed/spmd.py`** — The SPMD type system (I/R/V/P) for verifying correctness. Used as inline type comments throughout tensor_parallel.py and as optional runtime assertions.
- **`nanigpt/distributed/__init__.py`** — Orchestration. TP is applied first (before FSDP/DDP) because FSDP operates on the already-sharded weights.
- **`nanigpt/distributed/mesh.py`** — DeviceMesh creation. TP is the innermost mesh dimension so adjacent ranks share NVLink.

## Invariants

1. **`enter` and `exit` must be paired around a column+row block.** `enter_parallel_region` before the column-parallel matmul, `exit_parallel_region` after the row-parallel matmul. Missing either produces wrong gradients silently.

2. **`nonlinear(P)` is forbidden.** Applying a nonlinearity (GELU, softmax) to a partial-sum tensor gives wrong gradients with no error. GELU goes between column-parallel output (V@tp, safe) and row-parallel input — never after row-parallel and before the reduce. The SPMD type system exists primarily to catch this.

3. **Bias is added after reduce, not before.** In row-parallel linear, bias must be added *after* `exit_parallel_region`. Adding it before means every rank adds the full bias, and the all-reduce sums them — giving `tp_size × bias` instead of `bias`.

4. **TP is applied before FSDP.** `apply_tensor_parallelism` shards weights first, then `fully_shard` operates on the already-sharded parameters. Reversing this order means FSDP would shard the full weight, then TP would try to slice an already-distributed tensor.

5. **`n_heads` must be divisible by `tp_size`.** Each rank gets `n_heads // tp_size` attention heads. Non-divisible configurations are caught at apply time, not silently wrong.

6. **TP ranks must be on the same node (NVLink).** TP does 2 all-reduces per layer — the most communication-intensive parallelism. Mesh dimension ordering puts TP innermost (consecutive rank IDs) which maps to same-node GPUs by launcher convention. Running TP across nodes works but is bandwidth-bottlenecked.

7. **The model stays a pure `nn.Module`.** TP replaces `forward` methods on individual `nn.Linear` modules via closures. It does not subclass, wrap, or modify the model class. This means `model.state_dict()` still works, `torch.save`/`load` still works, and the model can be inspected with standard PyTorch tools.

## Design Decisions

**Functions, not layers.** Megatron implements `ColumnParallelLinear` as a ~300-line `nn.Module` subclass that fuses weight sharding, communication, async overlap, and gradient accumulation into one class. We use two ~10-line functions (`column_parallel_linear`, `row_parallel_linear`) that wrap standard `F.linear` calls with comm primitives. This trades Megatron's optimization surface for clarity and composability. See [architecture design entry](../../../journal/parallelisms/2026-03-26-parallelism-architecture-design.md) for the full tradeoff analysis.

**Explicit comm over DTensor.** torchtitan uses PyTorch's DTensor to handle communication implicitly — you declare placements and DTensor inserts the right collectives. We use Megatron's explicit `autograd.Function` pairs instead. Every collective is visible in the code and profileable. The tradeoff: more code, but no magic. See [porting Megatron optimizations entry](../../../journal/parallelisms/2026-03-18-porting-megatron-optimizations-to-dtensor.md) for why this doesn't conflict with `torch.compile`.

**Renamed Megatron primitives.** Megatron's `_CopyToModelParallelRegion` / `_ReduceFromModelParallelRegion` are renamed to `enter_parallel_region` / `exit_parallel_region`. The original names describe the mechanism; ours describe the position in the dataflow. The mapping table lives in `comm.py`'s module docstring.

**Type-based dispatch.** `resolve_sharding()` dispatches on `type(module)` via a registry, not by name suffix. The original name-suffix approach (`"attn.q_proj"` → colwise) was ambiguous for heterogeneous models (MoE blocks at the same structural position as dense blocks, vision vs language attention). Type dispatch is unambiguous. See [extending the architecture entry](../../../journal/parallelisms/2026-03-30-extending-the-parallelism-architecture.md) for the motivation.
