# Distributed Parallelism

## Overview

naniGPT's parallelism system is organized into three layers: description, resolution, and execution. Each layer has a clear responsibility, and the boundaries between them are the key architectural property — the first two layers work without GPUs, process groups, or real tensors.

**Description** (`sharding.py`) — Pure data types that express how tensors are distributed across a device mesh. `Shard(dim=0)` means a tensor is split along dimension 0; `Replicate()` means it's identical on all ranks. `ModuleSharding` ties these placements to a module's parameters, inputs, and outputs on named mesh dimensions. No dependencies on torch.distributed.

**Resolution** (`registry.py`) — Walks a model and decides which modules get which sharding, dispatching on `type(module)`. Returns a `ShardingPlan` — a flat list of entries saying "this child of this parent gets this `ModuleSharding`" plus attribute adjustments (e.g. halving `n_heads` for TP). Pure function, no side effects, works on meta-device models.

**Execution** (`tensor_parallel.py`, `data_parallel.py`) — Takes a resolved plan and mutates the model: slices weights, replaces forward methods with comm-aware closures, installs FSDP/DDP wrappers. Requires real devices and process groups. The entry point is `apply_parallelism()` in `__init__.py`, which orchestrates TP → FSDP → DDP in order.

This separation exists so the description and resolution layers can be reused for meta-device dry runs (exact memory estimation without GPUs), cost model projections, and checkpoint resharding plans.

## Key Files

- **`nanigpt/distributed/sharding.py`** — `Shard`, `Replicate`, `ModuleSharding`, `colwise()`/`rowwise()` factories, `compute_local_shape()`.
- **`nanigpt/distributed/registry.py`** — Type-based dispatch registry, `ResolvedSharding`, `ShardingPlan`, `resolve_sharding()`. Built-in registrations for `MultiHeadAttention` and `FeedForward`.
- **`nanigpt/distributed/plan.py`** — `ParallelPlan` dataclass declaring degrees for each mesh dimension (dp_replicate, dp_shard, tp_size).
- **`nanigpt/distributed/mesh.py`** — `create_device_mesh()` builds a multi-dimensional `DeviceMesh` from the plan. TP innermost (NVLink), DP outermost.
- **`nanigpt/distributed/__init__.py`** — `apply_parallelism()` orchestrates TP → FSDP → DDP. Process group lifecycle (init/cleanup).
- **`nanigpt/distributed/comm.py`** — Autograd.Function conjugate pairs for TP communication. See [tensor-parallelism spec](tensor-parallelism.md).
- **`nanigpt/distributed/tensor_parallel.py`** — TP execution: `apply_sharding()` maps resolved `ModuleSharding` entries to weight slicing + forward replacement.
- **`nanigpt/distributed/data_parallel.py`** — FSDP and DDP application using PyTorch's composable APIs.
- **`nanigpt/distributed/spmd.py`** — SPMD type system (I/R/V/P) for verifying TP correctness.

## Invariants

1. **Description and resolution are pure.** They must never import from torch.distributed, create process groups, or mutate model parameters. This is what makes dry runs and cost model sweeps possible. If you find yourself needing a `ProcessGroup` in `sharding.py` or `registry.py`, the abstraction is leaking.

2. **Resolution dispatches on module type, not name.** The registry maps `type(module) → ShardingFn`. Name-suffix matching was the original approach and is gone. Type dispatch is unambiguous for heterogeneous models (dense + MoE, vision + language).

3. **Application order is TP → FSDP → DDP.** TP shards weights first. FSDP then operates on the already-sharded parameters. DDP wraps the result. Reversing any pair produces incorrect behavior (e.g. FSDP would shard full weights, then TP would try to slice a distributed tensor).

4. **Mesh dimension ordering is dp_replicate (outermost) → dp_shard → tp (innermost).** TP innermost means consecutive rank IDs share NVLink. This is a convention enforced by `create_device_mesh()` — launcher code assumes it.

5. **The model stays a pure `nn.Module`.** No parallelism-aware base classes, no custom `state_dict` methods, no `ColumnParallelLinear` replacement modules. Parallelism is applied externally. This means standard PyTorch serialization, inspection, and debugging tools all work.

6. **Placement types are our own, not DTensor's.** `Shard` and `Replicate` in `sharding.py` are frozen dataclasses with no behavior. DTensor's placement types carry runtime machinery for redistribution and dispatch. Keeping our own ensures the description layer stays dependency-free. The API (`Shard(dim=N)`, `Replicate()`) matches DTensor's, so migration is a swap of imports if DTensor's types stabilize.

## Design Decisions

**Three layers, not two.** Many frameworks fuse resolution and execution — you call `parallelize(model, mesh)` and it decides what to shard and does it in one pass. Splitting them lets us reuse resolution for dry runs and cost models. The cost is an extra data structure (`ShardingPlan`) flowing between the two steps.

**Explicit comm over DTensor.** We use Megatron's autograd.Function conjugate pairs for communication rather than DTensor's implicit redistribution. Every collective is visible in the source and profileable. The tradeoff: more code to write, but no hidden communication. See [tensor-parallelism spec](tensor-parallelism.md) for details.

**Composable APIs for data parallelism.** FSDP and DDP are applied via PyTorch's composable APIs (`fully_shard()`, `replicate()`) rather than wrapper classes (`DistributedDataParallel`). The model stays unwrapped, and FSDP/DDP are just hooks installed on it. HSDP falls out naturally from a 2D mesh passed to `fully_shard()`.

**ParallelPlan is a data description.** The plan declares intent (degrees for each dimension) without knowing how to execute it. This lets the same plan drive both real execution and dry-run analysis. `dp_shard=-1` auto-fills with remaining ranks, making pure FSDP the zero-config default.
