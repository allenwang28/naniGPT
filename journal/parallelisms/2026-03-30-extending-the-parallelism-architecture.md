# Extending the Parallelism Architecture

*2026-03-30*

The [architecture design entry](2026-03-26-parallelism-architecture-design.md) laid out the decoupling strategy: Megatron's comm primitives, torchtitan's model purity, SPMD types for correctness. With DDP, FSDP, and TP now working, several design choices that were "good enough for TP-only" are starting to show cracks as we think about adding CP, EP, PP, and reshardable checkpoints. This entry identifies five concrete problems and proposes solutions.

---

## Problem 1: The Sharding Plan Can't Express Multi-Dimensional Sharding

The current `TensorParallelPlan` is a flat dict:

```python
style: dict[str, Literal["colwise", "rowwise"]] = {
    "attn.q_proj": "colwise",
    "attn.out_proj": "rowwise",
    "ffn.up": "colwise",
    "ffn.down": "rowwise",
}
```

This expresses exactly one thing: how each module is sharded on the TP dimension. But a module in a real training setup participates in *multiple* parallelism dimensions simultaneously:

- A column-parallel expert FFN weight is sharded on TP (output dim) *and* partitioned across EP (expert dim) *and* FSDP-sharded (for memory)
- Activations between TP-sharded layers are sequence-sharded on CP
- The same `nn.Linear` needs different treatment depending on whether it's inside a dense block or an MoE block

The flat dict can't express any of this. We'd end up with parallel dicts (`tp_style`, `ep_style`, `cp_style`) that need to be kept in sync — the same kind of coupling the architecture was designed to avoid.

### Solution: `ModuleSharding` dataclass

Replace the string-valued dict with a structured spec per module that covers all mesh dimensions:

```python
@dataclass(frozen=True)
class ModuleSharding:
    """How a single module's params, inputs, and outputs are sharded across the mesh."""
    params: dict[str, dict[str, Placement]]   # param_name -> {mesh_dim: Shard(dim) | Replicate()}
    input_placements: dict[str, Placement]     # mesh_dim -> expected input placement
    output_placements: dict[str, Placement]    # mesh_dim -> produced output placement
```

`"colwise"` and `"rowwise"` become factory functions that produce a `ModuleSharding`:

```python
def colwise(tp_dim: str = "tp") -> ModuleSharding:
    return ModuleSharding(
        params={"weight": {tp_dim: Shard(0)}, "bias": {tp_dim: Shard(0)}},
        input_placements={tp_dim: Replicate()},
        output_placements={tp_dim: Shard(-1)},
    )

def rowwise(tp_dim: str = "tp") -> ModuleSharding:
    return ModuleSharding(
        params={"weight": {tp_dim: Shard(1)}},
        input_placements={tp_dim: Shard(-1)},
        output_placements={tp_dim: Replicate()},   # after all-reduce/reduce-scatter
    )
```

This doesn't change the underlying comm — we still use Megatron's explicit `autograd.Function` pairs. The `ModuleSharding` is the *spec*, the comm functions are the *execution*. The spec tells `apply_tensor_parallelism()` what to do; the comm primitives tell it how.

For MoE, the spec extends naturally:

```python
def expert_colwise(tp_dim: str = "tp", ep_dim: str = "ep") -> ModuleSharding:
    return ModuleSharding(
        params={"weight": {tp_dim: Shard(0), ep_dim: Shard(0)}},  # expert dim + output dim
        input_placements={tp_dim: Replicate(), ep_dim: Shard(0)},
        output_placements={tp_dim: Shard(-1), ep_dim: Shard(0)},
    )
```

Same structure, more dimensions. No new abstractions needed.

---

## Problem 2: Name-Suffix Matching Breaks With Heterogeneous Models

The current dispatch in `apply_tensor_parallelism()` walks `named_modules()` and matches name suffixes:

```python
for name, module in model.named_modules():
    for suffix, style in plan.style.items():
        if name.endswith(suffix):
            ...
```

This has three failure modes we'll hit as the model zoo grows:

**Ambiguity.** A vision encoder attached to a language model may have its own `attn.q_proj`. The suffix `"attn.q_proj"` matches both, but they may need different sharding (the vision encoder's attention might not be TP'd at all, or might use a different TP group).

**Heterogeneous blocks.** A model with alternating dense and MoE layers has both `layers.3.ffn.up` and `layers.4.moe.gate`. The suffix trick works here by accident (different names), but a model where dense and MoE blocks share the same FFN structure (e.g., shared expert + routed experts) would break.

**Injected modules.** LoRA adapters, quantization wrappers, or activation checkpointing wrappers change the module tree structure. A suffix that matched `layers.3.attn.q_proj` may not match `layers.3.attn.q_proj.lora_A`.

### Solution: Type-based dispatch with suffix fallback

Dispatch on `type(module)` instead of name suffix:

```python
ShardingFn = Callable[[nn.Module, DeviceMesh, ParallelPlan], ModuleSharding | None]

SHARDING_REGISTRY: dict[type[nn.Module], ShardingFn] = {
    MultiHeadAttention: shard_attention,
    FeedForward: shard_ffn,
    # MoELayer: shard_moe,  # future
}

def apply_parallelism(model, plan, mesh):
    for name, module in model.named_modules():
        shard_fn = SHARDING_REGISTRY.get(type(module))
        if shard_fn is not None:
            sharding = shard_fn(module, mesh, plan)
            if sharding is not None:
                _apply_sharding(module, sharding, mesh)
```

Each `shard_fn` receives the module, mesh, and plan, and returns a `ModuleSharding` (or `None` to skip). This handles all three failure modes:
- `shard_attention` can check whether the module is inside a vision encoder or language model
- Different module types get different sharding functions
- Adapter layers don't match any registry entry and are skipped by default

Keep the current suffix-matching as a fallback for quick prototyping — if a module type isn't in the registry, fall through to suffix matching. This is backward-compatible.

**Revised position on Resolved Question 4 from the architecture entry:** Module name suffixes are sufficient for a single dense transformer. They are insufficient once you have heterogeneous module types at the same structural position (dense FFN vs MoE block, cross-attention vs self-attention, language vs vision attention). Type-based dispatch handles this naturally.

---

## Problem 3: Sharding Annotations Don't Distinguish Storage From Computation

The [architecture entry](2026-03-26-parallelism-architecture-design.md) proposed `ShardingInfo` stamped on parameters for checkpoint resharding. But the current design captures only *one* sharding state per parameter. In practice there are *two*:

- **Storage sharding**: how the parameter lives in memory between forward/backward. E.g., sharded on both TP (column-wise) and FSDP (ZeRO-3). This is what checkpointing sees.
- **Computation sharding**: how the parameter appears during the actual matmul. E.g., sharded on TP only — the FSDP dimension has been all-gathered away.

The delta between these two states *is* what FSDP does: all-gather the FSDP dimension in forward, reduce-scatter in backward. Currently FSDP2 handles this internally, and we don't track it. This creates two problems:

**1. Checkpoint resharding is blind to FSDP.** If a checkpoint was saved with `tp=8, fsdp=4` and we want to resume with `tp=4, fsdp=8`, the resharder needs to know which dimensions are TP-sharded (slice differently) vs FSDP-sharded (redistribute differently). Without the storage/computation distinction, it can't tell.

**2. We can't verify sharding specs.** If `ModuleSharding` says a weight is `{tp: Shard(0), fsdp: Shard(1)}` but FSDP is handling the second dimension, we have no way to check that the annotation matches what FSDP is actually doing.

### The three-phase lifecycle of a parameter

A parameter moves through three states during a training step. Each state has a different sharding, and the transitions between them are the actual distributed communication:

```
Phase 1: STORAGE (at rest)
    Weight lives in memory sharded on all relevant mesh dims.
    This is the cheapest memory footprint — the whole point of FSDP.

    Example: attention Q weight [4096, 4096]
    Storage: {tp: Shard(0), fsdp: Shard(0)}
    Local shape: [4096 / tp_size / fsdp_size, 4096]
    Each rank holds a tiny slice.

                    ┌─── all-gather on fsdp dim ───┐
                    │   (FSDP forward prefetch)     │
                    ▼                               │
Phase 2: COMPUTATION (during forward/backward)      │
    FSDP dimension has been all-gathered away.       │
    Weight is sharded only on TP.                    │
    This is what the matmul sees.                    │
                                                     │
    Computation: {tp: Shard(0)}                      │
    Local shape: [4096 / tp_size, 4096]              │
    Each TP rank has its column shard, fully          │
    gathered across the FSDP group.                  │
                                                     │
                    │                               │
                    └─── reduce-scatter on fsdp ────┘
                        (FSDP backward)

Phase 3: GRADIENT (after backward)
    Gradient is reduced across DP ranks and
    resharded back to storage layout for the
    optimizer step.
```

The key insight: **the `storage → computation` transition is not a separate operation we schedule — it's what FSDP already does.** FSDP's all-gather in forward *is* removing the `fsdp` dimension from the sharding. FSDP's reduce-scatter in backward *is* re-adding it. By tracking both states declaratively, we make this implicit behavior explicit and queryable.

### Concrete example: Q weight with tp=2, fsdp=4

```
Global weight shape: [4096, 4096]

Storage sharding:     {tp: Shard(0), fsdp: Shard(0)}
  → local shape:      [4096 / 2 / 4, 4096] = [512, 4096]
  → each rank holds 512 rows of Q

Computation sharding: {tp: Shard(0)}
  → local shape:      [4096 / 2, 4096] = [2048, 4096]
  → each TP rank holds 2048 rows (its full column shard)

Delta (storage → computation):
  → fsdp dim goes from Shard(0) to absent
  → meaning: all-gather 4 chunks of [512, 4096] → [2048, 4096]
  → this is exactly what FSDP2's fully_shard() does in forward
```

### Why this matters for checkpoint resharding

A checkpoint saved with `tp=8, fsdp=4` stores parameters in their storage layout. To resume with `tp=4, fsdp=8`, the resharder needs to:

1. **Reconstruct global shape** from storage placements + local shape (unambiguous if both states are recorded).
2. **Identify which dimensions changed**: TP went from 8→4 (merge adjacent shards), FSDP went from 4→8 (split shards further). These are different operations — you can't just "redistribute" uniformly.
3. **Apply TP resharding** by concatenating/slicing along the TP-sharded dimension.
4. **Apply FSDP resharding** by rechunking along the FSDP-sharded dimension.

Without the storage/computation distinction, the resharder sees a single `{Shard(0)}` and doesn't know whether that's TP or FSDP. With both states, it can diff them to identify exactly which mesh dimensions each placement belongs to.

### Solution: Extend `ShardingInfo` with both states

```python
@dataclass(frozen=True)
class ShardingInfo:
    storage: dict[str, Placement]      # how param is stored (sharded on TP + FSDP)
    computation: dict[str, Placement]  # how param appears during compute (sharded on TP only)
    global_shape: tuple[int, ...]      # shape in canonical (unsharded) space
    rank: int
    group_size: int
```

The `storage` state is what checkpointing reads. The `storage → computation` delta is what FSDP must provide. The `computation` state is what the `ModuleSharding` spec describes.

For checkpoint resharding: compare the source checkpoint's `storage` placements against the target's `storage` placements. Dimensions where placements differ need redistribution. Dimensions where they agree can be loaded directly.

### Relationship to `ModuleSharding`

The `ModuleSharding` dataclass (implemented in Problem 1) currently describes the computation sharding — it says "Q weight is `{tp: Shard(0)}`", meaning column-parallel during the matmul. To support Problem 3, we extend it with storage placements:

```python
@dataclass(frozen=True, slots=True)
class ModuleSharding:
    params: dict[str, dict[str, Placement]]           # computation placements
    storage_params: dict[str, dict[str, Placement]]    # storage placements (superset of above)
    input_placements: dict[str, Placement]
    output_placements: dict[str, Placement]
```

The `colwise()` factory would then accept an optional `fsdp_dim` to include FSDP in the storage placements:

```python
def colwise(mesh_dim: str = "tp", fsdp_dim: str | None = None) -> ModuleSharding:
    computation = {"weight": {mesh_dim: Shard(dim=0)}}
    storage = dict(computation["weight"])  # start with computation placements
    if fsdp_dim:
        storage[fsdp_dim] = Shard(dim=0)   # add FSDP sharding for storage
    return ModuleSharding(
        params=computation,
        storage_params={"weight": storage},
        ...
    )
```

This keeps `storage_params` as a strict superset of `params` — every computation placement is also a storage placement, plus FSDP adds more. The delta between them is what FSDP needs to all-gather/reduce-scatter.

We still let upstream FSDP2 handle the actual all-gather/reduce-scatter mechanics — the annotations are metadata for checkpointing and verification, not control signals for FSDP. This keeps us on upstream FSDP2 without needing a custom fork. If MoE with heterogeneous per-parameter meshes forces a fork, the annotations are already in place to drive it.

---

## Problem 4: The Profiler Can't See FSDP or DDP Collectives

The [predict-measure-explain entry](2026-03-27-predict-measure-explain.md) proposes instrumenting `comm.py` for the MEASURE step. Each comm primitive wraps its collective in `measure(EventType.TP_COMM)`:

```python
class _ReduceFromParallelRegion(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, group):
        with measure(EventType.TP_COMM, input_.nbytes):  # instrumented
            dist.all_reduce(input_, group=group)
        return input_
```

This works for TP where we own the `autograd.Function`s. But FSDP's all-gathers and reduce-scatters happen inside PyTorch's `fully_shard()` internals. DDP's gradient all-reduce happens inside `replicate()`. We can't instrument what we don't call.

The predict-measure-explain gap report needs to attribute *every* collective to a parallelism dimension — not just TP. If FSDP all-gathers account for 30% of step time, the gap report should say so. Currently it would be invisible.

### Solution: `InstrumentedProcessGroup` wrapper

Intercept collectives at the `ProcessGroup` level, below all framework layers:

```python
class InstrumentedProcessGroup:
    """Wraps a ProcessGroup to record all collective ops with CUDA timing."""

    def __init__(self, pg: ProcessGroup, mesh_dim: str, recorder: CommRecorder):
        self.pg = pg
        self.mesh_dim = mesh_dim
        self.recorder = recorder

    def all_reduce(self, tensor, op=ReduceOp.SUM):
        with self.recorder.measure(self.mesh_dim, "all_reduce", tensor.nbytes):
            return self.pg.all_reduce(tensor, op)

    def all_gather_into_tensor(self, output, input, **kwargs):
        with self.recorder.measure(self.mesh_dim, "all_gather", input.nbytes):
            return self.pg.all_gather_into_tensor(output, input, **kwargs)

    def reduce_scatter_tensor(self, output, input, **kwargs):
        with self.recorder.measure(self.mesh_dim, "reduce_scatter", input.nbytes):
            return self.pg.reduce_scatter_tensor(output, input, **kwargs)

    # delegate everything else unchanged
```

Install it by wrapping the process groups extracted from the `DeviceMesh` when profiling is enabled. The `mesh_dim` tag (e.g., `"tp"`, `"dp_shard"`, `"dp_replicate"`) lets the gap report attribute each collective to the right parallelism dimension.

This captures:
- TP comms from our `autograd.Function` pairs (they call `dist.all_reduce(group=...)`)
- FSDP all-gathers and reduce-scatters (they use the same process groups)
- DDP gradient syncs
- Future EP all-to-alls and PP send/recvs

The instrumentation in `comm.py` becomes redundant — the wrapper catches everything at a lower level. But the `comm.py` instrumentation can stay as a cross-check (same event should appear in both), which is useful for validating the wrapper during development.

---

## Problem 5: Separate FSDP and CP Mesh Dimensions Waste Bandwidth

This is a future concern (CP isn't implemented yet), but worth noting now because it affects `mesh.py` design.

With separate FSDP and CP dimensions, FSDP all-gathers run on a group of size `fsdp_size`. With `fsdp=4, cp=2`, that's 4 ranks. But the CP ranks have the same parameters (they differ only in which sequence chunk they process). They could participate in the all-gather, making it a group of size `fsdp * cp = 8` — which is more bandwidth-efficient because larger groups amortize latency better.

The fix is straightforward: keep the logical mesh N-dimensional (for sharding specs and the plan) but flatten FSDP + CP into one dimension for `fully_shard()`:

```python
def get_flattened_submesh(mesh: DeviceMesh, dims: list[str]) -> DeviceMesh:
    """Flatten multiple mesh dims into one for more efficient collectives.

    The logical mesh retains all dimensions (for sharding specs).
    The flattened submesh is used only for collective operations.
    """
    submesh = mesh[tuple(dims)]
    return submesh.flatten()

# Usage:
fsdp_cp_mesh = get_flattened_submesh(mesh, ["dp_shard", "cp"])
fully_shard(block, mesh=fsdp_cp_mesh)
```

No architectural changes needed — just a utility function in `mesh.py` and a one-line change in `data_parallel.py` where `fully_shard()` is called. Note for when CP lands.

---

## Summary

| Problem | Root Cause | Solution |
|---|---|---|
| Sharding plan can't express multi-dim | Flat `{suffix: "colwise"}` dict is TP-only | `ModuleSharding` dataclass with per-mesh-dim placements |
| Name-suffix dispatch breaks with heterogeneous models | Suffixes are ambiguous across model components | Type-based dispatch registry with suffix fallback |
| Annotations don't distinguish storage vs computation | Single sharding state per parameter | `ShardingInfo` with `storage` + `computation` placements |
| Profiler blind to FSDP/DDP collectives | Instrumentation only in `comm.py` | `InstrumentedProcessGroup` wrapping all mesh PGs |
| Separate FSDP+CP dims waste bandwidth | Smaller collective groups | Flatten for collectives, keep separate for specs |

All five are backward-compatible extensions. The core architecture holds — explicit comm primitives, pure models, `DeviceMesh` passed explicitly, SPMD type system. These changes extend the sharding *spec* and *profiling* layers without touching the comm or model layers.

---

## Problem 6: No Pre-Flight Validation Without GPUs

The [cost model entry](2026-02-25-cost-model-design.md) wants to answer "does it fit?" in milliseconds with zero GPUs. Currently this requires analytical formulas that approximate parameter counts (`12h²L`), which miss embeddings, LayerNorms, tied weights, and FSDP padding overhead. Meanwhile, the actual model definition — which knows *exactly* how many parameters there are and what shapes they have — can only be interrogated after building on a real device and initializing process groups.

PyTorch's meta device solves this. `torch.device("meta")` creates tensors with shapes but no storage — zero memory, no GPU needed. A model built on meta has every parameter shape available for inspection.

The problem is that the current parallelism application path conflates two things:

1. **Resolving the plan** — which modules get sharded, on which dimensions, producing what shapes
2. **Executing the plan** — slicing real tensors, installing comm-aware forwards, creating process groups

A meta-device dry run needs (1) without (2). Today these are fused together in `_shard_linear_colwise()` (shape arithmetic + tensor slicing + forward replacement in one function) and `apply_tensor_parallelism()` (plan matching + `DeviceMesh` extraction in one loop).

### Solution: `ModuleSharding` as the shared layer

The key observation: `ModuleSharding` from Problem 1 *is* the natural factoring point. It's a pure data structure describing what happens to shapes — no process groups, no real tensors. Both the dry run and the real path can consume it:

```python
# Shared: resolve which modules get which sharding
def resolve_sharding(module: nn.Module, parallel: ParallelConfig) -> ModuleSharding | None:
    """Look up sharding spec for a module. No side effects, no process groups."""
    shard_fn = SHARDING_REGISTRY.get(type(module))
    if shard_fn is not None:
        return shard_fn(module, parallel)
    return None

# Shared: compute local shape from global shape + spec
def compute_local_shape(
    global_shape: tuple[int, ...],
    placements: dict[str, Placement],
    parallel: ParallelConfig,
) -> tuple[int, ...]:
    """What shape does this parameter have on a single rank?"""
    local = list(global_shape)
    for mesh_dim, placement in placements.items():
        if isinstance(placement, Shard):
            degree = parallel.degree_for(mesh_dim)  # tp_size, dp_shard, etc.
            local[placement.dim] //= degree
    return tuple(local)
```

The dry run uses these to compute memory without GPUs:

```python
def dry_run(model_config: TransformerConfig, parallel: ParallelConfig, dtype) -> DryRunReport:
    with torch.device("meta"):
        model = DenseTransformer(model_config)

    params = []
    for name, module in model.named_modules():
        spec = resolve_sharding(module, parallel)
        if spec is None:
            continue
        for param_name, placements in spec.params.items():
            p = getattr(module, param_name, None)
            if p is None:
                continue
            local_shape = compute_local_shape(p.shape, placements, parallel)
            # ... accumulate into report
    # ... add FSDP padding, optimizer state multiplier, etc.
```

The real path uses the same functions, then does the actual work:

```python
def apply_parallelism(model, mesh, parallel):
    for name, module in model.named_modules():
        spec = resolve_sharding(module, parallel)   # same function as dry run
        if spec is not None:
            execute_sharding(module, spec, mesh)     # real path only: slice, install forward
```

`resolve_sharding()` and `compute_local_shape()` are shared. `execute_sharding()` is real-path only — it needs a `ProcessGroup` and real tensors. The dry run never touches execution.

### Why not share before `ModuleSharding` exists?

Right now the plan is a flat `DEFAULT_TP_STYLE` dict and the shape arithmetic is `dim // tp_size` — one line. A standalone `dry_run()` that reads `DEFAULT_TP_STYLE` directly (same dict, not duplicated) and reimplements the 5-line matching loop is fine. The drift risk is low because the plan is small and stable.

But once `ModuleSharding` and the type-based registry land (Problems 1 and 2), the shared layer emerges naturally. The registry is the single source of truth, and both paths are just different consumers of the same specs. Building a temporary shared abstraction before then means building something that gets replaced.

### What the dry run enables

- **Exact per-rank memory** — parameter bytes, gradient bytes, optimizer state, including FSDP padding overhead, weight tying dedup, and all the details that `12h²L` approximations miss
- **Divisibility validation** — `n_heads=48, tp_size=5` fails here, not 10 minutes into a job after process group init
- **Cost model integration** — feed exact sharded shapes into the analytical cost model instead of approximations. The cost model owns communication time, compute time, and MFU projection; the dry run provides the shapes those formulas operate on
- **Activation memory** (future) — run a meta-tensor forward pass to trace per-layer activation shapes, which the cost model currently can't compute without knowing the exact sequence of operations

---

## Summary

| Problem | Root Cause | Solution |
|---|---|---|
| Sharding plan can't express multi-dim | Flat `{suffix: "colwise"}` dict is TP-only | `ModuleSharding` dataclass with per-mesh-dim placements |
| Name-suffix dispatch breaks with heterogeneous models | Suffixes are ambiguous across model components | Type-based dispatch registry with suffix fallback |
| Annotations don't distinguish storage vs computation | Single sharding state per parameter | `ShardingInfo` with `storage` + `computation` placements |
| Profiler blind to FSDP/DDP collectives | Instrumentation only in `comm.py` | `InstrumentedProcessGroup` wrapping all mesh PGs |
| Separate FSDP+CP dims waste bandwidth | Smaller collective groups | Flatten for collectives, keep separate for specs |
| No pre-flight validation without GPUs | Plan resolution fused with execution | Meta-device dry run consuming `ModuleSharding` specs |

All six are backward-compatible extensions. The core architecture holds — explicit comm primitives, pure models, `DeviceMesh` passed explicitly, SPMD type system. These changes extend the sharding *spec* and *profiling* layers without touching the comm or model layers.

Problems 1 and 2 are the foundation — `ModuleSharding` and the type-based registry are prerequisites for everything else. Problem 6 (dry run) is a direct consumer of Problems 1 and 2. Problem 3 is a prerequisite for reshardable checkpoints. Problem 4 is a prerequisite for the predict-measure-explain loop covering all parallelism dimensions. Problem 5 is deferred until CP.
