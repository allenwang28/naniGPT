# Parallelism Architecture Design for naniGPT

*2026-03-18*

Previous entries established the two poles: Megatron encodes parallelism *into* model layers, torchtitan applies it *onto* models via DTensor. The [porting entry](2026-03-18-porting-megatron-optimizations-to-dtensor.md) showed how to bring Megatron optimizations into a DTensor-based stack. This entry asks the sharper question: **how do we retain the benefits of Megatron while keeping the things we like from torchtitan?**

The answer is a decoupling exercise. Megatron has everything needed for high MFU — the right communication primitives, the right overlap strategies, the right sharding patterns. But these are fused together in ways that make the code hard to modify, hard to extend, and hard for agents to work with. If we can identify exactly *what* is coupled and *why*, we can factor it apart without losing the performance properties.

Codebases studied:
- **Megatron-LM** @ [`32efeffd`](https://github.com/NVIDIA/Megatron-LM/tree/32efeffd2d3cca48f1bfa575e1f2d0f0feac3d0f)
- **torchtitan** @ [`main`](https://github.com/pytorch/torchtitan)
- **spmd_types** @ [`main`](https://github.com/ezyang/spmd_types/tree/main/sixlib/spmd_types) (Edward Yang's SPMD type system)

---

## Part 1: What We Like

### From Megatron

**1. Conjugate pair communication primitives.** The comm ops in `megatron/core/tensor_parallel/mappings.py` are `autograd.Function` subclasses that pair each forward op with its mathematically correct backward dual:

[`megatron/core/tensor_parallel/mappings.py#L197-L351`](https://github.com/NVIDIA/Megatron-LM/blob/32efeffd2d3cca48f1bfa575e1f2d0f0feac3d0f/megatron/core/tensor_parallel/mappings.py#L197-L351):

| Class | Forward | Backward |
|-------|---------|----------|
| `_CopyToModelParallelRegion` | identity | all-reduce |
| `_ReduceFromModelParallelRegion` | all-reduce | identity |
| `_GatherFromSequenceParallelRegion` | all-gather | reduce-scatter |
| `_ReduceScatterToSequenceParallelRegion` | reduce-scatter | all-gather |

These are ~20 lines each, elegant, and correct by construction. Backward is automatically right because the dual is encoded in the autograd graph.

**2. The Column+Row pairing.** A tensor-parallel transformer layer has exactly 2 all-reduces (or reduce-scatters) per layer in forward. Column-parallel on the first linear, row-parallel on the second. Everything between them is pure local compute. This is the minimal communication pattern.

**3. Explicit control over communication.** Every collective is visible in the code. You can see exactly where communication happens, profile it, and optimize it. No dispatch magic hiding collectives behind operator overloads.

**4. `ModuleSpec` for backend swapping.** The `ModuleSpec` / `BackendSpecProvider` pattern ([`megatron/core/transformer/spec_utils.py`](https://github.com/NVIDIA/Megatron-LM/blob/32efeffd2d3cca48f1bfa575e1f2d0f0feac3d0f/megatron/core/transformer/spec_utils.py), [`megatron/core/models/backends.py`](https://github.com/NVIDIA/Megatron-LM/blob/32efeffd2d3cca48f1bfa575e1f2d0f0feac3d0f/megatron/core/models/backends.py)) lets you swap `ColumnParallelLinear` for `TEColumnParallelLinear` by changing the spec. Good idea — decouple *what* from *how*.

### From torchtitan

**1. Pure model definitions.** The model is a standard `nn.Module` with `nn.Linear`. No `ColumnParallelLinear`, no `DistributedDataParallel` wrapper, no communication in the model. Parallelism is applied externally via `parallelize_module()`. You can test the model on a single GPU with no distributed setup.

**2. `DeviceMesh` as the abstraction.** No global state for process groups. The mesh is created once and passed explicitly. Adding a new parallelism dimension means adding a mesh dimension, not touching a global registry.

**3. Composable APIs.** `fully_shard()`, `replicate()`, `parallelize_module()` compose. You call them in sequence and they stack. No monolithic wrapper class that tries to handle all parallelism strategies.

### From spmd_types

**1. A type system that explains *why* communication patterns are correct.** Edward Yang's [spmd_types](https://github.com/ezyang/spmd_types) assigns each tensor one of four types per mesh axis:

| Type | Meaning | Gradient |
|------|---------|----------|
| **I** (Invariant) | Same value, same computation on all ranks | I |
| **R** (Replicate) | Same value, ranks may compute differently | P |
| **V** (Varying) | Different values per rank | V |
| **P** (Partial) | Pending sum across ranks | R |

The key property: **forward types determine backward types**. R in forward → P in backward (needs reduction). P in forward → R in backward. This is exactly why Megatron's conjugate pairs work: `CopyToParallelRegion` is `I→R` in forward, and the backward is `P→I` (all-reduce).

**2. Catches silent correctness bugs.** The rule `nonlinear(P) → FORBIDDEN` is the most valuable. If you apply GELU to a tensor that's `P@etp` (partial sum after a row-parallel matmul), the type checker catches it. Without the type system, this is a silent bug — wrong gradients, loss doesn't converge, no error message.

Propagation rules:

```
op(R, R) → R        op(I, I) → I        op(V, V) → V
op(R, V) → V        op(R, I) → forbidden without convert
linear(P) → P       nonlinear(P) → FORBIDDEN
P + P → P           P * P → FORBIDDEN (doesn't distribute over sums)
```

---

## Part 2: What's Coupled in Megatron (and How)

Megatron has all the right primitives. The problem is how they're wired together:

```
                        Megatron Dependency Graph
                        ========================

    ┌──────────────────────────────────────────────────────────┐
    │                  TransformerConfig                        │
    │              (100+ fields, god-object)                    │
    └──┬──────┬──────┬──────┬──────┬──────┬──────┬─────────────┘
       │      │      │      │      │      │      │
       ▼      ▼      ▼      ▼      ▼      ▼      ▼
    ┌──────┐┌────────────────────┐┌─────┐┌──────┐┌──────────┐
    │Attn  ││ColumnParallelLinear││ MLP ││MoE   ││TransLayer│
    │      ││RowParallelLinear   ││     ││Layer ││__init__  │
    │      ││                    ││     ││      ││200+ lines│
    │      ││ weight init        ││     ││      ││if MoE:   │
    │      ││ + forward matmul   ││     ││      ││ if CUDA: │
    │      ││ + comm (AR/RS)     ││     ││      ││  if TE:  │
    │      ││ + grad accum fusion││     ││      ││   if FP8:│
    │      ││ + async comm       ││     ││      ││    ...   │
    │      ││ + sharded_state_   ││     ││      ││         │
    │      ││   dict()           ││     ││      ││         │
    └──┬───┘└────────┬───────────┘└──┬──┘└──┬───┘└────┬─────┘
       │             │               │      │         │
       └──────┬──────┴───────┬───────┘      │         │
              │              │              │         │
              ▼              ▼              ▼         ▼
    ┌─────────────────────────────────────────────────────────┐
    │                  parallel_state.py                       │
    │          ~40 module-level global variables               │
    │  _TENSOR_MODEL_PARALLEL_GROUP = None                    │
    │  _DATA_PARALLEL_GROUP = None                            │
    │  _EXPERT_MODEL_PARALLEL_GROUP = None                    │
    │  ...                                                    │
    └─────────────────────────────────────────────────────────┘

    Every box depends on the god-object config.
    Every box reaches into global state for process groups.
    ColumnParallelLinear fuses 6 concerns into one class.
    TransformerLayer fuses all features via nested conditionals.
```

Here are the specific coupling points:

### Coupling 1: Communication ↔ Model Definition

`ColumnParallelLinear` is an `nn.Module` subclass that *is* the model's linear layer. It handles weight initialization, the forward matmul, gradient accumulation fusion, async communication, sequence parallelism variants, and checkpointing — all in one ~300 line class.

[`megatron/core/tensor_parallel/layers.py`](https://github.com/NVIDIA/Megatron-LM/blob/32efeffd2d3cca48f1bfa575e1f2d0f0feac3d0f/megatron/core/tensor_parallel/layers.py):

A transformer layer has ~15 lines of actual compute logic buried in ~1000 lines of parallelism machinery. You can't test the model's math without the parallelism, and you can't change the parallelism without understanding the model.

**What this costs:** Adding a new model architecture (e.g., MoE with a novel routing scheme) requires reimplementing it in terms of `ColumnParallelLinear`/`RowParallelLinear`. You're writing comm code when you should be writing model code.

### Coupling 2: Sharding Metadata ↔ Model Definition

Every module must implement [`sharded_state_dict()`](https://github.com/NVIDIA/Megatron-LM/blob/32efeffd2d3cca48f1bfa575e1f2d0f0feac3d0f/megatron/core/tensor_parallel/layers.py#L1062-L1072) correctly for its specific sharding pattern. For example, `ColumnParallelLinear.sharded_state_dict()` declares `{"weight": 0, "bias": 0}` — meaning both are sharded along axis 0 in the TP group. `RowParallelLinear` declares `{"weight": 1}` — sharded along axis 1, with bias replicated. Each module must know its own sharding layout and encode it in this method.

**What this costs:** Change a module's parameter layout → update its `sharded_state_dict()`. Add a new module → write a new `sharded_state_dict()`. The checkpoint system can't operate without intimate knowledge of every module's internals.

### Coupling 3: Process Groups ↔ Everything

[`megatron/core/parallel_state.py`](https://github.com/NVIDIA/Megatron-LM/blob/32efeffd2d3cca48f1bfa575e1f2d0f0feac3d0f/megatron/core/parallel_state.py) has ~40 module-level global variables:

```python
_MODEL_PARALLEL_GROUP = None
_TENSOR_MODEL_PARALLEL_GROUP = None
_PIPELINE_MODEL_PARALLEL_GROUP = None
_DATA_PARALLEL_GROUP = None
_DATA_PARALLEL_GROUP_GLOO = None
_CONTEXT_PARALLEL_GROUP = None
_CONTEXT_PARALLEL_GLOBAL_RANKS = None
_EMBEDDING_GROUP = None
_POSITION_EMBEDDING_GROUP = None
_EXPERT_MODEL_PARALLEL_GROUP = None
# ... ~30 more
```

Every module that does communication calls `get_tensor_model_parallel_group()` — a function that reads a module-level global. Adding a new parallelism dimension (e.g., expert TP) requires adding globals to this file *and* updating every consumer.

**What this costs:** The process group topology is invisible at the call site. You can't tell from reading a module which groups it uses or how they relate to each other. Testing requires initializing the global state. Composing parallelism strategies requires ensuring all globals are set correctly.

### Coupling 4: Config ↔ Everything

[`megatron/core/transformer/transformer_config.py`](https://github.com/NVIDIA/Megatron-LM/blob/32efeffd2d3cca48f1bfa575e1f2d0f0feac3d0f/megatron/core/transformer/transformer_config.py) is a 100+ field dataclass that every constructor takes. It contains model hyperparameters, parallelism settings, precision flags, recompute strategies, TE options, and more.

**What this costs:** You can't understand what a module actually reads from the config without grep. Modules have implicit dependencies on fields they may never use. Changing a config field's semantics is a codebase-wide change.

### Coupling 5: Feature Interactions in TransformerLayer

[`megatron/core/transformer/transformer_layer.py`](https://github.com/NVIDIA/Megatron-LM/blob/32efeffd2d3cca48f1bfa575e1f2d0f0feac3d0f/megatron/core/transformer/transformer_layer.py) — `__init__` is 200+ lines of conditional logic:

```
if MoE:
    if CUDA graphs:
        if TE:
            if FP8:
                ...
```

MoE, CUDA graphs, Transformer Engine, FP8, recompute, and sequence parallelism all interact, and the interactions are encoded as nested conditionals. Features can't be understood in isolation.

**What this costs:** Adding a new feature (say, context parallelism) requires understanding how it interacts with every other feature flag. The combinatorial explosion makes the code fragile — a change for one feature can break others.

---

## Part 3: The Decoupling

```
                        naniGPT Dependency Graph
                        ========================

    ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
    │ ModelConfig   │  │ ParallelPlan │  │ TrainingCfg  │  │ EvalConfig   │
    │ (d_model,     │  │ (tp=8, dp=4, │  │ (lr, batch,  │  │ (val_interval│
    │  n_heads,     │  │  ep=1,       │  │  dtype)      │  │  val_steps)  │
    │  n_layers)    │  │  sp=True)    │  │              │  │              │
    └──────┬───────┘  └──────┬───────┘  └──────────────┘  └──────────────┘
           │                 │            each config owns one concern
           │                 │
           ▼                 ▼
    ┌──────────────┐  ┌──────────────────────────────────────────────────┐
    │   models/    │  │                  parallel/                       │
    │              │  │                                                  │
    │ nn.Linear    │  │  ┌────────┐  ┌─────────────────┐  ┌──────────┐ │
    │ nn.Module    │  │  │comm.py │  │tensor_parallel.py│  │  plan.py │ │
    │              │  │  │        │  │                  │  │          │ │
    │ pure math,   │  │  │CopyTo  │  │col_parallel_lin()│  │ParPlan  │ │
    │ no comm,     │  │  │Reduce  │  │row_parallel_lin()│  │ tp_plan │ │
    │ no sharding  │  │  │Gather  │  │                  │  │ ep_plan │ │
    │              │  │  │Scatter │  │  uses comm.py ──►│  │         │ │
    │              │  │  │AllToAll│  │  functions, not   │  │declares │ │
    │              │  │  │        │  │  modules          │  │intent   │ │
    │              │  │  └────────┘  └──────────────────┘  └─────────┘ │
    │              │  │                                                  │
    │              │  │  ┌─────────────────┐  ┌───────────────────────┐ │
    │              │  │  │expert_parallel.py│  │ Backends (Protocol)   │ │
    │              │  │  │                  │  │                       │ │
    │              │  │  │ EPBackend proto  │  │ NativeBackend         │ │
    │              │  │  │ AllToAllBackend  │  │ AsyncBackend          │ │
    │              │  │  │ DeepEPBackend   │  │ CompiledBackend       │ │
    │              │  │  └─────────────────┘  └───────────────────────┘ │
    └──────────────┘  └──────────────────────────────────────────────────┘
           │                 │
           │                 │  apply_tensor_parallelism(model, plan, mesh)
           │                 │  stamps annotations at apply time
           ▼                 ▼
    ┌────────────────────────────────────────────────────────────────────┐
    │                        sharding/                                   │
    │                                                                    │
    │  annotations.py: param.sharding_info = ShardingInfo(...)          │
    │  schema.py:      ShardedModelSchema (from walking annotations)    │
    │  resharder.py:   canonical-space resharding at load time          │
    │                                                                    │
    │  No per-module methods. Checkpointing reads tensor metadata.      │
    └────────────────────────────────────────────────────────────────────┘
           │
           ▼
    ┌────────────────────────────────────────────────────────────────────┐
    │  DeviceMesh  (passed explicitly, no globals)                       │
    │  mesh.get_group("tp")  mesh.get_group("ep")  mesh.get_group("dp") │
    └────────────────────────────────────────────────────────────────────┘

    models/ knows nothing about parallelism.
    parallel/ knows nothing about checkpointing.
    sharding/ knows nothing about communication.
    DeviceMesh is passed, never global.
```

Four concerns are entangled in Megatron. Here's how they separate:

| Concern | Megatron | torchtitan | naniGPT (proposed) |
|---------|----------|------------|-------------------|
| **Communication primitives** | `autograd.Function` pairs in `mappings.py` | DTensor dispatch (automatic) | Megatron's `autograd.Function` pairs (explicit) |
| **Model definition** | Fused with comm (`ColumnParallelLinear`) | Separated (pure `nn.Module`) | Separated (pure `nn.Module`) |
| **Sharding metadata** | Per-module `sharded_state_dict()` | DTensor placements | Tensor-level annotations (stamped at apply time) |
| **Correctness verification** | None | None | `spmd_types` assertions |

We take Megatron's comm primitives (explicit, profileable, optimizable) and torchtitan's model separation (pure `nn.Module`, parallelism applied externally). We add tensor-level sharding annotations so checkpointing doesn't need per-module methods. And we layer spmd_types on top for verification.

```
nanigpt/
├── models/
│   └── dense_transformer.py          # Concern 2: pure nn.Module, no parallelism
│
├── parallel/
│   ├── comm.py                       # Concern 1: autograd.Function conjugate pairs
│   ├── spmd.py                       # Concern 4: R/I/V/P type tracking
│   ├── mesh.py                       # DeviceMesh creation, no globals
│   ├── tensor_parallel.py            # TP: column/row parallel as functions
│   ├── expert_parallel.py            # EP: dispatch/combine strategies
│   ├── pipeline_parallel.py          # PP: schedule IR + execution
│   ├── data_parallel.py              # DP/FSDP: wraps fully_shard()
│   ├── context_parallel.py           # CP: sequence sharding for attention
│   └── plan.py                       # What gets parallelized how
│
├── sharding/
│   ├── annotations.py                # Concern 3: ShardingInfo on tensors
│   ├── schema.py                     # ShardedModelSchema for checkpointing
│   └── resharder.py                  # Resharding at load time
│
├── distributed.py                    # Process group lifecycle
├── config.py                         # Config tree
└── train.py                          # Training loop
```

Each decoupling addresses a specific Megatron coupling:

| Megatron coupling | naniGPT solution |
|---|---|
| Comm ↔ Model (ColumnParallelLinear) | TP is functions applied to `nn.Linear`, not replacement modules |
| Sharding ↔ Model (sharded_state_dict) | Annotations stamped on tensors when parallelism is applied |
| Process groups ↔ Everything (globals) | `DeviceMesh` passed explicitly, groups extracted at call site |
| Config ↔ Everything (god-object) | Small, focused config dataclasses per concern |
| Feature interactions (nested conditionals) | Backends as a Protocol, features compose via function calls |

---

## Part 4: The Design

### Communication Primitives (`parallel/comm.py`)

Directly from Megatron. These are the atoms — `autograd.Function` subclasses that pair forward comm with backward comm:

```python
class CopyToParallelRegion(torch.autograd.Function):
    """Forward: identity.  Backward: all-reduce."""
    @staticmethod
    def forward(ctx, input_, group):
        ctx.group = group
        return input_
    @staticmethod
    def backward(ctx, grad_output):
        dist.all_reduce(grad_output, group=ctx.group)
        return grad_output, None

class ReduceFromParallelRegion(torch.autograd.Function):
    """Forward: all-reduce.  Backward: identity."""

class GatherFromSequenceParallelRegion(torch.autograd.Function):
    """Forward: all-gather along dim 0.  Backward: reduce-scatter."""

class ReduceScatterToSequenceParallelRegion(torch.autograd.Function):
    """Forward: reduce-scatter along dim 0.  Backward: all-gather."""

class AllToAll(torch.autograd.Function):
    """Forward: all-to-all.  Backward: inverse all-to-all (swapped splits)."""
```

Each is ~20 lines. The entire file is maybe 150 lines. This is the one thing Megatron gets exactly right — take it wholesale.

### TP as Functions, Not Layers (`parallel/tensor_parallel.py`)

This is the key decoupling from `ColumnParallelLinear`. Instead of replacing `nn.Linear` with a parallel-aware module, TP is a set of functions that wrap standard linear ops using the comm primitives:

```python
def column_parallel_linear(input, weight, bias, group, sequence_parallel=False):
    """Column-parallel: shard output dim. Forward: local matmul. Backward: all-reduce dgrad."""
    # SPMD: input is I@tp (or V@tp if SP), weight is V@tp, output is V@tp
    if sequence_parallel:
        input = gather_from_seq_parallel_region(input, group)   # V→R
    else:
        input = copy_to_parallel_region(input, group)           # I→R
    return F.linear(input, weight, bias)                        # R × V → V

def row_parallel_linear(input, weight, bias, group, sequence_parallel=False):
    """Row-parallel: shard input dim. Forward: local matmul + reduce. Backward: identity."""
    # SPMD: input is V@tp, weight is V@tp, matmul output is P@tp
    output = F.linear(input, weight, bias)                      # V × V → P
    if sequence_parallel:
        return reduce_scatter_to_seq_parallel_region(output, group)  # P→V
    else:
        return reduce_from_parallel_region(output, group)            # P→I
```

The model stays a pure `nn.Module`. TP is applied by replacing the forward of specific `nn.Linear` modules to call these functions, and slicing weight tensors at initialization. The SPMD type comments document the invariants that the comm primitives maintain.

Here's the data flow through a single TP transformer layer, showing which file owns each step:

```
  Forward pass through one TransformerBlock (TP=8, sequence_parallel=True)
  ═══════════════════════════════════════════════════════════════════════

  Input x: [S/TP, B, H]                              SPMD: V@tp
  │
  │  ┌─ models/dense_transformer.py ──────────────────────────────────┐
  │  │  ln1 = LayerNorm(x)                            V@tp           │
  │  └────────────────────────────────────────────────────────────────┘
  │
  │  ┌─ parallel/tensor_parallel.py (colwise) ────────────────────────┐
  │  │                                                                 │
  │  │  ┌─ comm.py: AllGather ─┐                                      │
  │  │  │  V@tp → R@tp         │  all-gather across TP group          │
  │  │  └──────────────────────┘                                      │
  │  │                                                                 │
  │  │  q = F.linear(x_gathered, W_q)    R × V → V   local matmul    │
  │  │  k = F.linear(x_gathered, W_k)    R × V → V                   │
  │  │  v = F.linear(x_gathered, W_v)    R × V → V                   │
  │  └────────────────────────────────────────────────────────────────┘
  │
  │  ┌─ models/dense_transformer.py ──────────────────────────────────┐
  │  │  attn_out = attention(q, k, v)                 V@tp           │
  │  │  (pure local compute — each rank has its own heads)            │
  │  └────────────────────────────────────────────────────────────────┘
  │
  │  ┌─ parallel/tensor_parallel.py (rowwise) ────────────────────────┐
  │  │  proj = F.linear(attn_out, W_out)  V × V → P  local matmul   │
  │  │                                                                 │
  │  │  ┌─ comm.py: ReduceScatter ─┐                                 │
  │  │  │  P@tp → V@tp             │  reduce-scatter across TP group │
  │  │  └──────────────────────────┘                                  │
  │  └────────────────────────────────────────────────────────────────┘
  │
  │  ┌─ models/dense_transformer.py ──────────────────────────────────┐
  │  │  x = x + proj                                  V@tp           │
  │  │  ln2 = LayerNorm(x)                            V@tp           │
  │  └────────────────────────────────────────────────────────────────┘
  │
  │  ┌─ parallel/tensor_parallel.py (colwise) ────────────────────────┐
  │  │  AllGather V→R, then ffn_up = F.linear(x, W_up)   → V@tp     │
  │  └────────────────────────────────────────────────────────────────┘
  │
  │  ┌─ models/dense_transformer.py ──────────────────────────────────┐
  │  │  ffn_act = GELU(ffn_up)                        V@tp           │
  │  └────────────────────────────────────────────────────────────────┘
  │
  │  ┌─ parallel/tensor_parallel.py (rowwise) ────────────────────────┐
  │  │  ffn_down = F.linear(ffn_act, W_down), then ReduceScatter P→V │
  │  └────────────────────────────────────────────────────────────────┘
  │
  │  ┌─ models/dense_transformer.py ──────────────────────────────────┐
  │  │  x = x + ffn_down                              V@tp           │
  │  └────────────────────────────────────────────────────────────────┘
  │
  Output x: [S/TP, B, H]                             SPMD: V@tp ✓

  ───────────────────────────────────────────────────────────────────
  Total communication: 2 AllGathers + 2 ReduceScatters per layer
  Model code (dense_transformer.py): unchanged, no comm awareness
  Comm code (comm.py): 4 autograd.Function calls, ~80 lines total
  Wiring (tensor_parallel.py): function calls, not module replacements
```

### The Parallelism Plan (`parallel/plan.py`)

Declares *what* gets parallelized *how*, without doing any of the work:

```python
@dataclass
class TensorParallelPlan:
    """Which modules get which TP treatment."""
    style: dict[str, Literal["colwise", "rowwise", "replicate"]]
    # Keys are module name suffixes, e.g. "attn.q_proj": "colwise"

@dataclass
class ExpertParallelPlan:
    """How MoE experts are distributed."""
    dispatch_strategy: Literal["allgather", "alltoall"]
    expert_tp_size: int = 1  # can differ from attention TP

@dataclass
class ParallelPlan:
    tp_size: int = 1
    pp_size: int = 1
    dp_size: int = 1        # inferred: world_size / (tp * pp * cp)
    cp_size: int = 1
    ep_size: int = 1
    sequence_parallel: bool = False

    tp_plan: TensorParallelPlan = field(default_factory=lambda: TensorParallelPlan(
        style={
            "attn.q_proj": "colwise",
            "attn.k_proj": "colwise",
            "attn.v_proj": "colwise",
            "attn.out_proj": "rowwise",
            "ffn.up": "colwise",
            "ffn.down": "rowwise",
        }
    ))

    expert_plan: ExpertParallelPlan | None = None
```

The plan is applied by `apply_tensor_parallelism(model, plan, mesh)`, which walks the model, shards weights, replaces forwards with TP-aware versions, and stamps sharding annotations — all in one pass.

### Sharding Annotations (`sharding/annotations.py`)

This decouples checkpointing from model internals. Instead of per-module `sharded_state_dict()` methods, each parameter carries metadata about its position in canonical (unsharded) space:

```python
@dataclass(frozen=True)
class TensorParallelShard:
    """Sharded across a parallel group on the given dimension."""
    dim: int
    unpadded_shape: tuple[int, ...] | None = None

@dataclass(frozen=True)
class Replicated:
    """Replicated across all ranks in the parallel group."""

@dataclass(frozen=True)
class FusedShards:
    """Concatenation of multiple logical tensors with different shard rules.
    E.g. fused QKV where Q is fully TP-sharded but K/V are partially shared (GQA).
    """
    components: tuple[FusedComponent, ...]

@dataclass(frozen=True)
class StackedShards:
    """Stack of sub-tensors along a new leading dimension.
    E.g. N MoE experts stacked as [N, H, FFN_H], where the stack dim is
    partitioned across EP and the FFN_H dim is partitioned across expert-TP.
    """
    stack_dim_group: ProcessGroup
    inner_shard: TensorParallelShard | Replicated

@dataclass(frozen=True)
class ShardingInfo:
    """Stamped on every parameter as `param.sharding_info`."""
    shard_type: TensorParallelShard | Replicated | FusedShards | StackedShards
    global_shape: tuple[int, ...]
    rank: int
    group_size: int

def annotate(param: nn.Parameter, shard_type, group: ProcessGroup):
    """Stamp sharding metadata on a parameter."""
    param.sharding_info = ShardingInfo(
        shard_type=shard_type,
        global_shape=_reconstruct_global_shape(param, shard_type, group),
        rank=dist.get_rank(group),
        group_size=dist.get_world_size(group),
    )
```

Annotations are set when parallelism is applied, not when the model is constructed. Checkpointing reads annotations to build a `ShardedModelSchema` mapping local slices to canonical slices. Resharding between different parallelism configs is automatic — the resharder computes slice intersections in canonical space. New model → annotations get stamped automatically → checkpointing just works.

### DeviceMesh — No Globals (`parallel/mesh.py`)

This directly addresses Megatron's `parallel_state.py` globals:

```python
def create_device_mesh(world_size: int, plan: ParallelPlan) -> DeviceMesh:
    """Create a multi-dimensional DeviceMesh from the parallelism plan.

    Rank layout: TP innermost (NVLink), PP outermost (cross-node).
    """
    dims, names = [], []
    for name, size in [("pp", plan.pp_size), ("dp", plan.dp_size),
                       ("ep", plan.ep_size), ("tp", plan.tp_size)]:
        if size > 1:
            dims.append(size)
            names.append(name)

    return DeviceMesh("cuda", torch.arange(world_size).reshape(*dims),
                      mesh_dim_names=tuple(names))
```

No globals. Created once in `train.py`, passed explicitly. Process groups extracted via `mesh.get_group("tp")` at the call site. Adding a new parallelism dimension means adding one entry to the loop above.

### Backend Dispatch (`parallel/tensor_parallel.py`)

This addresses Megatron's feature interaction problem. Instead of nested conditionals, backends are a Protocol:

```python
class TPBackend(Protocol):
    def column_parallel_linear(self, input, weight, bias, group, sp) -> Tensor: ...
    def row_parallel_linear(self, input, weight, bias, group, sp) -> Tensor: ...

class NativeBackend:
    """Pure PyTorch with explicit comm calls. Readable, debuggable."""

class AsyncBackend:
    """Megatron-style async overlap. Uses CUDA_DEVICE_MAX_CONNECTIONS=1.
    Overlaps dgrad computation with wgrad all-gather in backward."""

class CompiledBackend:
    """torch.compile with _micro_pipeline_tp. Let the compiler find overlap."""
```

You can profile the delta at each level — native → compiled → async — and understand exactly what each optimization buys. Debug with `NativeBackend`, train with `AsyncBackend`. No conditional explosion.

---

## Part 5: Stress Test — MoE

MoE is the stress test because it requires all four concerns to work together, adds new parallelism dimensions (EP, expert-TP), and has subtle correctness requirements. After tracing Megatron's MoE stack in detail, here's how the SPMD types flow through a complete MoE forward pass.

### The MoE Type Trace

On three axes: `@tp` (attention TP), `@ep` (expert parallel), `@etp` (expert TP). AlltoAll dispatcher path:

```
Step                              @tp    @ep    @etp   Comm
──────────────────────────────────────────────────────────────
Input [S/TP, B, H]                V      R      —      —

Router: F.linear(x, W_gate)       V      R      —      —
  W_gate [E, H] is I@tp, I@ep
  V × I → V

Top-k routing                     V      R      —      —

Permute by expert assignment      V      R      —      —
  (local reordering, no comm)

All-to-All(ep)                    V      V      —      all-to-all
  V@ep → V@ep
  "my local tokens" → "tokens for my experts"

AllGather(etp)                    V      V      R      all-gather
  V@etp → R@etp

Expert FC1: gmm(x, W1)            V      V      V      —
  R@etp × V@etp → V@etp
  W1 is [E_local, H, FFN_H/ETP]

Activation: SiLU(x)               V      V      V      —
  V@etp → V@etp  (OK: nonlinear on V, not P)

Expert FC2: gmm(x, W2)            V      V      P      —
  V@etp × V@etp → P@etp
  W2 is [E_local, FFN_H/ETP, H]

ReduceScatter(etp)                V      V      V      reduce-scatter
  P@etp → V@etp

All-to-All(ep, inverse)           V      R      —      all-to-all
  "my experts' results" → "my tokens' results"

Unpermute + prob weighting        V      R      —      —

Output [S/TP, B, H]               V      R      —      —
  Same types as input ✓
```

### What the Type System Catches

**Bug 1: Nonlinear on Partial.** After expert FC2, the output is `P@etp`. If someone applied GELU after FC2 instead of between FC1 and FC2:

```python
fc2_out = gmm(fc1_output, w2)    # P@etp
output = F.gelu(fc2_out)         # spmd_types: ERROR — nonlinear(P) forbidden
```

Silent correctness bug without the type system. Code runs, wrong gradients.

**Bug 2: Forgetting ReduceScatter after expert compute.**

```python
fc2_out = gmm(act_out, w2)       # P@etp
combined = all_to_all(ep, fc2_out)  # spmd_types: all-to-all expects V, got P
```

**Bug 3: The AllGather dispatcher's combine step.** After unpermute in the AllGather path, each EP rank has nonzeros only at its local experts' token positions. These need to be *summed* across EP ranks (it's `P@ep`), not concatenated. The type system distinguishes: `P` → valid transitions are `all_reduce(P→I)` or `reduce_scatter(P→V)`.

### What MoE Demands From the Design

**The plan needs `ExpertParallelPlan`.** Dense TP uses `colwise`/`rowwise` on a single TP group. Expert weights are 3D tensors on different groups (EP for the expert dimension, ETP for the FFN hidden dimension).

**The annotation system needs `StackedShards`.** An expert weight `[num_local_experts, H, FFN_H/ETP]` is sharded on two axes simultaneously. `TensorParallelShard(dim=N)` only describes one axis.

**Expert TP and attention TP must be separate mesh axes.** The `DeviceMesh` needs enough dimensions to express this:

```python
mesh = DeviceMesh("cuda", ranks.reshape(pp, dp, ep, tp),
                  mesh_dim_names=("pp", "dp", "ep", "tp"))
# Expert TP might be a sub-mesh of TP, or a separate dimension
```

**The comm primitives are sufficient.** MoE uses the same `all_gather`, `reduce_scatter`, and `all_to_all` primitives as dense TP. No new `autograd.Function` subclasses needed — just new wiring in `expert_parallel.py`.

All of these are additive. The core design handles MoE without architectural changes — just new plan types and annotation types.

---

## Part 6: How It Composes

The full wiring in `train.py`:

```python
# 1. Build plan
plan = ParallelPlan(
    tp_size=8, pp_size=1, dp_size=4, ep_size=1,
    sequence_parallel=True,
    tp_plan=TensorParallelPlan(style={
        "attn.q_proj": "colwise", "attn.k_proj": "colwise",
        "attn.v_proj": "colwise", "attn.out_proj": "rowwise",
        "ffn.up": "colwise", "ffn.down": "rowwise",
    }),
)

# 2. Create mesh (no globals)
mesh = create_device_mesh(world_size=32, plan=plan)

# 3. Build model (pure nn.Module, no parallelism)
model = DenseTransformer(config).to(device)

# 4. Apply parallelism (shards weights, replaces forwards, stamps annotations)
apply_tensor_parallelism(model, plan, mesh)

# 5. Apply data parallelism
apply_data_parallelism(model, mesh, strategy="fsdp")

# 6. Checkpoint reads param.sharding_info — no model-specific code
save_checkpoint(model, path)      # writes sharding_metadata.json
load_checkpoint(model, path)      # reshards automatically if plan changed
```

For MoE:

```python
plan = ParallelPlan(
    tp_size=8, pp_size=1, dp_size=2, ep_size=4,
    expert_plan=ExpertParallelPlan(
        dispatch_strategy="alltoall",
        expert_tp_size=2,
    ),
    ...
)
```

---

## Part 7: Agent-Maintainability

The whole point of this decoupling is that features map to files:

| Task | Files to modify |
|------|----------------|
| Port a new PP schedule from Megatron | `parallel/pipeline_parallel.py` only |
| Port a new dispatch strategy (e.g., DeepEP) | `parallel/expert_parallel.py` only |
| Add a new model architecture | `models/` only (no parallelism code) |
| Port a comm optimization (e.g., async backward overlap) | `parallel/comm.py` + backend in `tensor_parallel.py` |
| Support a new checkpoint format | `sharding/` only |

In Megatron, each of these tasks touches 5+ files with non-obvious interactions. The decoupled design means an agent can read a Megatron optimization, identify which concern it belongs to, and implement it in the right module without understanding the entire codebase. As Megatron lands new optimizations, agents can port them into the corresponding naniGPT module — keeping us up to date without accumulating Megatron's coupling.

---

## Part 8: Pipeline Parallelism

PP is fundamentally different from TP/EP/DP because it splits the model *sequentially* across devices — rank 0 has layers 0–7, rank 1 has layers 8–15. This has implications the other parallelisms don't:

1. **Model construction changes.** Each rank only instantiates its layers. The model is no longer whole on any single device.

2. **The training loop changes.** Instead of forward→backward→step, you run a *schedule* — 1F1B, interleaved, zero-bubble — that overlaps micro-batches across pipeline stages.

3. **Communication is point-to-point, not collective.** PP uses `send`/`recv` between adjacent stages, not all-reduce/all-gather across a group. Different primitives than what TP/EP use.

4. **Activation memory management.** You stash activations from forward micro-batches until their backward arrives. This interacts with activation checkpointing.

5. **The plan needs a layer-to-stage mapping.** Which layers go on which stage?

### Does PP require changes to the existing architecture?

Mostly additive:
- `parallel/pipeline_parallel.py` handles schedule execution (1F1B, etc.) and send/recv comms
- `parallel/plan.py` gains a layer-to-stage mapping (or a simple `num_layers_per_stage`)
- `parallel/comm.py` gains `send`/`recv` autograd.Function pairs (PP's point-to-point duals)
- `train.py` calls a different training loop when `pp_size > 1`
- The model needs a way to instantiate only its stage's layers

It doesn't change how TP or EP work *within a stage*. Each stage is a subset of layers with TP/EP/DP applied normally. PP composes *around* the other parallelisms.

The DeviceMesh handles the composition naturally — `("pp", "dp", "ep", "tp")` with TP innermost (NVLink) and PP outermost (cross-node). Each PP stage has its own TP/EP/DP groups.

### What PP adds to the directory tree

```
parallel/
├── comm.py                    # gains Send/Recv autograd.Function pairs
├── pipeline_parallel.py       # NEW: schedule IR + execution
│   ├── PipelineSchedule (1F1B, interleaved, zero-bubble)
│   ├── micro-batch management
│   └── activation stashing
└── plan.py                    # gains pp_size + stage mapping
```

### What PP demands from the plan

```python
@dataclass
class PipelineParallelPlan:
    """How to split layers across pipeline stages."""
    num_stages: int = 1
    # Even split by default; override for uneven partitioning
    layers_per_stage: list[int] | None = None
    num_micro_batches: int = 1
    schedule: Literal["1f1b", "interleaved", "zero_bubble"] = "1f1b"
```

### Schedule as IR

Rather than hardcoded schedule functions (Megatron has separate functions for 1F1B and interleaved), schedules can be represented as an IR of atomic actions:

```python
@dataclass
class ScheduleAction:
    type: Literal["FORWARD", "BACKWARD", "SEND_F", "RECV_F", "SEND_B", "RECV_B"]
    micro_batch: int
    stage: int

# 1F1B for stage 0, 4 micro-batches:
# [RECV_F(0), FORWARD(0), SEND_F(0),
#  RECV_F(1), FORWARD(1), SEND_F(1),
#  RECV_F(2), FORWARD(2), SEND_F(2),
#  RECV_F(3), FORWARD(3), SEND_F(3),
#  RECV_B(3), BACKWARD(3), SEND_B(3),
#  RECV_B(2), BACKWARD(2), SEND_B(2), ...]
```

This is more composable than hardcoded functions — you can transform schedules (e.g., merge BACKWARD+FORWARD into overlapped operations) and add new schedules without touching the executor.

---

## Resolved Questions

*2026-03-27 addendum*

### 1. Expert TP: sub-mesh of TP, not a separate dimension

**Resolution:** Expert TP is a subdivision of the TP dimension, not an independent mesh axis. With TP=8 and expert_tp=2, the 8 GPUs in a TP group form 4 ETP sub-groups of size 2.

Adding ETP as a separate mesh dimension would force the rank layout to be `(pp, dp, ep, etp, tp)`, which is wrong — ETP isn't independent of TP, it's *within* TP. `DeviceMesh` sub-meshes or manual `ProcessGroup` creation from rank subsets handles this. The `ExpertParallelPlan.expert_tp_size` field tells `expert_parallel.py` how to subdivide at apply time.

### 2. torch.compile + explicit comm primitives: no design risk

**Resolution:** Three compatible options, all available without architectural changes:

- **Accept graph breaks at comm boundaries.** Comms are natural break points. The valuable compilation happens on the compute between comms (matmuls, attention, activations). Loses very little.
- **Upgrade to `torch.library.custom_op`.** Mechanical transformation of the `autograd.Function` versions in `comm.py`. The compiler sees them as opaque but doesn't break the graph. DeepEP already does this (`torch.ops.deepep.dispatch`/`combine`).
- **Compiler-driven overlap** via `CompiledBackend`. Let `torch.compile` find comm/compute overlap opportunities. Already has a slot in the backend Protocol.

The explicit comm approach doesn't conflict with compilation. If needed, upgrading from `autograd.Function` to `custom_op` is a change inside `comm.py` only — nothing upstream changes.

### 3. spmd_types: documentation + assertion mode

**Resolution:** Two levels, both zero-architecture-cost:

| Level | What | Cost | When |
|---|---|---|---|
| Type comments | `# V@tp`, `# P@etp → V@etp` inline in TP/EP functions | Zero | Always |
| `assert_type()` | Checks at module boundaries (entry/exit of TP functions, dispatch/combine) | Negligible | Enabled by flag |

Full propagation mode (`TorchFunctionMode` intercepting every op) is a debugging tool, not a training-time feature. Enable it temporarily to verify a new parallelism configuration, then turn it off.

This doesn't affect the architecture — `spmd.py` is a leaf module with no dependents.

### 4. Heterogeneous layers: module name suffixes are sufficient

**Resolution:** MoE and dense layers have different module structures — a dense layer has `layers.3.ffn.up`, a MoE layer has `layers.4.moe.experts`. The plan's suffix matching already distinguishes them without needing per-layer-index overrides:

```python
style={
    "attn.q_proj": "colwise",      # matches all layers
    "ffn.up": "colwise",           # matches dense layers (which have .ffn)
    "moe.experts": "expert_plan",  # matches MoE layers (which have .moe)
}
```

If a future model requires two layers with the *same* module structure but *different* parallelism strategies, the plan can gain a `layer_overrides: dict[int, TensorParallelPlan]` field. This is a backward-compatible addition, not needed now.
