# Analytical Cost Model for naniGPT

*2026-02-25*

The [parallelism strategies entry](2026-02-24-parallelism-strategies.md) derives all the formulas — per-layer FLOPs, communication bytes per collective, bubble fractions, memory breakdowns. This entry explores making those formulas executable: a tool that takes a training config and tells you where the time and memory will go, before you run anything.

---

## What problem this solves

Right now, choosing a parallelism config is trial-and-error. You pick `tp=4, dp=8`, launch the job, wait for it to start, maybe OOM, adjust, relaunch. A cost model answers these questions in milliseconds with zero GPUs:

- **Does it fit?** Memory breakdown: params + grads + optimizer state + activations. Per-rank, after sharding.
- **Where's the bottleneck?** Compute time vs communication time per parallelism dimension. Is TP comm-bound on this interconnect? Is the PP bubble eating 20% of throughput?
- **What's the projected MFU?** Theoretical peak vs what you'd actually achieve given the communication overhead.
- **How should I parallelize?** Sweep `tp × dp × pp` configurations for a given model size and GPU count, rank them by projected throughput.

The gap between projected and measured MFU is itself interesting — it tells you where the analytical model breaks down (kernel launch overhead, memory bandwidth bottlenecks, scheduling inefficiency, network congestion).

---

## End-to-end experience

```python
from nanigpt.cost_model import CostModel, HardwareConfig

# Define hardware
h100_sxm = HardwareConfig(
    flops_bf16=989e12,         # 989 TFLOPS peak bf16
    hbm_bandwidth=3.35e12,     # 3.35 TB/s
    hbm_capacity=80e9,         # 80 GB
    nvlink_bandwidth=450e9,    # 450 GB/s per direction
    ib_bandwidth=50e9,         # 400 Gb/s = 50 GB/s per direction
)

# Use an existing training config (our config system uses dataclasses + tyro, not files)
from nanigpt.configs.registry import REGISTRY
train_config = REGISTRY["70b-synthetic"]()
model = CostModel.from_train_config(train_config, hardware=h100_sxm)

# Project costs for a specific parallelism config
report = model.project(tp=8, dp=16, pp=4, cp=1, num_gpus=512)

print(report)
```

```
Model: 70B (64 layers, h=8192, s=4096)
GPUs: 512 × H100 SXM (tp=8, dp=16, pp=4)

Memory per rank:
  Parameters:       4.38 GB  (70B / 512 ranks × 2 bytes bf16... but FSDP shards across dp)
  Gradients:        4.38 GB
  Optimizer (Adam): 17.50 GB  (4× params for fp32 master + momentum + variance)
  Activations:      2.10 GB  (pp=4 microbatches in flight)
  ─────────────────────────
  Total:            28.36 GB  ✓ fits in 80 GB

Compute per step:
  Forward FLOPs:    1.23e15   (per rank, after TP/PP sharding)
  Backward FLOPs:   2.46e15
  Total FLOPs:      3.69e15
  Compute time:     3.73 ms   (at 989 TFLOPS peak)

Communication per step:
  TP (all-reduce):  2.15 ms   (268 MB × 2 per layer, 16 layers/rank, NVLink)
  FSDP (all-gather): 1.31 ms  (gather params before fwd, IB)
  FSDP (reduce-scatter): 1.31 ms  (scatter grads in bwd, IB)
  PP (P2P):         0.07 ms   (activation tensor between stages, IB)
  PP bubble:        3.73 ms   (bubble fraction = 3/32 = 9.4%)

Projected step time:  12.30 ms
Projected MFU:        48.2%
Bottleneck:           TP communication + PP bubble
```

```python
# Sweep parallelism configs
sweep = model.sweep(
    num_gpus=512,
    tp=[1, 2, 4, 8],
    pp=[1, 2, 4, 8],
    # dp is inferred from num_gpus / (tp * pp)
)

sweep.print_table(sort_by="mfu")
```

```
tp  pp  dp   MFU    bottleneck       fits?
─────────────────────────────────────────────
 4   4  32  52.1%  PP bubble          ✓
 8   2  32  50.8%  TP comm            ✓
 4   2  64  49.3%  TP comm            ✓
 8   4  16  48.2%  TP comm + bubble   ✓
 2   4  64  47.5%  PP bubble          ✓
 8   8   8  41.0%  PP bubble (21.9%)  ✓
 1   1 512  38.6%  FSDP comm          ✓
 1   8  64  36.2%  PP bubble (21.9%)  ✓
 8   1  64    -    OOM (needs 94 GB)  ✗
```

---

## Core API shape

Three main concepts:

**`HardwareConfig`** — describes a GPU and its interconnects. FLOPS, memory, NVLink bandwidth, IB bandwidth. A few presets for common GPUs (H100 SXM, A100 80GB, etc.) but fully user-configurable.

**`CostModel`** — takes a model config (layers, hidden dim, heads, vocab, FFN multiplier) and a hardware config. Knows how to compute FLOPs per layer, parameter counts, activation sizes. Has no opinion about parallelism — that's a separate input.

**`CostReport`** — the output of `model.project(tp=..., dp=..., pp=..., ...)`. Contains the full breakdown: memory per rank, compute time, communication time per dimension, bubble fraction, projected MFU, and whether it fits. Can be printed, compared, or used programmatically.

The `sweep()` method is sugar over `project()` — it loops over a grid of configs, filters out the ones that don't fit, and ranks the rest.

---

## What the formulas are

These are all derived in the [parallelism strategies entry](2026-02-24-parallelism-strategies.md). The cost model just makes them executable:

- **FLOPs per layer:** `24bsh² + 4bs²h` (forward), `3×` for full step
- **Parameter count:** `12h²·L` (standard transformer, approximate)
- **Memory:** params (`P·Φ`) + grads (`P·Φ`) + optimizer (`P·4·Φ_fp32`) + activations (depends on AC, PP depth)
- **TP comm:** `2·Φ·b·s·h / β_nvlink` per layer (reduce-scatter + all-gather for SP, or all-reduce)
- **FSDP comm:** all-gather `P·Φ / (dp·β_ib)` + reduce-scatter same
- **PP bubble:** `(pp-1) / M` for 1F1B, `(pp-1) / (M·v)` for interleaved
- **MFU:** `actual_flops / (peak_flops × step_time)`

---

## What it deliberately ignores

The model is analytical, not a simulator. Things it won't capture:

- **Memory bandwidth bottlenecks.** LayerNorm, softmax, GeLU, residual adds are memory-bound, not compute-bound. They're FLOP-negligible but can eat wall-clock time. The cost model doesn't know about them.
- **Kernel launch overhead.** Dozens of kernel launches per layer, each with ~5-10μs overhead. Adds up but hard to predict analytically.
- **Network congestion.** The formulas assume each rank gets its full share of bandwidth. Real networks have contention, especially for all-to-all (EP).
- **Overlap.** The model assumes communication and compute are serialized (pessimistic). In practice, FSDP prefetches and TP can overlap with compute. The real number is better.
- **Compiler effects.** torch.compile fuses kernels, changes memory access patterns, affects both compute and memory bandwidth. Not modeled.

The gap between the model's projection and reality is typically 10-30%. The interesting work is explaining that gap — which of these factors accounts for the difference, and by how much.

---

## Open questions

- **Should the cost model know about torch.compile?** Probably not initially — keep it formula-based. But eventually, comparing "projected without compile" vs "measured with compile" is a useful data point.
- **Activation checkpointing.** AC trades compute for memory. The cost model needs to know the AC strategy (none, full, selective) to compute both memory and FLOPs correctly. This is the first thing beyond simple formulas.
- **How to integrate with the config system.** The model config already exists in the `TrainConfig` dataclass tree (see [config design entry](../config_system/2026-02-24-config-design.md)). The cost model should read directly from `TransformerConfig` fields (`d_model`, `n_layers`, `n_heads`, `d_ff`), not require re-specifying model dimensions. `CostModel.from_train_config(train_config)` should extract what it needs from the existing config.
- **Validation loop.** After running actual training, compare measured step time / MFU against the projection. This comparison should be automatic — part of the training loop output.
