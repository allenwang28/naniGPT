# Parallelism Strategies for Large-Scale Model Training

*2026-02-24*

This entry covers the major parallelism strategies used in LLM training, then compares how two real frameworks — NVIDIA's Megatron-LM and PyTorch's torchtitan — implement them. The goal is to understand what each strategy actually does at the communication-primitive level, then see how framework design choices lead to very different ways of expressing the same ideas.

Repos studied:
- **torchtitan** @ [`1ce7f76`](https://github.com/pytorch/torchtitan/tree/1ce7f761d76bc408c9e6f32e2bbd2cf9f2ac25ae)
- **Megatron-LM** @ [`32efeffd`](https://github.com/NVIDIA/Megatron-LM/tree/32efeffd2d3cca48f1bfa575e1f2d0f0feac3d0f)

---

## Part 1: The Strategies

**Notation** used throughout the cost models below:

| Symbol | Meaning |
|--------|---------|
| `P` | Total model parameters (count, not bytes) |
| `L` | Number of transformer layers |
| `h` | Hidden dimension |
| `s` | Sequence length |
| `b` | Micro-batch size (per GPU) |
| `N` | Number of ranks for the parallelism being discussed |
| `β` | Interconnect bandwidth (bytes/sec, unidirectional per link) |
| `F` | Peak device FLOPS (e.g. 989 TFLOPS for H100 bf16) |
| `Φ` | Element size in bytes (2 for bf16, 1 for fp8) |

**Per-layer FLOP counting.** The rule for matrix multiplications: multiplying `[m, k] × [k, n]` costs `2mkn` FLOPs (one multiply and one add per output element, for `k` terms). A linear layer `Y = X·W` where `X` is `[b·s, d_in]` and `W` is `[d_in, d_out]` costs `2·b·s·d_in·d_out` FLOPs.

Applying this to a standard transformer layer (FFN intermediate dim = `4h`):

```
  Component            Operation                  Shape                    FLOPs
  ─────────────────────────────────────────────────────────────────────────────────
  Q projection         X·W_Q                     [bs, h] × [h, h]        2bsh²
  K projection         X·W_K                     [bs, h] × [h, h]        2bsh²
  V projection         X·W_V                     [bs, h] × [h, h]        2bsh²
  Attention (QK^T)     Q·K^T                     [bs, s] (per head)      2bs²h
  Attention (·V)       scores·V                  [bs, s] × [s, h/heads]  2bs²h
  Output projection    attn_out·W_O              [bs, h] × [h, h]        2bsh²
  ─────────────────────────────────────────────────────────────────────────────────
  Attention subtotal                                                      8bsh² + 4bs²h
  ─────────────────────────────────────────────────────────────────────────────────
  FFN up projection    X·W_up                    [bs, h] × [h, 4h]       8bsh²
  FFN down projection  X·W_down                  [bs, 4h] × [4h, h]      8bsh²
  ─────────────────────────────────────────────────────────────────────────────────
  FFN subtotal                                                            16bsh²
  ─────────────────────────────────────────────────────────────────────────────────
  TOTAL (forward)                                                         24bsh² + 4bs²h
```

For the attention scores: each of the `n_heads` heads computes `Q_head · K_head^T` where `Q_head` is `[b, s, h/n_heads]` and `K_head` is `[b, s, h/n_heads]`, costing `2 · b · s · s · (h/n_heads)` FLOPs per head. Summed over `n_heads` heads: `2bs²h`. Same for the `scores · V` multiplication.

Backward is approximately 2× forward (one matmul for the input gradient, one for the weight gradient, per layer), so **full step per layer ≈ `3 × (24bsh² + 4bs²h) = 72bsh² + 12bs²h`**.

Note: this counts matmuls only. It ignores softmax, LayerNorm, GeLU, residual adds, etc. — these are all element-wise or reduction ops that are negligible relative to the O(bsh²) GEMMs for typical model sizes. They do matter for roofline analysis (they're memory-bandwidth-bound, not compute-bound), but not for FLOP counting.

**Communication cost counting.** The standard model for collective communication cost uses the "bus bandwidth" model, which measures the total bytes each rank must push through its bottleneck link. This lets you divide by the link's bandwidth `β` to get a time estimate.

**All-reduce** (e.g., synchronizing gradients across `N` ranks). An all-reduce of `M` bytes is implemented as a reduce-scatter followed by an all-gather on a ring:

```
  Ring all-reduce of M bytes across N=4 ranks:

  Step 1: Reduce-scatter                    Step 2: All-gather
  Each rank sends M/N, receives M/N,        Each rank sends M/N, receives M/N,
  repeated (N-1) times around ring.         repeated (N-1) times around ring.

  Rank 0 ──M/4──▶ Rank 1                   Rank 0 ──M/4──▶ Rank 1
    ▲                │                        ▲                │
   M/4             M/4                       M/4             M/4
    │                ▼                        │                ▼
  Rank 3 ◀──M/4── Rank 2                   Rank 3 ◀──M/4── Rank 2

  Per-rank bytes sent: M·(N-1)/N            Per-rank bytes sent: M·(N-1)/N
  ─────────────────────────────────────────────────────────────────────
  Total per-rank bytes: 2M·(N-1)/N ≈ 2M for large N
  Time: 2M/β  (assuming bandwidth β per link)
```

The key insight: the `2M` cost is independent of `N`. Adding more ranks doesn't increase per-rank bandwidth demand (it does add latency from more ring steps, but for large messages the bandwidth term dominates). This is why data parallelism scales well.

**All-gather** (`N` ranks each hold `M/N` bytes; after the op, all ranks hold all `M` bytes):

```
  Each rank sends its M/N shard around the ring in (N-1) steps.
  Per-rank bytes sent: M·(N-1)/N ≈ M
  Time: M/β
```

**Reduce-scatter** (inverse of all-gather — `N` ranks each hold `M` bytes; after the op, each rank holds `M/N` bytes of the reduced result):

```
  Each rank receives and reduces M/N from each other rank in (N-1) steps.
  Per-rank bytes sent: M·(N-1)/N ≈ M
  Time: M/β
```

Note: all-reduce = reduce-scatter + all-gather, so `2M ≈ M + M`. This decomposition matters because FSDP and SP use reduce-scatter and all-gather *separately* rather than as a combined all-reduce.

**All-to-all** (`N` ranks each send a different `M/N`-sized chunk to each other rank):

```
  Each rank sends (N-1) chunks of size M/N to distinct peers.
  Per-rank bytes sent: M·(N-1)/N ≈ M
  Time: M/(N·β)  per link, but requires N-1 simultaneous links
  Effective time: M/β  if network has full bisection bandwidth
                  M·N/β_bisection  if bisection-limited
```

All-to-all is fundamentally different from ring collectives: it's an all-pairs exchange. Each rank talks to every other rank simultaneously, so it needs bisection bandwidth rather than point-to-point bandwidth. On networks with limited bisection bandwidth (e.g., fat-tree with oversubscription), all-to-all degrades much faster than all-reduce.

**P2P send/recv** (one rank sends `M` bytes to one other rank):

```
  Per-rank bytes sent: M
  Time: M/β
```

Simple and cheap, but only moves data between two ranks. Used by pipeline parallelism.

**Why all this counting matters.** The point of these cost models is to answer the guiding question: *given a compute budget, what's the best way to spend it?* Specifically:

1. **Will it fit?** Sum up the per-rank memory for your chosen parallelism config. If `model_state + activations + workspace > GPU_memory`, you need more sharding (FSDP, TP) or recomputation (AC).

2. **Will it be fast?** Compute `compute_time = FLOPs / F` and `comm_time = bytes / β` for each parallelism dimension. If `comm_time > compute_time`, communication is exposed and you're leaving FLOPS on the table. The ratio `compute_time / (compute_time + comm_time)` is your theoretical efficiency ceiling before considering overlap.

3. **Which parallelism goes where?** Different interconnects have different `β`: NVLink ≈ 900 GB/s (H100), InfiniBand ≈ 50-400 GB/s. TP's `8bshΦ` per layer is small but latency-sensitive (on critical path) → needs NVLink. FSDP's `3PΦ` is large but overlappable → can use IB. PP's `b_μshΦ` is tiny → IB is fine. Plugging in your actual `β` values tells you which strategies can tolerate which interconnects.

4. **What's the MFU?** Model FLOPS Utilization = `actual_FLOPs / (time · F)`. The cost models give you the theoretical `actual_FLOPs` (the compute that's useful work) and the overheads (communication, bubble) that eat into `time`. MFU = 1 would mean zero communication, zero bubble, zero overhead. The gap between your measured MFU and the theoretical ceiling from these formulas tells you where to look for optimization.

5. **When to switch strategies.** As you scale, the relative costs shift. DDP's comm is `2PΦ/β` — fixed with model size, independent of batch or sequence length. TP's comm is `8LbshΦ/β` — grows with sequence length. CP's comm is `2LbshΦ/β` but hides behind O(s²) compute. These crossover points tell you when to add each parallelism dimension.

### 1. Data Parallelism (DDP)

**What it partitions:** Data. The model is replicated on every rank. Each rank processes a different mini-batch slice.

**Communication:** All-reduce on gradients after backward. This is the only collective — forward is communication-free.

**Memory vs communication:** Maximum memory usage (full model replica per GPU), minimum communication (one all-reduce per step). The all-reduce volume is `2 * model_size` per step (reduce-scatter + all-gather phases).

**Cost model:**

| Cost | Formula | Notes |
|------|---------|-------|
| Memory per rank | `2P + 2P + 12P = 16P` bytes | bf16 params + bf16 grads + Adam (fp32 master + momentum + variance) |
| Comm volume/step | `2PΦ` bytes | Ring all-reduce of all gradients |
| Comm time | `2PΦ / β` | Single bottleneck link; overlaps with backward |
| Compute/step | `L(72bsh² + 12bs²h)` FLOPS | Same as single-GPU (no parallelism tax) |
| Compute time | `L(72bsh² + 12bs²h) / F` | |

The ratio `compute_time / comm_time` tells you whether DDP is compute-bound (good) or communication-bound. Plugging in: for a 7B model (`P ≈ 7e9`) in bf16 over 400 GB/s NVLink, comm time ≈ `2 · 7e9 · 2 / 400e9 ≈ 70ms`. If your forward+backward takes > 70ms, you're compute-bound and DDP scales well.

```
  Rank 0          Rank 1          Rank 2          Rank 3
┌──────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐
│ Model    │    │ Model   │    │ Model   │    │ Model   │
│ (full)   │    │ (full)  │    │ (full)  │    │ (full)  │
├──────────┤    ├─────────┤    ├─────────┤    ├─────────┤
│ Batch 0  │    │ Batch 1 │    │ Batch 2 │    │ Batch 3 │
└────┬─────┘    └────┬────┘    └────┬────┘    └────┬────┘
     │               │              │              │
     └───────────────┴──────────────┴──────────────┘
                    all-reduce gradients
```

**Seminal paper:** Li et al., "PyTorch Distributed: Experiences on Accelerating Data Parallel Training," 2020. [arxiv:2006.15704](https://arxiv.org/abs/2006.15704)

---

### 2. FSDP / ZeRO

**What it partitions:** Parameters, gradients, and optimizer states across data-parallel ranks. It's conceptually the same thing as DDP — every rank processes a different data shard and gradients are synchronized — but instead of every rank holding a full copy of the model, the model state is *sharded* across ranks and materialized on-the-fly.

DDP's memory cost is `params + grads + opt_state` per rank, all fully replicated. ZeRO eliminates this redundancy in three stages:
- **Stage 1:** Partition optimizer states (momentum, variance) only
- **Stage 2:** + partition gradients
- **Stage 3:** + partition parameters

FSDP implements Stage 3 — every rank holds only `1/N`-th of the model parameters at rest. Before each layer's forward pass, parameters are all-gathered so every rank has the full layer transiently, then discarded after use. It's DDP with the model "paged in" layer by layer.

**Communication:**
- **Forward:** All-gather parameters before each layer's computation. Each rank contributes its shard, all ranks get the full layer.
- **Backward:** All-gather parameters again (for layers that resharded after forward), then reduce-scatter gradients. Each rank ends up with reduced gradients for only its parameter shard.

The reduce-scatter in backward is doing the same job as DDP's all-reduce (synchronizing gradients), just in sharded form — each rank only needs the gradient slice corresponding to the parameters it owns.

**Memory vs communication:** Trades communication for memory. Each rank stores `params/N + grads/N + opt_state/N`, but pays for all-gather in both forward and backward. Communication volume is `3 * model_size` per step (forward all-gather + backward all-gather + reduce-scatter), compared to `2 * model_size` for DDP. The extra cost over DDP is the forward all-gather — the price of not keeping a full param replica in memory.

**Cost model:**

| Cost | Formula | Notes |
|------|---------|-------|
| Memory per rank (model state) | `16P / N` bytes | Everything sharded N-ways |
| Memory per rank (activations) | `≈ 2Lbsh · Φ` | Activations are NOT sharded (same as DDP) |
| Comm volume/step | `3PΦ` bytes | fwd all-gather + bwd all-gather + reduce-scatter |
| Comm time (no overlap) | `3PΦ / β` | 1.5× DDP's communication |
| Comm time (with overlap) | `≈ PΦ / β` | fwd all-gather overlaps compute; only residual is exposed |

Comparing DDP vs FSDP for a 70B model in bf16 on 64 GPUs: DDP needs `16 · 70e9 ≈ 1.12 TB` per rank (doesn't fit in 80GB). FSDP needs `1.12 TB / 64 ≈ 17.5 GB` per rank for model state, plus activations. The memory win is decisive; the 1.5× communication increase is the price.

```
  Forward pass:
  ┌──────────┐    ┌──────────┐    ┌──────────┐
  │ Shard 0  │    │ Shard 1  │    │ Shard 2  │    (each rank holds 1/3)
  └────┬─────┘    └────┬─────┘    └────┬─────┘
       │               │               │
       └───────────────┴───────────────┘
              all-gather (per layer)
       ┌───────────────────────────────┐
       │       Full layer params       │    (transient, on all ranks)
       └───────────────┬───────────────┘
                    compute
       ┌───────────────┴───────────────┐
       │        Full gradients         │    (transient)
       └───────────────┬───────────────┘
              reduce-scatter
  ┌──────────┐    ┌──────────┐    ┌──────────┐
  │ Grad 0   │    │ Grad 1   │    │ Grad 2   │    (each rank holds 1/3)
  └──────────┘    └──────────┘    └──────────┘
```

The key optimization is **overlapping** the next layer's all-gather with the current layer's compute, and overlapping reduce-scatter with the next layer's backward.

**Seminal papers:**
- Rajbhandari et al., "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models," 2019. [arxiv:1910.02054](https://arxiv.org/abs/1910.02054)
- Zhao et al., "PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel," 2023. [arxiv:2304.11277](https://arxiv.org/abs/2304.11277)

---

### 3. Tensor Parallelism (TP)

**What it partitions:** Weight matrices within each layer. The seminal pattern from Megatron-LM splits each linear layer's weight along one of its two dimensions. TP never splits the batch or sequence dimensions — those stay intact on every rank. Only the weight matrix `W` is divided.

**Column-parallel vs row-parallel: what "column" and "row" mean.**

A linear layer computes `Y = X · W` where `X` is `[b*s, h_in]` (inputs) and `W` is `[h_in, h_out]` (weights). "Column" and "row" refer to how `W` is split:

```
  W = [h_in, h_out]

  Column-parallel: split W along h_out (the columns)
  ┌─────────┬─────────┐
  │         │         │
  │   W_0   │   W_1   │     W_0 = [h_in, h_out/2]
  │         │         │     W_1 = [h_in, h_out/2]
  │         │         │
  └─────────┴─────────┘
   h_in rows, h_out cols

  Row-parallel: split W along h_in (the rows)
  ┌───────────────────┐
  │       W_0         │     W_0 = [h_in/2, h_out]
  ├───────────────────┤
  │       W_1         │     W_1 = [h_in/2, h_out]
  └───────────────────┘
```

**Column-parallel** — each rank holds all input features but a slice of output features:

```
  Rank 0: Y_0 = X · W_0  ->  [b*s, h_out/2]   (a slice of the output)
  Rank 1: Y_1 = X · W_1  ->  [b*s, h_out/2]   (a different slice)
```

No communication needed in forward — `X` is the same on every rank, and each rank independently multiplies by its column slice. The outputs `Y_0` and `Y_1` are different columns of the full result `Y`.

**This is why GeLU can be applied locally after column-parallel.** GeLU is element-wise: `GeLU(Y_0)` gives exactly the same result as taking `GeLU(Y_full)` and slicing out the first half of columns. The non-linearity doesn't mix across the output dimension. Each rank's slice is self-contained.

**Row-parallel** — each rank holds all output features but a slice of input features:

```
  Input X must also be split: X_0 = [b*s, h_in/2], X_1 = [b*s, h_in/2]

  Rank 0: Z_0 = X_0 · W_0  ->  [b*s, h_out]   (partial sum!)
  Rank 1: Z_1 = X_1 · W_1  ->  [b*s, h_out]   (partial sum!)

  Z_full = Z_0 + Z_1   <- requires all-reduce
```

Each rank computes a partial dot product — same output shape but only covering half the inner dimension. You need an all-reduce to sum the partials. You could NOT apply GeLU before this all-reduce, because `GeLU(Z_0) + GeLU(Z_1) ≠ GeLU(Z_0 + Z_1)`. Non-linearities don't distribute over addition.

**The Megatron paired pattern for MLPs:**

For a two-layer MLP `Y = dropout(GeLU(X·A)·B)`, split `A` column-wise and `B` row-wise:

```
         Rank 0                       Rank 1
  ┌─────────────────────┐    ┌─────────────────────┐
  │  X (replicated)     │    │  X (replicated)     │
  │         │           │    │         │           │
  │         ▼           │    │         ▼           │
  │  A_0 (cols 0..d/2)  │    │  A_1 (cols d/2..d)  │
  │  Y_0 = GeLU(X·A_0)  │    │  Y_1 = GeLU(X·A_1)  │
  │         │           │    │         │           │
  │         ▼           │    │         ▼           │
  │  B_0 (rows 0..d/2)  │    │  B_1 (rows d/2..d)  │
  │  Z_0 = Y_0·B_0      │    │  Z_1 = Y_1·B_1      │
  └────────┬────────────┘    └────────┬────────────┘
           │                          │
           └──────────┬───────────────┘
                 all-reduce
              Z = Z_0 + Z_1
```

The pairing is not arbitrary — column-parallel `A` produces outputs split on `h_out`, and row-parallel `B` expects inputs split on `h_in`. These are the **same split**, because `A`'s output dimension is `B`'s input dimension (the FFN intermediate size). So `Y_0` feeds directly into `B_0` with no communication between `A` and `B`. The only all-reduce is at the end of `B`, giving one all-reduce for the entire MLP forward.

For the self-attention block, the same trick applies: Q, K, V projections are column-parallel (splitting on `h_out` = splitting across attention heads, since each head is an independent slice of the output dimension), and the output projection `W_O` is row-parallel (its `h_in` is the concatenated head outputs, already split across ranks).

**Communication primitives:** Two conjugate `autograd.Function` pairs:
- **`f`**: identity in forward, all-reduce in backward
- **`g`**: all-reduce in forward, identity in backward

Column-parallel uses `f` before the GEMM (input is copied forward; in backward, the gradient w.r.t. the input needs contributions from all ranks, so all-reduce). Row-parallel uses `g` after the GEMM (partial sums are all-reduced forward; in backward, each rank only needs the gradient for its own input slice, so identity). The pair ensures exactly one all-reduce in each direction per transformer block sub-layer.

**Memory vs communication:** Each rank stores `params/tp` for TP-sharded layers, plus full activations (unless combined with SP). TP is bandwidth-bound and works best over NVLink.

**Cost model:**

Each transformer layer has 2 all-reduces in forward (one for attention sublayer, one for FFN) and 2 in backward (the conjugate ops). Each all-reduce moves a tensor of shape `[b, s, h]`.

| Cost | Formula | Notes |
|------|---------|-------|
| Params per rank | `P / tp` | Sharded layers only; embeddings may be replicated |
| Activation memory per rank | `2bshΦ · L` | Full activations replicated (without SP) |
| Comm per layer (fwd) | `2 × 2bshΦ = 4bshΦ` bytes | 2 all-reduces of `[b,s,h]` tensors |
| Comm per layer (fwd+bwd) | `4 × 2bshΦ = 8bshΦ` bytes | Backward has the conjugate all-reduces |
| Total comm/step | `8LbshΦ` bytes | Over all layers |
| Compute per rank/step | `L(72bsh² + 12bs²h) / tp` | Work divided across tp ranks |

The comm/compute ratio scales as `8LbshΦ / (L · 72bsh²/tp) = 8Φ·tp / (72h)`. For `h=4096`, `tp=8`, bf16: `8·2·8 / (72·4096) ≈ 0.00043` — TP communication is tiny relative to compute *per all-reduce*, but it's latency-sensitive because it's on the critical path (no overlap without async TP). This is why TP must sit on NVLink (high bandwidth, low latency) and why `tp=8` is the practical max (one NVLink domain).

**Seminal paper:** Shoeybi et al., "Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism," 2019. [arxiv:1909.08053](https://arxiv.org/abs/1909.08053)

---

### 4. Sequence Parallelism (SP)

**What it partitions:** Activations along the sequence dimension, specifically in the regions *between* TP columns — LayerNorm and dropout.

**Why it exists:** With TP alone, LayerNorm and dropout still operate on full (replicated) activations. This wastes memory. SP shards activations along the sequence dimension in these regions.

**Communication:** Replaces the all-reduce in TP with a **reduce-scatter** (going into SP regions) and an **all-gather** (leaving SP regions). The total communication volume is the same, but activation memory is reduced by `1/tp` in SP regions.

**Cost model:** SP doesn't change communication volume — an all-reduce is mathematically equivalent to a reduce-scatter followed by an all-gather, which is exactly what SP does (reduce-scatter into SP region, all-gather out of SP region). The win is purely in activation memory:

| Cost | Without SP | With SP |
|------|-----------|---------|
| Activation memory per layer | `2bshΦ` | `2bshΦ / tp` (in SP regions) |
| Comm volume per layer (fwd) | `4bshΦ` | `4bshΦ` (same) |

SP reduces activation memory by `tp×` in LayerNorm/dropout regions at zero additional communication cost. This is a free lunch for memory.

```
  TP region              SP region              TP region
  (replicated)           (seq-sharded)          (replicated)
  ┌──────────┐          ┌──────────┐          ┌──────────┐
  │ Attn out │─reduce──▶│ LayerNorm│──gather─▶│ FFN in   │
  │ (full)   │ scatter  │ (1/tp)   │          │ (full)   │
  └──────────┘          └──────────┘          └──────────┘
```

**Seminal paper:** Korthikanti et al., "Reducing Activation Recomputation in Large Transformer Models," 2022. [arxiv:2205.05198](https://arxiv.org/abs/2205.05198)

---

### 5. Context Parallelism (CP)

**What it partitions:** The input sequence across ranks, specifically for attention computation. Each rank holds `seq_len / cp_size` tokens.

**Why it exists:** For very long sequences, the attention computation's O(n²) memory and compute becomes the bottleneck, not model size. CP distributes this cost.

**How it works — ring attention in detail.**

Each rank holds a chunk of the sequence. The key mechanic: **Q (queries) never moves. Only KV (keys and values) rotates around the ring.** Each rank's job is to compute the correct attention output for its local queries attending to *all* keys across the full sequence — it just does this incrementally, one KV block at a time.

Setup with `cp=4`, sequence length `S`:

```
  Rank 0 has:  Q_0, KV_0   (tokens 0..S/4)
  Rank 1 has:  Q_1, KV_1   (tokens S/4..S/2)
  Rank 2 has:  Q_2, KV_2   (tokens S/2..3S/4)
  Rank 3 has:  Q_3, KV_3   (tokens 3S/4..S)
```

From rank 0's perspective:

```
  Step 0: compute attn(Q_0, KV_0)   — local chunk
          send KV_0 → rank 1, recv KV_3 ← rank 3    (overlapped with compute)

  Step 1: compute attn(Q_0, KV_3)   — accumulate with step 0's result
          send KV_3 → rank 1, recv KV_2 ← rank 3

  Step 2: compute attn(Q_0, KV_2)   — accumulate
          send KV_2 → rank 1, recv KV_1 ← rank 3

  Step 3: compute attn(Q_0, KV_1)   — accumulate, done

  Result: rank 0 has the correct attention output for Q_0
          attending to ALL of KV_0, KV_1, KV_2, KV_3
```

All four ranks do this simultaneously — each rank's Q stays fixed while KV blocks rotate through. The full sequence is **never concatenated** on any single rank.

**How partial attention results are accumulated (online softmax).**

You can't just average partial attention outputs, because softmax is computed over the *full* key sequence. `softmax(Q·K_0^T)·V_0 + softmax(Q·K_1^T)·V_1` is wrong — the softmax denominators differ.

The online softmax trick (from FlashAttention) maintains a running output `O` and running log-sum-exp `L` that are "correct so far":

```
  After processing KV block i:

  1. Compute local scores:    S_i = Q_0 · K_i^T
  2. Find new running max:    m_new = max(m_old, rowmax(S_i))
  3. Correction factor:       c = exp(m_old - m_new)
  4. Update output:           O = c · O_old + exp(S_i - m_new) · V_i
  5. Update normalizer:       L = c · L_old + rowsum(exp(S_i - m_new))

  After all blocks:           O_final = O / L
```

The correction factor `c` rescales the previous accumulated result to account for the global max changing. This is mathematically exact — after all KV blocks in any order, `O_final` is identical to computing full attention in one shot.

**What happens after attention.** Rank 0's attention output is the correct result for tokens `0..S/4`. It feeds into FFN, which is embarrassingly parallel (no cross-token interaction — just independent per-token transforms). Then into the next transformer layer, which does its own fresh round of KV ring rotation. The ring rotation is per-layer, not accumulated across layers:

```
  Layer 1:  ring-rotate KV, compute attention  →  FFN (local, no comm)
  Layer 2:  ring-rotate KV, compute attention  →  FFN (local, no comm)
  ...
  Layer L:  ring-rotate KV, compute attention  →  FFN (local, no comm)

  Each rank always holds only its s/cp tokens. Nothing is ever concatenated.
  KV blocks are transient — rotated during attention, discarded after.
```

**Cost model:**

Each rank holds `s/cp` query positions and must attend to all `s` key positions over `cp` ring steps. In each ring step, a KV block of shape `[b, s/cp, 2h]` (K and V concatenated) is sent/received.

| Cost | Formula | Notes |
|------|---------|-------|
| Activation memory (attn) | `≈ 2b(s/cp)²` | QK^T attention matrix is `(s/cp) × s` but computed blockwise; peak ≈ `(s/cp)²` |
| Comm per attn layer | `2bsh · (cp-1)/cp · Φ` bytes | `(cp-1)` ring steps, each sending KV of size `2b(s/cp)hΦ` |
| Compute per rank (attn) | `4bs²h / cp` FLOPS | `(s/cp)` queries × `s` keys × `h` dims × 2 (QK^T + attn·V) |

The comm/compute ratio for attention: `2bshΦ / (4bs²h/cp) = cpΦ/(2s)`. For `s=128k`, `cp=8`, bf16: `8·2/(2·131072) ≈ 0.00006`. Communication is negligible compared to the O(s²) attention compute, which is exactly why ring attention works — the KV transfer hides behind the quadratic compute. CP becomes less efficient as `s` gets shorter, because there's less compute to hide the communication behind.

**Backward pass.** The same ring rotation happens again, but now each rank has `dO` (gradient of the attention output for its local queries) and needs `dQ`, `dK`, `dV`. KV blocks rotate through again; at each step the rank computes:
1. `dQ` contribution from this KV block — accumulated locally (same online correction as forward)
2. `dK`, `dV` for this KV block — partial gradients that need to end up at the rank that *owns* those keys/values

The `dK`/`dV` partials are accumulated as they rotate in the opposite direction around the ring. Each rank adds its contribution and passes it along. After `cp-1` steps, the accumulated `dK`/`dV` arrives at the owning rank with all contributions summed. Total backward communication is roughly the same as forward.

**Does CP increase step latency?** No — CP *decreases* per-rank compute. Each rank does `4bs²h/cp` attention FLOPs and `16bsh²/cp` FFN FLOPs (since it processes `s/cp` tokens). The whole point is distributing work across more GPUs. CP's cost is not increased latency — it's the communication overhead (KV rotation, weight gradient sync across CP ranks via FSDP) and the opportunity cost (those `cp` GPUs could have been `dp` GPUs processing more data instead of splitting one sequence).

**Load balancing concern:** With causal attention, early sequence positions attend to fewer tokens than late positions. Naive partitioning creates load imbalance. Solutions include interleaving (head-tail pairing) or mask-aware partitioning.

**Seminal papers:**
- Liu et al., "Ring Attention with Blockwise Transformers for Near-Infinite Context," 2023. [arxiv:2310.01889](https://arxiv.org/abs/2310.01889)
- Yang et al., "Context Parallelism for Scalable Million-Token Inference," 2024. [arxiv:2411.01783](https://arxiv.org/abs/2411.01783)
- Jacobs et al., "DeepSpeed Ulysses: System Optimizations for Enabling Training of Extreme Long Sequence Transformer Models," 2023. [arxiv:2309.14509](https://arxiv.org/abs/2309.14509) (alternative all-to-all approach)

---

### 6. Pipeline Parallelism (PP)

**What it partitions:** Layers (depth). The model is split into stages; each stage lives on a different rank. A given rank only holds `layers / pp_size` layers.

**Communication:** Point-to-point send/recv of activations (forward) and gradients (backward) between adjacent pipeline stages.

**The bubble problem:** With a single microbatch, stages execute sequentially — stage 1 runs forward, sends to stage 2, stage 2 runs forward, sends to stage 3, etc. Most ranks sit idle most of the time. The solution is **microbatching**: split the mini-batch into `M` microbatches and pipeline them through the stages. But how you schedule the forward and backward passes of those microbatches matters a lot for memory.

**GPipe (fill-drain):** All `M` forwards first, then all `M` backwards. Simple, but rank 0 does `F0, F1, ..., F_{M-1}` before any backward runs, so it must store activations for **all M microbatches** simultaneously. Memory scales as O(M).

```
  GPipe with pp=4, M=6:

  Rank 0: F0 F1 F2 F3 F4 F5 __________ B5 B4 B3 B2 B1 B0
  Rank 1:    F0 F1 F2 F3 F4 F5 ______ B5 B4 B3 B2 B1 B0
  Rank 2:       F0 F1 F2 F3 F4 F5 __ B5 B4 B3 B2 B1 B0
  Rank 3:          F0 F1 F2 F3 F4 F5 B5 B4 B3 B2 B1 B0
                                      ↑
                         backward starts only after ALL forwards finish

  Rank 0 stores activations for:  F0, F1, F2, F3, F4, F5  (all 6!)
  Peak activation memory: O(M) per rank
```

**1F1B (One Forward One Backward):** After a warmup phase that fills the pipeline, each rank alternates one forward with one backward. The key insight: **each backward frees one microbatch's activations before the next forward creates new ones**. In steady state, the number of in-flight microbatches stays constant at `pp` — not growing with `M`.

The standard way to draw pipeline schedules uses **staggered rows** — each rank's timeline is indented by one slot to represent the pipeline propagation delay. This is compact but can be misleading because operations in the same column happen at *different global times*:

```
  1F1B with pp=4, M=8 (staggered view — each row offset by 1 time unit):

  Rank 0: F0 F1 F2 F3 _  _  _  _  _  _  B0 F4 B1 F5 B2 F6 B3 F7 B4 B5 B6 B7
  Rank 1:    F0 F1 F2 F3 _  _  _  _  B0 F4 B1 F5 B2 F6 B3 F7 B4 B5 B6 B7
  Rank 2:       F0 F1 F2 F3  _  _ B0 F4 B1 F5 B2 F6 B3 F7 B4 B5 B6 B7
  Rank 3:          F0 F1 F2 F3 B0 F4 B1 F5 B2 F6 B3 F7 B4 B5 B6 B7
                               ↑
                warmup    steady state (1F1B)              drain
```

**Important:** B0 appears in the same column across ranks, but this does NOT mean they happen simultaneously. The row offset means rank 3's B0 happens first (it has the loss), then rank 2's B0 one time slot later (after receiving gradient from rank 3), then rank 1, then rank 0. The gradient **cascades** from the last stage back to the first via P2P send/recv:

```
  Gradient cascade for microbatch 0 (aligned to global time):

  t=4:  Rank 3 finishes F0, computes loss
  t=5:  Rank 3 computes B0, sends grad ──▶ Rank 2
  t=6:  Rank 2 computes B0, sends grad ──▶ Rank 1
  t=7:  Rank 1 computes B0, sends grad ──▶ Rank 0
  t=8:  Rank 0 computes B0

  Rank 0 was idle at t=5, t=6, t=7 — waiting pp-1 = 3 slots for the
  gradient to propagate back. It filled some of the round trip with
  warmup forwards (F1, F2, F3) but still has pp-1 unavoidable idle slots.
```

The bubble fraction is the same as GPipe: `(pp-1) / M`. The win is **memory**: rank 0's peak in-flight activations go from M (all microbatches) down to pp (only the warmup's worth). For `pp=4, M=64`: GPipe stores 64 microbatches of activations, 1F1B stores 4.

**Interleaved 1F1B:** The insight: if each rank holds `v` non-contiguous "virtual stages" instead of 1, microbatches cycle through the pipeline `v` times faster — shrinking the bubble by a factor of `v`. With `pp=4` physical ranks and `v=2` chunks, you get 8 virtual stages. The stage-to-rank mapping is **looped** (round-robin): rank 0 gets virtual stages 0 and 4, rank 1 gets 1 and 5, etc.

```
  Stage layout with pp=4, v=2 (8 virtual stages, 32 layers):

  Rank 0:  [layers 0-3]  ···(other ranks)···  [layers 16-19]
  Rank 1:  [layers 4-7]  ···(other ranks)···  [layers 20-23]
  Rank 2:  [layers 8-11] ···(other ranks)···  [layers 24-27]
  Rank 3:  [layers 12-15] ··(other ranks)···  [layers 28-31]
           ╰─ chunk 0 ─╯                      ╰─ chunk 1 ─╯
```

A microbatch traverses: rank 0 (chunk 0) → rank 1 → rank 2 → rank 3 → rank 0 (chunk 1) → rank 1 → rank 2 → rank 3. It passes through each rank **twice**, so it completes a full pipeline round-trip in half the wall-clock time of plain 1F1B. The pipeline "fills up" faster, leaving less idle time.

```
  Interleaved 1F1B with pp=4, v=2, M=8:
  (subscripts = microbatch, superscripts = chunk; each F/B is a virtual stage)

  Rank 0: F⁰₀ F⁰₁ F⁰₂ F⁰₃   F¹₀ F¹₁  B⁰₀ F⁰₄ B¹₀ F¹₂  B⁰₁ F⁰₅ B¹₁ F¹₃ ...
  Rank 1:    F⁰₀ F⁰₁ F⁰₂ F⁰₃   F¹₀  B⁰₀ F⁰₄ B¹₀ F¹₁  B⁰₁ F⁰₅ B¹₁ F¹₂ ...
  Rank 2:       F⁰₀ F⁰₁ F⁰₂ F⁰₃   B⁰₀ F⁰₄ B¹₀ F¹₀  B⁰₁ F⁰₅ B¹₁ F¹₁ ...
  Rank 3:          F⁰₀ F⁰₁ F⁰₂ F⁰₃ B⁰₀ F⁰₄ B¹₀ F¹₀  B⁰₁ F⁰₅ B¹₁ F¹₁ ...

  Warmup fills faster: only pp/v = 2 microbatches before chunk 1 starts flowing.
  Bubble fraction: (pp-1) / (M·v)  =  3 / 16 ≈ 19%  (vs 3/8 = 37% without interleaving)
```

The tradeoff: each microbatch now crosses `2 × (pp-1)` stage boundaries instead of `pp-1`, so **P2P communication doubles**. Each crossing also goes over the inter-node network (the microbatch leaves its rank, goes through all other ranks, and comes back). For NVLink-connected GPUs within a node this is cheap; across nodes it adds up. Still, since PP communication volume is small (one activation tensor per crossing), the doubled comm is usually worth the halved bubble.

**Zero Bubble (ZB1P):** The bubble exists because earlier ranks sit idle waiting for gradients to cascade back from the last rank. What if we could make that cascade faster?

The key observation: a normal backward pass does two independent matmuls per linear layer. Consider a single linear layer `Y = X · W`. The forward pass is just `Y = X · W`. In the backward pass, we receive `dL/dY` from downstream and need to produce two things:

```
  Full backward for Y = X · W:

    B_input  (I):  dL/dX = dL/dY · W^T
    B_weight (W):  dL/dW = X^T · dL/dY

  Both consume the same input (dL/dY), but they serve completely different consumers:

    I produces dL/dX  →  sent to the UPSTREAM rank so it can run its backward pass.
                         This is on the critical path: the upstream rank is blocked
                         waiting for this value.

    W produces dL/dW  →  stays LOCAL. Only the optimizer needs it, and only at the
                         very end of the step when we call optimizer.step().
                         Nothing else depends on this. It can wait.
```

In standard 1F1B, we compute both together as one atomic "B" operation. But there's no mathematical reason they have to be coupled — I only needs `dL/dY` and `W` (both already available), W only needs `dL/dY` and `X` (the saved activation). They're two independent matmuls that happen to share the same input `dL/dY`.

**Why this shrinks the bubble:** In the gradient cascade, each rank does its backward and then sends dL/dX to the upstream rank. In standard 1F1B, each rank computes the full B (= I + W) before sending, so the cascade takes `(pp-1) × T_b` time. In ZB1P, each rank computes *only I* before sending, so the cascade takes `(pp-1) × T_i` time — roughly **half** as long, since `T_i ≈ T_b/2`.

Here's the effect on rank 0's timeline (pp=4, M=8). In the diagrams below, time flows left to right. Each cell is one time unit. `T_f = T_i = T_w = 1`, so standard `T_b = T_i + T_w = 2`.

```
  Standard 1F1B (B = 2 time units, shown as [IW]):
  Rank 0 waits (pp-1) × T_b = 3 × 2 = 6 time units for the first gradient.

  Rank 0: F0 F1 F2 F3 __ __ __ __ __ __ [IW]₀ F4 [IW]₁ F5 [IW]₂ F6 [IW]₃ F7 [IW]₄ [IW]₅ [IW]₆ [IW]₇
                       ╰── 6 idle slots ─╯
                           (the bubble)
```

```
  ZB1P (I and W separated):
  Rank 0 waits (pp-1) × T_i = 3 × 1 = 3 time units for the first gradient.
  W is deferred and fills the time freed up by the shorter wait.

  Rank 0: F0 F1 F2 F3 __ __ __ I0 F4 I1 F5 I2 F6 I3 F7 I4 I5 I6 I7 W0 W1 W2 W3 W4 W5 W6 W7
                       ╰ 3 idle╯
                       (half the bubble!)
```

The bubble shrank from 6 to 3 time units. The total work is the same (each microbatch still needs F + I + W = 3 time units), but by splitting I out of B, the critical-path cascade runs at half the latency, so rank 0 starts receiving gradients sooner.

In the ideal case (`T_f = T_i = T_w`), the deferred W computations slot neatly into the remaining gaps, and the bubble fraction drops to approximately `(pp-1) / (3M)` — a 3× reduction over standard 1F1B's `(pp-1) / M`.

The catch: deferring B_weight means the weight gradient from microbatch `k` isn't applied until later, so the weight used in microbatch `k+1`'s forward is slightly stale. This creates a minor inconsistency (similar to async SGD) that doesn't affect convergence in practice.

**ZBV (Zero Bubble V-shaped):** Takes the B_input/B_weight split further with a **V-shaped stage layout** (as opposed to interleaved 1F1B's looped layout). With `pp` ranks and exactly 2 stages per rank, the mapping is mirrored:

```
  V-shaped layout with pp=4 (8 virtual stages):

  Rank 0:  stage 0, stage 7     ╲    ╱   (outer V)
  Rank 1:  stage 1, stage 6      ╲  ╱
  Rank 2:  stage 2, stage 5      ╱  ╲
  Rank 3:  stage 3, stage 4     ╱    ╲   (inner V)

  Forward:  0 → 1 → 2 → 3 → 4 → 5 → 6 → 7
  Rank:     0   1   2   3   3   2   1   0
```

The V layout means the same rank hosts both an "early" and a "late" stage. When rank 0's stage 0 is idle (waiting for gradients to cascade back), its stage 7 may be busy — and vice versa. Combined with the B_input/B_weight split, this lets you interleave work across the two local stages to fill almost all bubble slots. Theoretically achieves **zero bubble** when F time ≈ B_input time ≈ B_weight time.

**DualPipeV (DeepSeek):** Takes the V-shaped layout and adds **overlapped forward+backward execution**. On each rank, the two local stages can run concurrently: while one stage executes forward, the other executes backward, hiding the compute of one behind the other. This requires careful CUDA stream management — forward and backward run on different streams with explicit synchronization points for P2P communication.

DeepSeek specifically designed this for models with MoE layers, where the all-to-all EP communication during expert dispatch/combine can be overlapped with the concurrent forward/backward compute of the other stage.

```
  DualPipeV on Rank 0 (has stage 0 and stage 7):

  Stream A:  F₀(stage 0)                  B₀(stage 7)
  Stream B:                B₀(stage 0)                  F₁(stage 0)
                  ↕ overlap ↕                   ↕ overlap ↕

  While stream A does F(stage 0), stream B does B(stage 7)
  The EP all-to-all for one stage's MoE layer runs while
  the other stage's attention/FFN compute is on the other stream.
```

**Summary of pipeline schedules:**

| Schedule | Bubble fraction | Stages/rank | Stage layout | Key idea |
|----------|----------------|-------------|-------------|----------|
| GPipe | `(pp-1) / M` | 1 | sequential | All F then all B |
| 1F1B | `(pp-1) / M` | 1 | sequential | Interleave F/B in steady state |
| Interleaved 1F1B | `(pp-1) / (M·v)` | `v` | looped (round-robin) | Multiple virtual stages per rank |
| ZB1P | `≈ 0` (in theory) | `v` | looped | Split B into B_I + B_W, defer B_W |
| ZBV | `≈ 0` (in theory) | 2 | V-shaped (mirror) | V layout + B split fills all gaps |
| DualPipeV | `≈ 0` (in theory) | 2 | V-shaped (mirror) | Concurrent F+B on two CUDA streams |

1F1B's real win over GPipe is **memory**, not bubble: both have the same bubble fraction `(pp-1)/M`. Interleaved 1F1B is the first to actually reduce the bubble (by factor `v`), at the cost of `v×` P2P communication. Zero Bubble schedules eliminate the bubble almost entirely by breaking backward into independently schedulable pieces. DualPipeV is the current state-of-the-art for MoE models, co-designed with expert parallelism overlap.

Both torchtitan and Megatron-LM implement 1F1B and Interleaved 1F1B. Torchtitan (via PyTorch's `torch.distributed.pipelining`) additionally implements GPipe, ZB1P, ZBV, DualPipeV, and Looped BFS — expressed through a **schedule IR** (list of actions that gets lowered through passes to add communication, FSDP unshard/reshard, etc.). Megatron-LM has fewer named schedules but goes deeper on **MoE overlap** with fine-grained layer-level scheduling on dual CUDA streams.

**Memory vs communication:** Minimal communication (only activations between stages), significant memory savings (each rank holds `1/pp` of the model). The cost is the pipeline bubble (idle time) and increased latency.

**Cost model:**

Let `M` = number of microbatches, `b_μ` = micro-batch size, `v` = virtual stages per rank.

| Cost | Formula | Notes |
|------|---------|-------|
| Params per rank | `P / pp` | Layers divided across stages |
| Comm per microbatch (1F1B) | `(pp-1) · b_μshΦ` bytes | P2P send/recv at each stage boundary |
| Comm per microbatch (interleaved) | `v · (pp-1) · b_μshΦ` bytes | `v×` more stage boundaries |
| Bubble fraction (1F1B) | `(pp-1) / M` | Same as GPipe |
| Bubble fraction (interleaved) | `(pp-1) / (M·v)` | Reduced by factor `v` |
| Bubble fraction (ZB/DualPipe) | `≈ 0` | When F ≈ B_I ≈ B_W time |
| Peak activations (1F1B) | `pp · b_μshΦ` | At most `pp` microbatches in flight |
| Peak activations (GPipe) | `M · b_μshΦ` | All microbatches stored |

The bubble is the real cost. For `pp=8` and `M=32`: 1F1B bubble = `7/32 ≈ 22%`. With interleaved 1F1B and `v=2` chunks: `7/64 ≈ 11%`, better but at the cost of 2× P2P communication. With Zero Bubble schedules: theoretically 0%, but requires that `F ≈ B_I ≈ B_W` in wall-clock time, which isn't always achievable.

PP's communication volume is tiny compared to TP or FSDP — a single activation tensor vs. all model parameters — which is why PP can go over InfiniBand while TP needs NVLink.

**Seminal papers:**
- Huang et al., "GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism," 2018. [arxiv:1811.06965](https://arxiv.org/abs/1811.06965)
- Harlap et al., "PipeDream: Fast and Efficient Pipeline Parallel DNN Training," 2018. [arxiv:1806.03377](https://arxiv.org/abs/1806.03377)
- Narayanan et al., "Memory-Efficient Pipeline-Parallel DNN Training," 2020. [arxiv:2006.09503](https://arxiv.org/abs/2006.09503) (PipeDream-Flush, basis for modern synchronous 1F1B)
- Narayanan et al., "Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM," 2021. [arxiv:2104.04473](https://arxiv.org/abs/2104.04473) (Interleaved 1F1B)
- Qi et al., "Zero Bubble Pipeline Parallelism," 2024. [arxiv:2401.10241](https://arxiv.org/abs/2401.10241) (ZB1P and ZBV)
- DeepSeek-AI, "DeepSeek-V3 Technical Report," 2024. [arxiv:2412.19437](https://arxiv.org/abs/2412.19437) (DualPipeV)

---

### 7. Expert Parallelism (EP)

**What it partitions:** MoE experts across ranks. With `E` experts and `ep_size` ranks, each rank holds `E / ep_size` experts.

**Communication:** All-to-all dispatch and combine. The router selects which expert each token goes to, then:
1. **Dispatch (all-to-all):** Tokens are sent from their home rank to the rank hosting their assigned expert
2. **Expert compute:** Each rank processes tokens routed to its local experts
3. **Combine (all-to-all):** Results are sent back to the tokens' home ranks

```
  Before dispatch:        After dispatch:
  Rank 0: [t0,t1,t2,t3]  Rank 0 (Expert 0): [t0,t3,t5]
  Rank 1: [t4,t5,t6,t7]  Rank 1 (Expert 1): [t1,t2,t4,t6,t7]

  Routing: t0→E0, t1→E1, t2→E1, t3→E0, t4→E1, t5→E0, t6→E1, t7→E1

  all-to-all(dispatch) ──▶ expert_compute ──▶ all-to-all(combine)
```

**Memory vs communication:** Each rank stores fewer experts, but pays for two all-to-all collectives per MoE layer. All-to-all is particularly sensitive to network topology — it's an all-pairs exchange, not a tree reduction.

**Cost model:**

With `E` total experts, top-k routing, and capacity factor `C`:

| Cost | Formula | Notes |
|------|---------|-------|
| Expert params per rank | `P_expert / ep` | Each rank holds `E/ep` experts |
| Tokens per expert | `C · k · b · s / E` | Capacity factor controls max tokens; `k` = top-k |
| Comm per MoE layer (dispatch) | `≈ bshΦ · (ep-1)/ep` bytes | Each rank sends `(ep-1)/ep` of its tokens to other ranks |
| Comm per MoE layer (total) | `≈ 2bshΦ · (ep-1)/ep` bytes | Dispatch + combine (symmetric) |
| Compute per rank (expert) | `2 · tokens_per_expert · (E/ep) · d_expert · d_model · 2` | FLOPs for local experts on received tokens |

All-to-all is the hardest collective to scale. Unlike all-reduce (which uses a tree/ring and moves `2M` bytes regardless of `N`), all-to-all with `N` ranks has each rank exchanging with `N-1` peers. The per-link bandwidth demand grows as the message size shrinks with `N`, making bisection bandwidth the bottleneck. This is why EP scaling is limited compared to DP/TP.

**Combined with TP:** When TP is used alongside EP, the dispatch needs to handle the fact that expert weights are further sharded across TP ranks. This requires an additional all-gather/reduce-scatter around the expert computation.

**Seminal paper:** Lepikhin et al., "GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding," 2020. [arxiv:2006.16668](https://arxiv.org/abs/2006.16668)

---

## Part 2: How Megatron-LM and Torchtitan Express These

The strategies in Part 1 are mathematical specifications — they define what gets partitioned and what collectives run where. But there's a separate and equally interesting question: how do you actually express these in code? It turns out there are two genuinely different schools of thought here, and Megatron and torchtitan represent them cleanly.

The fundamental difference: **Megatron encodes parallelism *into* the model. Torchtitan applies parallelism *onto* the model.** In Megatron, you write `ColumnParallelLinear` instead of `nn.Linear` — the model code *is* the parallelism code. In torchtitan, you write a vanilla `nn.Module`, then apply a transformation: `parallelize_module(model, mesh, plan)`.

| Aspect | Megatron-LM | Torchtitan |
|--------|------------|------------|
| Core philosophy | Parallelism baked into model code | Parallelism applied as external transformation |
| TP mechanism | Custom `autograd.Function` subclasses with hand-fused comm+compute | DTensor placements applied via `parallelize_module()` plan dicts |
| Process groups | `RankGenerator` with mask-based group algebra | `DeviceMesh` with `unflatten()` / dimension slicing |
| Communication | Explicit collective calls, hand-scheduled async overlap | Implicit via DTensor redistribution rules |
| Optimization strategy | Engineer hand-tunes each op (assembly) | Runtime + compiler discover optimizations (high-level language) |
| Model portability | Model requires parallel infrastructure to run | Model is vanilla PyTorch; parallelism is a separate concern |

### Philosophical comparison

Megatron's approach lets engineers hand-fuse communication with computation inside a single `autograd.Function` — controlling exactly which CUDA stream each kernel launches on, which async op overlaps with which matmul, which pre-allocated buffer to reuse. The cost is coupling: the model isn't portable without the parallel infrastructure, and changing the parallelism strategy means changing the model.

Torchtitan's plan says *what* the parallel layout should be ("this layer is ColwiseParallel, that one is RowwiseParallel") — not *how* to communicate. The DTensor runtime figures out what collectives are needed based on the placement mismatch between producer and consumer. This is a classic **manual optimization vs. declarative specification** tradeoff. The declarative approach means there's another layer of indirection between you and the hardware when you want to hand-tune something. But PyTorch gives you knobs to reach through that layer:

**Inductor compiler passes.** `torch.compile` lowers your model to an FX graph and runs optimization passes over it. Some built-in passes already discover comm-compute overlaps automatically. The `micro_pipeline_tp_pass` (enabled via `enable_async_tensor_parallel` in torchtitan) pattern-matches `all_gather → matmul` and `matmul → reduce_scatter` in the FX graph and fuses them — doing exactly what Megatron's `LinearWithGradAccumulationAndAsyncCommunication` does by hand, but as a compiler transformation. There's also `reorder_for_compute_comm_overlap` which reorders the graph to hide collectives behind matmuls. You don't need to fork PyTorch to add your own passes — Inductor exposes custom pass hooks as config attributes:

```python
import torch._inductor.config as inductor_config

def my_comm_overlap_pass(graph: torch.fx.Graph):
    # your custom FX graph transformation here
    ...

# Runs after Inductor's built-in pattern matcher
inductor_config.post_grad_custom_post_pass = my_comm_overlap_pass

# Other hook points:
#   joint_custom_pre_pass / joint_custom_post_pass  — joint fwd+bwd graph
#   pre_grad_custom_pass                            — pre-grad IR
#   _pre_fusion_custom_pass / _post_fusion_custom_pass — scheduler IR (before/after kernel fusion)

model = torch.compile(model)  # your pass runs as part of compilation
```

You can also replace Inductor entirely with `torch.compile(model, backend=my_backend_fn)`. What you *can't* do without a fork is modify the internals of a built-in pass — but you can disable it and write your own replacement.

**Custom `autograd.Function`.** Nothing stops you from using these in a torchtitan-style codebase — it's literally what Megatron does, and PyTorch supports it everywhere.

**Custom Triton kernels.** Register custom Triton kernels as `torch.compile`-compatible ops via `torch.library`. Relevant for compute-bound bottlenecks (fused attention, fused MoE gating) rather than communication overlap.

**DTensor op strategies.** DTensor decides which collective to use when redistributing a tensor (e.g., `Shard(0) → Replicate` triggers an all-gather). You can register custom strategies to override the default choices for specific ops.

**FSDP/TP runtime hooks.** `fully_shard()` has built-in prefetching (all-gather the next layer's params while the current layer computes) and `reshard_after_forward` policies. These are runtime-level overlap decisions rather than compiler-level.

The bet torchtitan is making is that these knobs — especially the compiler passes — will converge toward the same performance that Megatron achieves through manual scheduling. The payoff is that the model code stays clean and composable, and optimizations discovered for one model architecture automatically apply to others. The gap is narrowing but not fully closed.

With that framing in mind, let's look at how these two philosophies play out in specific implementations — starting with TP, where the difference is sharpest.

### Tensor Parallelism

**Megatron-LM: Explicit autograd functions**

Megatron's TP is built on `torch.autograd.Function` subclasses. These need some explaining because they're the mechanism that makes communication invisible to the rest of PyTorch's autograd engine.

**What is `autograd.Function`?** When you do `loss.backward()`, PyTorch walks backwards through the computation graph, calling each operation's backward method to compute gradients. Normally, PyTorch knows how to differentiate built-in operations (matmul, add, relu, etc.). But `autograd.Function` lets you define *custom* operations where you specify exactly what happens in both forward and backward. This is how Megatron injects communication into the autograd graph — the communication ops look like normal differentiable operations to PyTorch.

**What is `ctx`?** Every `autograd.Function` has a context object `ctx` that persists between forward and backward. In forward, you save whatever the backward pass will need (tensors via `ctx.save_for_backward()`, other values as attributes like `ctx.group = group`). In backward, you read them back. The `ctx.group = group` line saves the process group so that backward knows which ranks to communicate with. Without it, backward wouldn't know who to all-reduce with.

Here's the key conjugate pair that makes TP work:

[`megatron/core/tensor_parallel/mappings.py#L197-L233`](https://github.com/NVIDIA/Megatron-LM/blob/32efeffd2d3cca48f1bfa575e1f2d0f0feac3d0f/megatron/core/tensor_parallel/mappings.py#L197-L233):

```python
class _CopyToModelParallelRegion(torch.autograd.Function):
    """f: identity in forward, all-reduce in backward."""
    @staticmethod
    def forward(ctx, input_, group):
        ctx.group = group      # save the process group for backward
        return input_           # forward: just pass the input through (identity)

    @staticmethod
    def backward(ctx, grad_output):
        # backward: all-reduce the gradient across TP ranks
        return _reduce(grad_output, ctx.group), None  # None = no gradient for `group`


class _ReduceFromModelParallelRegion(torch.autograd.Function):
    """g: all-reduce in forward, identity in backward."""
    @staticmethod
    def forward(ctx, input_, group):
        return _reduce(input_, group)  # forward: all-reduce the input across TP ranks

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None       # backward: just pass the gradient through (identity)
```

These are wrapped in helper functions that the model code calls — `copy_to_tensor_model_parallel_region(x)` calls `_CopyToModelParallelRegion.apply(x, group)`, which inserts the custom op into the autograd graph. In forward it does nothing (identity), but when backward runs, it all-reduces the gradient. The `.apply()` method is how you call a custom `autograd.Function` — it connects the op into the graph.

Why are they conjugates? Because **an all-reduce in forward requires an identity in backward, and vice versa**. This comes from the chain rule: if the forward pass sums partial results across ranks (all-reduce), then in backward, the incoming gradient already represents the full gradient and just needs to be passed through. If the forward pass is an identity (each rank has the same input), then in backward, the partial gradients from each rank need to be summed (all-reduced).

**How they compose into `ColumnParallelLinear` and `RowParallelLinear`.** These aren't generic utilities — they're wired directly into the model's linear layers. The composition follows the paired TP pattern described in Part 1:

```
  ColumnParallelLinear (first linear in a pair, e.g. W_Q, W_K, W_V, W_up):

    forward:  input_parallel = copy_to_tp_region(input)   ← f: identity fwd, all-reduce bwd
              output = matmul(input_parallel, W_shard)     ← each rank uses its column shard
              return output                                 ← partial result, NOT reduced

    The output stays split — each rank has Y_i = X · A_i. This is intentional:
    GeLU/attention can operate on the split output without communication.


  RowParallelLinear (second linear in a pair, e.g. W_O, W_down):

    forward:  output_parallel = matmul(input_parallel, W_shard)  ← each rank uses its row shard
              output = reduce_from_tp_region(output_parallel)     ← g: all-reduce fwd, identity bwd
              return output                                        ← full result, ready for LayerNorm

    The input is already split (came from ColumnParallelLinear's output).
    The all-reduce sums the partial products: Y = Σ X_i · A_i.
```

The f/g pairing cancels out: ColumnParallelLinear uses `f` (identity fwd, all-reduce bwd) on the input, RowParallelLinear uses `g` (all-reduce fwd, identity bwd) on the output. Between them, no communication happens — that's where GeLU, attention scores, etc. run on the split tensors without any collectives.

Here's the actual `ColumnParallelLinear.forward()` stripped to essentials:

[`megatron/core/tensor_parallel/layers.py#L956-L1053`](https://github.com/NVIDIA/Megatron-LM/blob/32efeffd2d3cca48f1bfa575e1f2d0f0feac3d0f/megatron/core/tensor_parallel/layers.py#L956-L1053):

```python
def forward(self, input_):
    # If not using SP or explicit expert comm, wrap input with f (identity fwd, all-reduce bwd)
    if self.allreduce_dgrad or self.sequence_parallel or self.explicit_expert_comm:
        input_parallel = input_       # skip f — communication handled elsewhere
    else:
        input_parallel = copy_to_tensor_model_parallel_region(input_)  # ← f

    # Matmul with this rank's column shard of the weight
    # (self.weight shape: [output_size_per_partition, input_size])
    output_parallel = self._forward_impl(input=input_parallel, weight=self.weight, ...)

    if self.gather_output:
        output = gather_from_tensor_model_parallel_region(output_parallel)  # all-gather
    else:
        output = output_parallel  # stay split — next layer (GeLU, attention) handles it
    return output
```

And `RowParallelLinear.forward()`:

[`megatron/core/tensor_parallel/layers.py#L1251-L1307`](https://github.com/NVIDIA/Megatron-LM/blob/32efeffd2d3cca48f1bfa575e1f2d0f0feac3d0f/megatron/core/tensor_parallel/layers.py#L1251-L1307):

```python
def forward(self, input_):
    # Input is already split across TP ranks (came from ColumnParallelLinear)
    if self.input_is_parallel:
        input_parallel = input_
    else:
        input_parallel = scatter_to_tensor_model_parallel_region(input_)

    # Matmul with this rank's row shard of the weight
    # (self.weight shape: [output_size, input_size_per_partition])
    output_parallel = self._forward_impl(input=input_parallel, weight=self.weight, ...)

    # Reduce partial products across TP ranks — this is g (all-reduce fwd, identity bwd)
    if self.sequence_parallel:
        output_ = reduce_scatter_to_sequence_parallel_region(output_parallel)  # SP variant
    else:
        output_ = reduce_from_tensor_model_parallel_region(output_parallel)    # ← g
    return output_
```

**The `_forward_impl` inside both layers calls `LinearWithGradAccumulationAndAsyncCommunication`** — a single `autograd.Function` that does the actual matmul and handles all the communication complexity in backward. This is where Megatron gets its performance: instead of separate "matmul" and "communicate" steps, everything is fused into one autograd op that can overlap them.

In forward, when sequence parallelism is enabled, it all-gathers the sequence-sharded input before the matmul (because the matmul needs the full sequence, but SP shards activations along the sequence dim between TP regions):

[`megatron/core/tensor_parallel/layers.py#L448-L493`](https://github.com/NVIDIA/Megatron-LM/blob/32efeffd2d3cca48f1bfa575e1f2d0f0feac3d0f/megatron/core/tensor_parallel/layers.py#L448-L493):

```python
class LinearWithGradAccumulationAndAsyncCommunication(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, ..., sequence_parallel, ..., tp_group):
        # Save everything backward will need
        ctx.save_for_backward(input, weight)
        ctx.sequence_parallel = sequence_parallel
        ctx.tp_group = tp_group

        if sequence_parallel:
            # All-gather the sequence-sharded input so we have the full sequence for matmul
            dim_size = list(input.size())
            dim_size[0] = dim_size[0] * tp_group.size()
            all_gather_buffer = get_global_memory_buffer().get_tensor(dim_size, input.dtype, "mpu")
            dist_all_gather_func(all_gather_buffer, input, group=tp_group)
            total_input = all_gather_buffer
        else:
            total_input = input

        output = torch.matmul(total_input, weight.t())
        return output
```

In backward, it overlaps communication with computation. The ordering is carefully designed: launch the all-gather async, compute dgrad (which doesn't need the all-gathered input), wait for all-gather, then compute wgrad (which does). Meanwhile the reduce-scatter of dgrad also runs async, and `CUDA_DEVICE_MAX_CONNECTIONS=1` ensures the GPU's single hardware queue serializes kernel launches in the right order:

[`megatron/core/tensor_parallel/layers.py#L495-L628`](https://github.com/NVIDIA/Megatron-LM/blob/32efeffd2d3cca48f1bfa575e1f2d0f0feac3d0f/megatron/core/tensor_parallel/layers.py#L495-L628):

```python
    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors

        if ctx.sequence_parallel:
            # 1. Launch async all-gather of input (needed for wgrad, not dgrad)
            handle = dist_all_gather_func(all_gather_buffer, input, group=tp_group, async_op=True)
            total_input = all_gather_buffer

        # 2. Compute dgrad (doesn't need all-gathered input, only weight)
        #    This runs while the all-gather is in flight
        grad_input = grad_output.matmul(weight)

        if ctx.sequence_parallel:
            handle.wait()  # 3. Wait for all-gather to finish

        # 4. Compute wgrad (needs the all-gathered input)
        grad_weight = grad_output.t().matmul(total_input)

        if ctx.sequence_parallel:
            # 5. Launch async reduce-scatter of dgrad
            handle = dist_reduce_scatter_func(sub_grad_input, grad_input, group=tp_group, async_op=True)
            # wgrad computation runs while reduce-scatter is in flight
            # (CUDA_DEVICE_MAX_CONNECTIONS=1 ensures correct ordering)

        # ... handle.wait() at the end ...
        return grad_input, grad_weight, ...
```

The `CUDA_DEVICE_MAX_CONNECTIONS=1` trick deserves a note: by default CUDA can have multiple hardware queues, which means async ops might execute out of order. Setting this to 1 forces a single queue, guaranteeing that kernels execute in launch order. This lets Megatron launch `reduce_scatter(async)` → `wgrad_matmul` and know that the reduce-scatter starts first on the network while the matmul runs on the compute units — true overlap.

**Background: What DTensor is**

Before looking at torchtitan's TP code, it helps to understand DTensor — it's the abstraction everything else builds on.

A `DTensor` is a `torch.Tensor` subclass that carries metadata about how the tensor is distributed across a `DeviceMesh`. The metadata is a tuple of *placements*, one per mesh dimension:

- `Shard(dim)` — the tensor is sharded along dimension `dim` across this mesh dimension. Each rank holds a `1/N`-th slice.
- `Replicate()` — each rank holds a full copy of the tensor.
- `Partial()` — each rank holds a partial result (e.g., partial sum from a row-parallel matmul) that hasn't been reduced yet.

A regular `nn.Linear` has a weight of shape `[out_features, in_features]`. After applying `ColwiseParallel()` on a 4-rank TP mesh, the weight becomes a DTensor with placement `(Shard(0),)` — sharded on dimension 0 (the output features dimension). Each rank's local tensor is `[out_features/4, in_features]`, but the DTensor *knows* that it represents a larger logical tensor.

The key idea is **automatic redistribution**. When a DTensor op detects a placement mismatch between what an operator requires and what its inputs provide, it inserts the necessary collective to fix it. For example, if a matmul needs its input to be `Replicate()` but receives `Shard(1)`, DTensor automatically inserts an all-gather. If a row-parallel matmul produces `Partial()` output but the consumer needs `Replicate()`, DTensor inserts an all-reduce (or a reduce-scatter, to get `Shard()`). These rules are defined per-op in DTensor's *op strategy* registry.

`parallelize_module()` converts an existing `nn.Module`'s parameters from regular tensors to DTensors with the specified placements, and installs input/output hooks that handle the redistribution at module boundaries. The model code is unchanged — it still calls `F.linear(input, self.weight)` — but the DTensor dispatch system intercepts the op and handles the distributed semantics.

**Torchtitan: DTensor placements applied externally**

The model is written as a vanilla `nn.Module`. TP is applied *after construction* via `parallelize_module()` with a plan dict that maps submodule paths to parallel styles:

[`torchtitan/models/llama3/parallelize.py#L187-L263`](https://github.com/pytorch/torchtitan/blob/1ce7f761d76bc408c9e6f32e2bbd2cf9f2ac25ae/torchtitan/models/llama3/parallelize.py#L187-L263):

```python
# Top-level: embeddings, final norm, output projection
parallelize_module(
    model, tp_mesh,
    {
        "tok_embeddings": RowwiseParallel(
            input_layouts=Replicate(), output_layouts=Shard(1),
        ),
        "norm": SequenceParallel(),
        "output": ColwiseParallel(
            input_layouts=Shard(1),
            output_layouts=Shard(-1) if loss_parallel else Replicate(),
            use_local_output=not loss_parallel,
        ),
    },
)

# Per transformer block: the TP + SP plan
for transformer_block in model.layers.values():
    layer_plan = {
        "attention_norm": SequenceParallel(),
        "attention": prepare_module_input(
            input_layouts=(Shard(1), None, None, None),
            desired_input_layouts=(Replicate(), None, None, None),
        ),
        "attention.wq": colwise_parallel(),
        "attention.wk": colwise_parallel(),
        "attention.wv": colwise_parallel(),
        "attention.wo": rowwise_parallel(output_layouts=Shard(1)),
        "ffn_norm": SequenceParallel(),
        "feed_forward": prepare_module_input(
            input_layouts=(Shard(1),),
            desired_input_layouts=(Replicate(),),
        ),
        "feed_forward.w1": colwise_parallel(),
        "feed_forward.w2": rowwise_parallel(output_layouts=Shard(1)),
        "feed_forward.w3": colwise_parallel(),
    }
    parallelize_module(module=transformer_block, device_mesh=tp_mesh, parallelize_plan=layer_plan)
```

The communication is implicit: `PrepareModuleInput` with `input_layouts=Shard(1), desired_input_layouts=Replicate()` means "this module's input is currently sequence-sharded; all-gather it to replicated before the module runs." The DTensor runtime handles the actual collective call.

The data flow within a transformer block:

```
[Shard(1)] → attention_norm (SP) → [Shard(1)]
  → PrepareModuleInput: all-gather → [Replicate()]
  → wq/wk/wv (ColwiseParallel) → [sharded on head dim]
  → attention compute
  → wo (RowwiseParallel, output=Shard(1)) → reduce-scatter → [Shard(1)]
  → ffn_norm (SP) → [Shard(1)]
  → PrepareModuleInput: all-gather → [Replicate()]
  → w1/w3 (ColwiseParallel) → [sharded intermediate]
  → w2 (RowwiseParallel, output=Shard(1)) → reduce-scatter → [Shard(1)]
```

**Key difference:** In Megatron, `ColumnParallelLinear` *is* the linear layer — the model code uses it directly. In torchtitan, `nn.Linear` is the linear layer, and `ColwiseParallel()` is a placement spec applied after the fact. The model code doesn't know about parallelism. This separation makes torchtitan models simpler to read and test single-GPU, at the cost of the parallelism logic being less visible when reading the model.

### FSDP

**Megatron-LM: `DistributedOptimizer` (ZeRO-style)**

Megatron implements ZeRO-style optimizer state sharding via a custom `DistributedOptimizer`. Each DP rank owns a contiguous shard of the gradient buffer and only maintains optimizer state for that shard:

[`megatron/core/optimizer/distrib_optimizer.py`](https://github.com/NVIDIA/Megatron-LM/blob/32efeffd2d3cca48f1bfa575e1f2d0f0feac3d0f/megatron/core/optimizer/distrib_optimizer.py):

```python
# Each bucket is divided into dp_world_size contiguous shards
gbuf_size = bucket.grad_data.numel()
max_gbuf_range_size = gbuf_size // data_parallel_world_size
for r in range(data_parallel_world_size):
    gbuf_world_start = r * max_gbuf_range_size
    gbuf_world_end = min(gbuf_size, gbuf_world_start + max_gbuf_range_size)
```

The flow: backward accumulates into a full gradient buffer → reduce-scatter gives each rank its shard → each rank runs optimizer step on its shard only → all-gather broadcasts updated params.

Megatron uses contiguous gradient buffers with bucket-based overlapping. The `DistributedDataParallel` wrapper manages these buffers and coordinates reduce-scatter/all-gather with the backward pass.

**Torchtitan: PyTorch FSDP2 (`fully_shard()`)**

FSDP2 is applied via the composable API. The pattern is: wrap submodules bottom-up, root module last.

[`torchtitan/models/llama3/parallelize.py#L302-L376`](https://github.com/pytorch/torchtitan/blob/1ce7f761d76bc408c9e6f32e2bbd2cf9f2ac25ae/torchtitan/models/llama3/parallelize.py#L302-L376):

```python
mp_policy = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=reduce_dtype)
fsdp_config = {"mesh": dp_mesh, "mp_policy": mp_policy}

# For PP, do not reshard after forward to avoid per-microbatch all-gathers
reshard_after_forward = not pp_enabled

fully_shard(model.tok_embeddings, **fsdp_config, reshard_after_forward=reshard_after_forward)
for layer_id, transformer_block in model.layers.items():
    fully_shard(transformer_block, **fsdp_config, reshard_after_forward=reshard_after_forward)

# Last layers: no reshard since backward prefetches them immediately
fully_shard([model.norm, model.output], **fsdp_config, reshard_after_forward=False)

# Root wrap
fully_shard(model, **fsdp_config)
```

HSDP (Hybrid Sharded Data Parallelism) is activated by passing a 2D mesh `["dp_replicate", "fsdp"]` — replication on the first dimension, sharding on the second:

```python
names = ["dp_replicate", "fsdp"] if parallel_dims.dp_replicate_enabled else ["fsdp"]
dp_mesh = parallel_dims.get_mesh(names)
```

**Key difference:** Megatron's approach is a custom optimizer wrapper that manages gradient buffers explicitly. Torchtitan uses PyTorch's native FSDP2 which handles all-gather/reduce-scatter automatically via hooks on the module's forward/backward. Megatron has finer-grained control over buffer management and overlapping; torchtitan is more declarative.

### Pipeline Parallelism

**Megatron-LM: Explicit schedule functions**

PP schedules are implemented as standalone functions that orchestrate the entire forward-backward loop. The non-interleaved 1F1B schedule implements three phases explicitly:

[`megatron/core/pipeline_parallel/schedules.py#L1975-L2295`](https://github.com/NVIDIA/Megatron-LM/blob/32efeffd2d3cca48f1bfa575e1f2d0f0feac3d0f/megatron/core/pipeline_parallel/schedules.py#L1975-L2295):

```python
# Warmup: fill the pipeline with forward-only microbatches
num_warmup_microbatches = pp_group.size() - pp_group.rank() - 1
num_warmup_microbatches = min(num_warmup_microbatches, num_microbatches)
num_microbatches_remaining = num_microbatches - num_warmup_microbatches

# Steady state: 1F1B
for k in range(num_microbatches_remaining):
    # Forward + send_forward + recv_backward
    # Backward + send_backward + recv_forward
```

P2P communication uses `torch.distributed.isend`/`irecv` with careful even/odd rank scheduling to avoid deadlocks.

The interleaved variant assigns multiple non-contiguous stages to each rank (e.g., with `num_model_chunks=2`, rank 0 holds stages 0 and 4 out of 8). This reduces the bubble fraction by a factor of `num_model_chunks`.

**Torchtitan: PyTorch's `torch.distributed.pipelining` library**

Torchtitan's PP is built on PyTorch's `torch.distributed.pipelining` library. The pipeline has four steps: decide which modules go on which stage, split the model, wrap each chunk in a `PipelineStage`, and build a schedule.

**Step 1: Distribute layers across stages.** `generate_llm_fqn_per_model_part()` takes the total layer count and number of virtual stages, then distributes layers as evenly as possible. It also accepts `input_weight` and `output_weight` parameters — these treat the embedding and output layers as worth more than one transformer layer for load-balancing purposes, so the first and last stages get fewer transformer layers to compensate for the extra cost of embeddings / output projection:

[`torchtitan/distributed/pipeline_parallel.py#L262-L370`](https://github.com/pytorch/torchtitan/blob/1ce7f761d76bc408c9e6f32e2bbd2cf9f2ac25ae/torchtitan/distributed/pipeline_parallel.py#L262-L370):

```python
def generate_llm_fqn_per_model_part(num_stages, num_layers, input_weight=1, output_weight=1):
    # Treats tok_embeddings as `input_weight` layers, norm+output as `output_weight` layers
    num_effective_layers = num_layers + input_weight + output_weight
    layers_per_stage = num_effective_layers // num_stages

    # First stage: tok_embeddings + (layers_per_stage - input_weight) transformer layers
    # Middle stages: layers_per_stage transformer layers each
    # Last stage: (layers_per_stage - output_weight) transformer layers + norm + output
    ...
```

For a 32-layer model with 8 virtual stages and `input_weight=1, output_weight=1`: effective total is 34, so ~4 per stage. Stage 0 gets `tok_embeddings + layers.0-2` (3 transformer layers), stage 7 gets `layers.29-31 + norm + output` (3 transformer layers), middle stages get 4 transformer layers each.

**Step 2: Split the model by deep-copy + pruning.** `pipeline_module_split()` creates each stage by deep-copying the *entire model* and then deleting everything that doesn't belong to that stage:

[`torchtitan/distributed/pipeline_parallel.py#L373-L513`](https://github.com/pytorch/torchtitan/blob/1ce7f761d76bc408c9e6f32e2bbd2cf9f2ac25ae/torchtitan/distributed/pipeline_parallel.py#L373-L513):

```python
def _build_stage_from_modules(stage_idx, module_names, num_stages):
    model = copy.deepcopy(whole_model)       # full copy (model is on meta device, so cheap)
    modules_to_keep = set(module_names)

    for module_name, module_value in model.named_children():
        if isinstance(module_value, (nn.ModuleDict, nn.ModuleList)):
            # Keep only the specified layers, delete the rest
            layers_to_keep = {name.split(".", 1)[1] for name in modules_to_keep
                              if name.startswith(f"{module_name}.")}
            # ... prune module_value to keep only layers_to_keep ...
        elif module_name not in modules_to_keep:
            setattr(model, module_name, None)   # delete this submodule

    stage = PipelineStage(model, stage_idx, num_stages, device, group=pp_mesh.get_group("pp"))
    return stage, model
```

This works because the model's `forward()` is written to tolerate `None` submodules — e.g., `if self.tok_embeddings is not None: ...`. The model is initially on meta device (no actual tensor storage), so `deepcopy` is cheap.

Which stages a given PP rank builds depends on the schedule style:

```python
if style == "loop":        # interleaved 1F1B, looped BFS
    # rank 0: [0, 4], rank 1: [1, 5], rank 2: [2, 6], rank 3: [3, 7]
    stage_indices = tuple(pp_rank + s * pp_degree for s in range(stages_per_rank))
elif style == "v":         # ZBV, DualPipeV
    # rank 0: [0, 7], rank 1: [1, 6], rank 2: [2, 5], rank 3: [3, 4]
    stage_v_pairs = list(zip(range(pp_degree), range(num_stages - 1, pp_degree - 1, -1)))
    stage_indices = stage_v_pairs[pp_rank]
```

**Step 3: `PipelineStage` — the communication wrapper.** `PipelineStage` wraps a model chunk and adds the P2P communication infrastructure. It manages:

- **Activation send/recv.** Pre-allocates receive buffers for each microbatch. Forward activations go downstream via `dist.P2POp(dist.isend, ...)` and `dist.P2POp(dist.irecv, ...)`, batched together with `dist.batch_isend_irecv()`. Backward gradients flow the opposite direction using the same mechanism.
- **Forward cache.** After each forward chunk, saves `(output_tuple, input_values)` in `self.fwd_cache[chunk_id]` — these are needed for the backward pass.
- **Backward split support.** For zero-bubble schedules, backward needs to be split into `stage_backward_input()` (computes `dL/dX` via `torch.autograd.grad()` with `retain_graph=True`) and `stage_backward_weight()` (computes `dL/dW` from saved intermediate gradients). This split is what enables ZB1P and ZBV — you run `B_input` on the critical path and defer `B_weight` to fill bubbles.
- **Local optimization for V-schedules.** When two stages live on the same rank (e.g., stages 0 and 7 on rank 0 in a V-layout), activations are passed by direct tensor reference instead of P2P send/recv, using `detach().requires_grad_(True)` to create a fresh autograd leaf at the boundary.

**Step 4: The schedule IR and execution.** Schedules are defined as sequences of `_Action` tuples:

```python
class _Action(NamedTuple):
    stage_index: int
    computation_type: _ComputationType   # FORWARD, BACKWARD_INPUT, BACKWARD_WEIGHT, FULL_BACKWARD, ...
    microbatch_index: int | None = None
```

String representation: `"2F0"` means stage 2, forward, microbatch 0. `"1I3"` means stage 1, backward-input, microbatch 3.

Schedule classes generate a **compute-only** schedule — just F, I, W, and B actions — which then gets *lowered* through a series of passes that add communication and memory management:

1. `_merge_bw()` — merges adjacent I+W for the same microbatch into a single B (full backward)
2. `_add_unshard_reshard()` — inserts FSDP unshard (all-gather params) and reshard (release) ops, with prefetch policy
3. `_add_send_recv()` — inserts SEND_F, RECV_F, SEND_B, RECV_B ops between compute actions, using dependency tracking to ensure correct ordering
4. `_add_reduce_grad()` — appends gradient reduction (FSDP/DDP sync) after the last backward per stage

The `_PipelineScheduleRuntime` then executes this fully lowered IR action-by-action, dispatching each `_Action` to the appropriate stage method (`forward_one_chunk`, `backward_one_chunk`, `backward_weight_one_chunk`) or P2P communication primitive. All the advanced schedules (Interleaved 1F1B, ZBV, DualPipeV) inherit from this runtime — they only need to generate the compute-only schedule; the communication is handled generically.

SPMD parallelisms (TP, AC, compile, FSDP) are applied *per pipeline stage* after splitting:

[`torchtitan/distributed/pipeline_parallel.py#L145-L157`](https://github.com/pytorch/torchtitan/blob/1ce7f761d76bc408c9e6f32e2bbd2cf9f2ac25ae/torchtitan/distributed/pipeline_parallel.py#L145-L157):

```python
for i, m in enumerate(model_parts):
    m = parallelize_fn(m, parallel_dims=parallel_dims, ...)
    model_parts[i] = m
    stages[i].submod = m    # update the stage's module reference
```

**Key difference:** Megatron's PP is deeply integrated — the schedule function directly controls the training loop, calling `forward_step()` and `backward_step()` and managing P2P communication inline. Torchtitan separates the concerns: the schedule is an IR that gets lowered and executed by a generic runtime. Adding a new schedule means defining a new compute-only action sequence — the communication and FSDP integration come for free from the lowering passes. This is the same "declarative spec → runtime execution" pattern as DTensor for TP.

---

## Part 3: Composability

### The N-Dimensional Grid

All parallelism strategies compose by partitioning the set of ranks along orthogonal dimensions. The fundamental constraint:

```
tp × cp × ep × dp × pp = world_size
```

Each rank is uniquely identified by its coordinates in this grid: `(tp_rank, cp_rank, ep_rank, dp_rank, pp_rank)`.

### Megatron: `RankGenerator` with Mask-Based Group Algebra

Megatron uses an algebraic approach. The `RankGenerator` takes the parallelism sizes and an ordering string, then generates process groups for any combination of dimensions using boolean masks:

[`megatron/core/parallel_state.py#L449-L524`](https://github.com/NVIDIA/Megatron-LM/blob/32efeffd2d3cca48f1bfa575e1f2d0f0feac3d0f/megatron/core/parallel_state.py#L449-L524):

```python
class RankGenerator(object):
    def __init__(self, tp, ep, dp, pp, cp, order, rank_offset=0):
        self.world_size = tp * dp * pp * cp * ep
        self.name_to_size = {"tp": tp, "pp": pp, "dp": dp, "ep": ep, "cp": cp}
        self.ordered_size = [self.name_to_size[token] for token in order.split("-")]

    def get_mask(self, order, token):
        """e.g. order='tp-cp-ep-dp-pp', token='tp-dp' -> [True, False, False, True, False]"""
        ordered_token = order.split("-")
        mask = [False] * len(ordered_token)
        for t in token.split("-"):
            mask[ordered_token.index(t)] = True
        return mask

    def get_ranks(self, token):
        mask = self.get_mask(self.order, token)
        return generate_masked_orthogonal_rank_groups(self.world_size, self.ordered_size, mask)
```

To get TP groups: `generator.get_ranks('tp')`. To get the combined DP-CP group for gradient reduction: `generator.get_ranks('dp-cp')`. The mask algebra computes the correct rank subsets automatically.

Two separate `RankGenerator` instances are used — one for the decoder (where `ep=1`) and one for experts (where `cp=1`):

[`megatron/core/parallel_state.py#L777-L808`](https://github.com/NVIDIA/Megatron-LM/blob/32efeffd2d3cca48f1bfa575e1f2d0f0feac3d0f/megatron/core/parallel_state.py#L777-L808):

```python
decoder_rank_generator = RankGenerator(
    tp=tensor_model_parallel_size, ep=1, dp=data_parallel_size,
    pp=pipeline_model_parallel_size, cp=context_parallel_size, order=order,
)
expert_decoder_rank_generator = RankGenerator(
    tp=expert_tensor_parallel_size, ep=expert_model_parallel_size,
    dp=expert_data_parallel_size, pp=pipeline_model_parallel_size, cp=1, order=order,
)
```

This separation is enforced by an assertion — EP and CP cannot both be > 1 in the same generator because they occupy the same position in the rank decomposition.

### Torchtitan: `DeviceMesh` with `unflatten()`

Torchtitan creates a single flat 1D world mesh and then unflatten it into different views for different purposes:

[`torchtitan/distributed/parallel_dims.py#L137-L166`](https://github.com/pytorch/torchtitan/blob/1ce7f761d76bc408c9e6f32e2bbd2cf9f2ac25ae/torchtitan/distributed/parallel_dims.py#L137-L166):

```python
batch = self.dp_replicate * self.dp_shard
fsdp = self.dp_shard * self.cp           # CP is folded into FSDP
efsdp = fsdp * self.tp // (self.etp * self.ep)

self._world_mesh = init_device_mesh(device_type, (self.world_size,), mesh_dim_names=("world",))

# Three views of the same world:
dataloading_mesh = unflatten_mesh(self._world_mesh, ("pp", "batch", "cp", "tp"),
                                  (self.pp, batch, self.cp, self.tp))
dense_mesh = unflatten_mesh(self._world_mesh, ("pp", "dp_replicate", "fsdp", "tp"),
                            (self.pp, self.dp_replicate, fsdp, self.tp))
sparse_mesh = unflatten_mesh(self._world_mesh, ("pp", "dp_replicate", "efsdp", "ep", "etp"),
                             (self.pp, self.dp_replicate, efsdp, self.ep, self.etp))
```

The three views expose different facets of the same rank mapping:
- **dataloading**: `(pp, batch, cp, tp)` — for computing per-rank batch sizes
- **dense**: `(pp, dp_replicate, fsdp, tp)` — for non-MoE parameter sharding
- **sparse**: `(pp, dp_replicate, efsdp, ep, etp)` — for MoE expert parameter sharding

To get a sub-mesh for a specific parallelism, you slice by dimension name: `parallel_dims.get_mesh("tp")` or `parallel_dims.get_mesh(["dp_replicate", "fsdp"])`.

### Why Rank Ordering Matters

The ordering string (e.g., `"tp-cp-ep-dp-pp"`) determines which ranks are adjacent in the global rank space. Adjacent ranks in the innermost dimensions share the fastest interconnect. The default ordering places TP innermost (lowest stride), which means TP groups are physically co-located on the same node sharing NVLink bandwidth. PP is outermost, since it only needs P2P bandwidth (often over InfiniBand).

```
  Default order: tp-cp-ep-dp-pp

  Rank 0  Rank 1  Rank 2  Rank 3 | Rank 4  Rank 5  Rank 6  Rank 7
  ├── TP group 0 (NVLink) ───────┤ ├── TP group 1 (NVLink) ───────┤
  ├── DP group (IB) ─────────────────── DP group (IB) ─────────────┤
```

### Combined Groups

Some operations need process groups that span multiple parallelism dimensions. Key examples:

- **`dp-cp` (gradient reduction):** Weights are replicated across CP ranks (each rank has the full model, just different sequence chunks), so gradient all-reduce must span both DP and CP ranks.
- **`fsdp = dp_shard × cp`:** In torchtitan, CP is folded into the FSDP mesh dimension. FSDP's all-gather and reduce-scatter operate over `dp_shard * cp` ranks, handling both data-parallel sharding and the weight synchronization needed by CP.
- **`tp-ep` (expert computation):** When expert TP differs from decoder TP, a combined group is needed for the all-gather/reduce-scatter around expert computation.

### The Parallelization Application Order

Torchtitan applies parallelism strategies in a strict order. This order is not arbitrary — each step depends on the previous:

[`torchtitan/models/llama3/parallelize.py#L61-L172`](https://github.com/pytorch/torchtitan/blob/1ce7f761d76bc408c9e6f32e2bbd2cf9f2ac25ae/torchtitan/models/llama3/parallelize.py#L61-L172):

```
1. Tensor Parallelism (TP)          — parallelize_module() with plan dicts
2. Context Parallelism (CP)         — parallelize_module() on attention modules
3. Activation Checkpointing (AC)    — wrap transformer blocks
4. torch.compile                    — compile each transformer block
5. FSDP / HSDP / DDP               — fully_shard() each module
```

**Why this order:**

1. **TP first:** Converts `nn.Linear` weight parameters to DTensors with specific sharding placements. Must happen before any other transformation that wraps or modifies parameters.

2. **CP second:** Applied to attention modules via `parallelize_module()`. Both TP and CP modify the same attention modules, but TP shards the weights while CP shards the sequence — they're orthogonal.

3. **AC third:** Wraps transformer blocks in checkpoint wrappers. Must happen before compile so the compiler can see checkpoint boundaries.

4. **compile fourth:** Compiles each transformer block individually (not the full model). Must happen after AC and before FSDP — FSDP hooks would cause graph breaks if compile ran after FSDP.

5. **FSDP last:** Adds runtime hooks for all-gather/reduce-scatter. Must wrap around the already-compiled, checkpointed, TP-sharded blocks.

---

## Part 4: Performance

### Published Benchmarks

**Megatron-LM** (from [README](https://github.com/NVIDIA/Megatron-LM/blob/32efeffd2d3cca48f1bfa575e1f2d0f0feac3d0f/README.md)):

- Up to **47% MFU** on H100 clusters, training models from 2B to 462B parameters
- 6144 H100 GPUs: 462B parameter model benchmarked
- **Superlinear weak scaling:** MFU increases from 41% (smallest model) to 47-48% (largest) because larger GEMMs have higher arithmetic intensity
- **Strong scaling GPT-3 175B:** 96 to 4608 H100 GPUs, MFU drops from 47% to 42% as communication becomes more exposed
- Config: vocab=131072, seq_len=4096, with `--overlap-grad-reduce --overlap-param-gather --tp-comm-overlap`
- End-to-end measurement including data loading, optimizer, communication, and logging

**Torchtitan** (from [benchmarks/](https://github.com/pytorch/torchtitan/tree/1ce7f761d76bc408c9e6f32e2bbd2cf9f2ac25ae/benchmarks)):

Llama 3.1 8B on H100s:

| Config | GPUs | TPS/GPU | MFU |
|--------|------|---------|-----|
| FSDP | 8 | 5,762 | ~33% |
| FSDP + compile | 8 | 6,667 | ~39% |
| FSDP + compile + Float8 | 8 | 8,532 | N/A* |
| FSDP | 128 | 5,605 | — |
| FSDP + compile | 128 | 6,514 | — |
| FSDP + compile + Float8 | 128 | 8,380 | — |

*\* MFU is not well-defined when both BF16 and FP8 tensor cores are used with different peak FLOPS.*

Llama 3.1 larger models on H100s:

| Config | Model | GPUs | TPS/GPU |
|--------|-------|------|---------|
| 2D (FSDP32+TP8) + compile + Float8 | 70B | 256 | 829 |
| 2D + AsyncTP | 70B | 256 | 876 |
| 3D (FSDP8+TP8+PP8) + 1F1B | 405B | 512 | 100 |
| 3D + Interleaved 1F1B | 405B | 512 | 128 |

Context parallelism scaling (405B, 512 H100s, 3D + CP):

| FSDP | CP | Seq Length | TPS/GPU |
|------|----|-----------|---------|
| 8 | 1 | 32,768 | 76 |
| 4 | 2 | 65,536 | 47 |
| 2 | 4 | 131,072 | 31 |
| 1 | 8 | 262,144 | 16 |

TPS/GPU drops as CP increases because: longer sequences mean more attention compute (O(n²)), and CP communication (ring KV exchange) is harder to overlap than FSDP communication.

### What the Numbers Tell Us

Megatron-LM achieves higher absolute MFU (47% vs 33-39%), but the comparison is not apples-to-apples:
- Different models (GPT-3 vs Llama 3.1)
- Different scales (up to 6144 GPUs vs up to 512)
- Megatron uses custom CUDA kernels and Transformer Engine, torchtitan benchmarks use stock PyTorch

The torchtitan benchmarks show clear wins from `torch.compile` (~16% improvement) and Float8 (~28% improvement over baseline), demonstrating that the PyTorch compiler stack can capture significant optimization opportunities even without hand-written kernels.

The async TP results are notable: 9-16% speedup by pipelining all-gather/compute/reduce-scatter within the compiler graph. This is an optimization that's essentially impossible to do by hand but falls naturally out of the compiler approach.

---

## Part 5: Implications for naniGPT

### What to build, and in what order

1. **DDP first.** Replicate the model, partition data, all-reduce gradients. This is the simplest parallelism and the baseline for everything else. Build the training loop, measurement infrastructure, and profiling here.

2. **FSDP second.** Use PyTorch's `fully_shard()` API. This unlocks training models that don't fit in one GPU's memory. Follow torchtitan's approach — it's more accessible and the model code stays clean.

3. **TP third.** Implement column/row parallel following the Megatron pattern, but use torchtitan's `parallelize_module()` approach to keep model code separate from parallelism code. Profile the difference between NVLink and IB for the all-reduce to understand why TP is placed innermost.

4. **SP alongside TP.** Once TP works, SP is a natural extension — replace the all-reduce with reduce-scatter/all-gather and shard activations in LayerNorm regions. This is more about memory savings than throughput.

5. **EP when building MoE.** Expert parallelism only matters with MoE models. The all-to-all dispatch/combine is the interesting part — profile it to understand the network sensitivity.

6. **PP and CP as needed.** PP adds significant complexity (schedule management, microbatching, bubble optimization) for a payoff that's most visible at very large scale. CP is only needed for long-context training. Both are worth studying but lower priority for small-scale experiments.

### Which approach to follow

Follow torchtitan's style for the parallelism application pattern (external, declarative, composable), but study Megatron's explicit autograd functions to understand what the DTensor runtime is actually doing under the hood. Build the `DeviceMesh`-style composability — it's cleaner than managing process groups manually.

For the `RankGenerator` vs `DeviceMesh` question: `DeviceMesh` is the right abstraction for new code. But implementing a simplified version of `RankGenerator`'s mask algebra is a good exercise for understanding why the rank ordering matters.

### Analytical cost model

All the formulas in this entry — per-layer FLOPs, communication volumes per collective, bubble fractions, memory breakdowns — should be executable, not just prose. naniGPT should have a cost model that takes a model config + parallelism config + hardware config and projects compute time, communication time, memory footprint, and MFU *before running anything*. This lets you sweep parallelism configs cheaply (milliseconds, zero GPUs) and answer questions like "at what TP degree does communication start dominating?" or "does this config even fit in memory?" When you later run the actual training and measure real MFU, the gap between projected and actual is itself the interesting thing to study. See [journal entry on cost model design](2026-02-25-cost-model-design.md).
