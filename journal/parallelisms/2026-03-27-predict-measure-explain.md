# The Predict-Measure-Explain Loop

*2026-03-27*

The [cost model design](2026-02-25-cost-model-design.md) sketched an analytical cost model that projects step time and MFU from model dimensions, parallelism config, and hardware specs. The [architecture design](2026-03-26-parallelism-architecture-design.md) described the parallelism system that executes the actual training. This entry connects them: **how do we build a tight loop between what we predict should happen and what actually happens, and use the gap to drive optimization?**

---

## The Loop

```
    ┌──────────────────────────────────────────────────────────────┐
    │                                                              │
    │   1. PREDICT                                                 │
    │   Cost model + hardware benchmarks → projected breakdown     │
    │   "TP comm should take 2.1ms at 87% of NVLink BW,           │
    │    compute should take 3.7ms at 89% of peak FLOPS"           │
    │                                                              │
    └──────────────────────┬───────────────────────────────────────┘
                           │
                           ▼
    ┌──────────────────────────────────────────────────────────────┐
    │                                                              │
    │   2. MEASURE                                                 │
    │   Comm-aware profiler → actual breakdown                     │
    │   "TP comm took 3.8ms, compute took 4.1ms, other 0.9ms"     │
    │                                                              │
    └──────────────────────┬───────────────────────────────────────┘
                           │
                           ▼
    ┌──────────────────────────────────────────────────────────────┐
    │                                                              │
    │   3. EXPLAIN THE GAP                                         │
    │   Automatic comparison → ranked opportunities                │
    │   "TP comm at 58% BW. Compute at 91% peak. 0.9ms overhead." │
    │                                                              │
    └──────────────────────┬───────────────────────────────────────┘
                           │
                           ▼
    ┌──────────────────────────────────────────────────────────────┐
    │                                                              │
    │   4. ACT                                                     │
    │   Change backend, plan, or config → go to step 1             │
    │   "Try AsyncBackend for TP, torch.compile for compute"       │
    │                                                              │
    └──────────────────────────────────────────────────────────────┘
```

Each step in the loop produces artifacts that feed the next. The vocabulary is shared across all steps — the same `ParallelPlan` that drives the training also parameterizes the cost model and labels the profiler output. When you change `tp_size` from 8 to 4, the prediction updates, the profiler categories update, and the gap report updates.

---

## Part 1: Hardware Micro-Benchmarks

The cost model design used spec-sheet numbers — 989 TFLOPS for H100, 450 GB/s NVLink. These are theoretical peaks that real workloads never hit. The gap between spec and achievable varies by operation, message size, and tensor shape. Without measuring it, the cost model is working from fantasy.

### What to benchmark

```python
# nanigpt/benchmarks/hardware.py

@dataclass
class BenchmarkResult:
    """Result from a single micro-benchmark."""
    achieved: float          # achieved throughput (TFLOPS or GB/s)
    peak: float              # theoretical peak
    efficiency: float        # achieved / peak
    samples: list[float]     # raw measurements for variance analysis

@dataclass
class HardwareProfile:
    """Measured capabilities of the current hardware."""
    gpu_name: str

    # Compute
    matmul_tflops: dict[tuple[int,int,int], BenchmarkResult]
    # keyed by (M, N, K) — different shapes hit different efficiency

    # Memory bandwidth
    hbm_bandwidth: BenchmarkResult

    # Communication (per group type)
    allreduce_bandwidth: dict[int, BenchmarkResult]
    # keyed by message size in bytes — bandwidth varies dramatically

    allgather_bandwidth: dict[int, BenchmarkResult]
    reducescatter_bandwidth: dict[int, BenchmarkResult]
    alltoall_bandwidth: dict[int, BenchmarkResult]
    p2p_bandwidth: BenchmarkResult
```

### Compute benchmarks

GEMM throughput depends on matrix shape. A `[4096, 4096] × [4096, 4096]` matmul might hit 89% of peak. A `[128, 4096] × [4096, 4096]` (small batch) might hit 40%. This matters because TP reduces the per-rank matmul size — at TP=8, each rank's Q projection goes from `[bs, h] × [h, h]` to `[bs, h] × [h, h/8]`. Smaller matmuls → lower compute efficiency → TP becomes less worthwhile.

```python
def benchmark_matmul(m: int, n: int, k: int, dtype=torch.bfloat16,
                     warmup=50, trials=200) -> BenchmarkResult:
    """Measure achieved TFLOPS for a specific GEMM shape."""
    a = torch.randn(m, k, dtype=dtype, device="cuda")
    b = torch.randn(k, n, dtype=dtype, device="cuda")

    # Warmup
    for _ in range(warmup):
        torch.mm(a, b)
    torch.cuda.synchronize()

    # Timed trials
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(trials):
        start.record()
        torch.mm(a, b)
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))

    flops = 2 * m * n * k
    median_ms = sorted(samples)[len(samples) // 2]
    achieved_tflops = flops / (median_ms / 1000) / 1e12
    peak_tflops = GPU_PEAK_TFLOPS.get(detect_gpu(), 989.0)

    return BenchmarkResult(
        achieved=achieved_tflops,
        peak=peak_tflops,
        efficiency=achieved_tflops / peak_tflops,
        samples=samples,
    )
```

The shapes that matter are the ones from the actual model. Given a `TransformerConfig` and a `ParallelPlan`, the benchmark suite should auto-generate the relevant GEMM shapes:

```python
def model_gemm_shapes(config: TransformerConfig, plan: ParallelPlan,
                      batch_size: int) -> list[tuple[int, int, int, str]]:
    """Extract the GEMM shapes that will actually run during training."""
    bs = batch_size * config.seq_len
    h = config.d_model
    tp = plan.tp_size
    ffn = config.d_ff or 4 * h

    return [
        (bs, h // tp, h,       "Q/K/V projection (col-parallel)"),
        (bs, h, h // tp,       "output projection (row-parallel)"),
        (bs, ffn // tp, h,     "FFN up (col-parallel)"),
        (bs, h, ffn // tp,     "FFN down (row-parallel)"),
    ]
```

### Communication benchmarks

This is where spec-sheet numbers diverge most from reality. NVLink bandwidth depends on message size:

```
Message size    Spec BW    Achieved BW    Efficiency
─────────────────────────────────────────────────────
1 KB            450 GB/s   ~10 GB/s       2%    ← latency-dominated
64 KB           450 GB/s   ~120 GB/s      27%
1 MB            450 GB/s   ~300 GB/s      67%
16 MB           450 GB/s   ~400 GB/s      89%
256 MB          450 GB/s   ~420 GB/s      93%
```

(Numbers are illustrative — the point is the shape of the curve, not the exact values.)

```python
def benchmark_allreduce(size_bytes: int, group: ProcessGroup,
                        warmup=20, trials=100) -> BenchmarkResult:
    """Measure achieved all-reduce bandwidth at a given message size."""
    numel = size_bytes // 2  # bf16
    tensor = torch.randn(numel, dtype=torch.bfloat16, device="cuda")

    for _ in range(warmup):
        dist.all_reduce(tensor, group=group)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(trials):
        start.record()
        dist.all_reduce(tensor, group=group)
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))

    # All-reduce transfers 2*(N-1)/N * size bytes (ring algorithm)
    world = dist.get_world_size(group)
    algo_bytes = 2 * (world - 1) / world * size_bytes
    median_ms = sorted(samples)[len(samples) // 2]
    achieved_bw = algo_bytes / (median_ms / 1000)

    return BenchmarkResult(
        achieved=achieved_bw / 1e9,  # GB/s
        peak=get_link_bandwidth(group),
        efficiency=achieved_bw / 1e9 / get_link_bandwidth(group),
        samples=samples,
    )
```

The benchmarks should sweep message sizes to build a bandwidth-vs-size curve. This curve is what the cost model uses for predictions — not a single spec number, but an interpolation from the curve at the actual message sizes the parallelism config produces.

```python
def benchmark_comm_sweep(group: ProcessGroup,
                         sizes: list[int] | None = None) -> dict[int, BenchmarkResult]:
    """Sweep message sizes to build a bandwidth curve."""
    if sizes is None:
        sizes = [2**i for i in range(10, 29)]  # 1KB to 256MB
    return {size: benchmark_allreduce(size, group) for size in sizes}
```

### From benchmarks to calibrated predictions

The cost model gains a `HardwareProfile` that replaces the spec-sheet `HardwareConfig`:

```python
# Before: spec-sheet fantasy
h100 = HardwareConfig(flops_bf16=989e12, nvlink_bandwidth=450e9)

# After: measured reality
profile = run_hardware_benchmarks(tp_group=mesh.get_group("tp"))
# profile.matmul_tflops[(4096, 512, 4096)] = BenchmarkResult(achieved=880, peak=989, eff=0.89)
# profile.allreduce_bandwidth[16_000_000] = BenchmarkResult(achieved=400, peak=450, eff=0.89)
# profile.allreduce_bandwidth[1_000_000] = BenchmarkResult(achieved=300, peak=450, eff=0.67)

# Cost model uses profile to predict with real numbers
report = cost_model.project(plan, profile)
```

---

## Part 2: Communication-Aware Profiling

The existing profiler (`nanigpt/profiling/timer.py`) breaks the training step into DATA, FORWARD, BACKWARD, OPTIMIZER phases. This is good for overall step decomposition but doesn't tell you where the time goes *within* forward/backward — specifically, how much is compute vs communication.

### What we need

The profiler should produce a breakdown like:

```
Step 42 breakdown:
  data:              0.12 ms  ( 1.1%)
  forward_compute:   1.85 ms  (17.3%)
  forward_tp_comm:   1.92 ms  (18.0%)
  forward_fsdp_comm: 0.65 ms  ( 6.1%)
  backward_compute:  3.70 ms  (34.6%)
  backward_tp_comm:  1.88 ms  (17.6%)
  backward_fsdp_comm:0.65 ms  ( 6.1%)
  optimizer:         0.31 ms  ( 2.9%)
  ──────────────────────────────────
  total:            10.68 ms

  Achieved MFU: 41.0%
  Compute efficiency: 91% of peak FLOPS
  TP comm efficiency: 58% of measured NVLink BW
  FSDP comm efficiency: 82% of measured IB BW
```

### Two approaches

**Approach 1: Instrument `comm.py` directly.**

Every comm primitive in `comm.py` wraps its collective in a `measure()` call:

```python
class CopyToParallelRegion(torch.autograd.Function):
    @staticmethod
    def backward(ctx, grad_output):
        with measure(EventType.TP_COMM):      # ← new
            dist.all_reduce(grad_output, group=ctx.group)
        return grad_output, None
```

This is simple and precise — the timing is right at the source. The downside is that `measure()` records CUDA events, which adds a small amount of overhead per comm op. With 4 comm ops per layer × 64 layers, that's 256 extra event pairs per step.

**Approach 2: Post-process `torch.profiler` traces.**

`torch.profiler` already captures NCCL kernel durations when profiling is enabled. We can post-process the trace to extract comm time without any instrumentation in `comm.py`:

```python
def extract_comm_time(trace) -> dict[str, float]:
    """Extract NCCL kernel durations from a torch.profiler trace."""
    comm_time = defaultdict(float)
    for event in trace.events():
        if "nccl" in event.name.lower():
            # Classify by kernel name: ncclAllReduce → TP_COMM, etc.
            category = classify_nccl_kernel(event.name, event)
            comm_time[category] += event.cuda_time_total / 1e3  # μs to ms
    return comm_time
```

The downside: `torch.profiler` has non-trivial overhead when enabled, and the classification (which NCCL call belongs to TP vs FSDP?) requires heuristics or matching against known group handles.

**Recommendation: start with Approach 1.** The per-event overhead is measurable (~1μs per event pair, ~256μs total per step) but tiny compared to a step that takes 10ms. It's exact, requires no heuristics, and works without enabling the full `torch.profiler`. We can gate it behind a flag for zero-overhead production training.

### New event types

```python
class EventType(StrEnum):
    STEP = "step"
    DATA = "data"
    FORWARD = "forward"
    BACKWARD = "backward"
    OPTIMIZER = "optimizer"
    EVAL = "eval"

    # Communication breakdown (new)
    TP_COMM = "tp_comm"
    EP_COMM = "ep_comm"
    FSDP_COMM = "fsdp_comm"
    PP_COMM = "pp_comm"
```

The comm primitives in `comm.py` use these directly. Since `comm.py` already knows what kind of communication it's doing (that's its whole purpose), the labeling is natural — no post-hoc classification needed.

### Computing efficiency ratios

The profiler knows the elapsed time for each comm op. Combined with the message size (which the comm primitive knows) and the hardware profile's bandwidth curve, we can compute efficiency in real-time:

```python
def comm_efficiency(elapsed_ms: float, message_bytes: int,
                    profile: HardwareProfile, group: ProcessGroup) -> float:
    """What fraction of achievable bandwidth did this comm op use?"""
    achievable_bw = profile.allreduce_bandwidth[message_bytes].achieved  # GB/s
    actual_bw = message_bytes / (elapsed_ms / 1000) / 1e9
    return actual_bw / achievable_bw
```

This is the key number. If a comm op achieves 95% of the benchmarked bandwidth, there's nothing to optimize. If it achieves 58%, the question is *why* — contention from overlapping comms, suboptimal message sizes, or lack of overlap with compute.

---

## Part 3: The Cost Model

The [cost model design entry](2026-02-25-cost-model-design.md) already specifies the formulas and API shape. What changes with hardware benchmarks:

### Before: spec-sheet predictions

```python
# TP comm time = message_bytes / spec_nvlink_bw
tp_time = (2 * Φ * b * s * h) / 450e9  # seconds
```

### After: calibrated predictions

```python
# TP comm time = message_bytes / benchmarked_bw_at_this_message_size
msg_size = 2 * Φ * b * s * h  # bytes for this collective
benchmarked_bw = profile.interpolate_bandwidth("allreduce", msg_size)
tp_time = msg_size / benchmarked_bw  # seconds
```

Similarly for compute:

```python
# Before: FLOPS / peak_tflops
compute_time = step_flops / 989e12

# After: use benchmarked TFLOPS for the actual GEMM shapes
total_compute_time = 0
for m, n, k, label in model_gemm_shapes(config, plan, batch_size):
    gemm_flops = 2 * m * n * k
    benchmarked_tflops = profile.matmul_tflops[(m, n, k)].achieved
    total_compute_time += gemm_flops / (benchmarked_tflops * 1e12)
```

The gap between calibrated predictions and measured results now isolates the *interesting* factors: overlap efficiency, scheduling overhead, kernel launch costs, memory bandwidth bottlenecks — things the analytical model can't predict but profiling can measure.

### The CostReport with calibration

```python
@dataclass
class CostReport:
    # Predictions (from cost model)
    predicted_compute_ms: float
    predicted_tp_comm_ms: float
    predicted_fsdp_comm_ms: float
    predicted_pp_bubble_ms: float
    predicted_total_ms: float
    predicted_mfu: float

    # Calibration source
    using_benchmarks: bool  # True if HardwareProfile, False if spec-sheet

    # After measurement (filled in by gap analysis)
    measured_compute_ms: float | None = None
    measured_tp_comm_ms: float | None = None
    measured_fsdp_comm_ms: float | None = None
    measured_other_ms: float | None = None
    measured_total_ms: float | None = None
    measured_mfu: float | None = None
```

---

## Part 4: The Gap Report

The gap report is the artifact that closes the loop — it compares predicted vs measured and ranks optimization opportunities.

```python
def gap_report(prediction: CostReport, measured: StepMetrics,
               profile: HardwareProfile) -> GapReport:
    """Compare predicted breakdown against measured breakdown."""
    ...
```

```
Gap Report: tp=8, dp=4, pp=1, FSDP
═══════════════════════════════════════════════════════════════════

                    Predicted    Measured    Ratio    Notes
                    ─────────    ────────    ─────    ──────────
Compute (ms):          3.73        4.10     0.91    91% of benchmarked peak
TP comm (ms):          2.15        3.80     0.57    58% of benchmarked BW
FSDP comm (ms):        1.31        1.20     1.09    overlap helping
PP bubble (ms):         —           —        —      no PP
Other (ms):             —          0.90      —      kernel launch + Python
                    ─────────    ────────
Total (ms):            7.19       10.00
MFU:                  52.1%       41.0%              Gap: 11.1pp

Top opportunities (ranked by recoverable ms):
  1. TP comm: 3.80ms measured vs 2.15ms predicted (1.65ms gap)
     → 58% of benchmarked NVLink BW
     → try: AsyncBackend (overlap with compute), torch.compile
  2. Unexplained overhead: 0.90ms
     → try: CUDA graphs, reduce kernel launch count, torch.compile
  3. Compute: 4.10ms vs 3.73ms predicted (0.37ms gap)
     → 91% of benchmarked FLOPS, close to ceiling
     → low priority unless gap grows at larger scale
```

### Automatic integration with training

The gap analysis should run automatically at the end of training (or on-demand during training):

```python
# In train.py, after training completes:
if is_main:
    # Prediction was computed before training started
    prediction = cost_model.project(plan, profile)

    # Measurement comes from the profiler
    measured = get_global_metrics()

    # Gap report
    gap = gap_report(prediction, measured, profile)
    gap.print()

    # Also log to wandb for experiment tracking
    wandb.summary.update(gap.to_dict())
```

For the development loop to be tight, you should be able to see the gap report *during* training, not just after. A periodic gap check (every N steps) helps catch anomalies early — e.g., comm efficiency dropping because of network congestion during a multi-tenant burst.

---

## Part 5: What the Benchmarks Teach About the Hardware

Beyond calibrating the cost model, the benchmark suite is a research tool in its own right. It answers questions that spec sheets can't:

### Compute roofline

By benchmarking GEMMs at various shapes, you build the compute side of a roofline model:

```
Achievable TFLOPS vs arithmetic intensity (ops/byte)

  TFLOPS
    900 ┤                              ●──●──●──●──●  ← compute bound
        │                           ●
    800 ┤                        ●
        │                     ●
    700 ┤                  ●
        │               ●
    600 ┤            ●
        │         ●
    400 ┤      ●                           ← memory-bandwidth bound
        │   ●
    200 ┤ ●
        │●
      0 ┤─────────────────────────────────────────────
        0    10    20    50   100   200   500  1000
                  arithmetic intensity (FLOP/byte)

  Ridge point: where compute bound meets memory bound
  GEMMs below the ridge → memory-bound → won't benefit from faster math
  GEMMs above the ridge → compute-bound → MFU ceiling is the benchmark
```

The ridge point tells you the minimum GEMM size where you're compute-bound. Below that, TP splitting hurts — you're moving GEMMs from compute-bound to memory-bound territory.

### Communication latency vs bandwidth curve

By sweeping message sizes, you characterize the full latency-bandwidth tradeoff:

```
Achieved bandwidth vs message size (NVLink all-reduce, 8 GPUs)

  GB/s
   420 ┤                                         ●──●
       │                                      ●
   400 ┤                                   ●
       │                                ●
   350 ┤                             ●
       │                          ●
   300 ┤                       ●
       │                    ●
   200 ┤                 ●
       │              ●
   100 ┤           ●
       │        ●
    50 ┤     ●
       │  ●
    10 ┤●
       │
     0 ┤─────────────────────────────────────────────
       1K   4K  16K  64K 256K  1M   4M  16M  64M 256M
                      message size

  Latency-dominated region: <64KB — bandwidth is wasted waiting for launch
  Transition region: 64KB–4MB — latency and bandwidth both matter
  Bandwidth-dominated: >4MB — approaching peak, message size doesn't matter
```

This curve tells you whether sequence parallelism (which changes message sizes from `b*s*h` to `b*s*h/tp`) puts you in a worse region of the curve. It also explains why NVSHMEM matters for small messages — GPU-initiated comms skip the launch latency.

### Cross-node vs intra-node

The same benchmarks run on different process groups reveal the topology:

```python
intra_node = benchmark_comm_sweep(group=tp_group)   # NVLink
inter_node = benchmark_comm_sweep(group=dp_group)   # IB/RoCE

# Now the cost model knows:
# - TP comm at 400 GB/s (NVLink, intra-node)
# - FSDP comm at 40 GB/s (IB, inter-node)
# - the 10× gap explains why TP should be innermost in the mesh layout
```

---

## Part 6: Model-Aware Parallelism Optimizer

The cost model design sketches a `sweep()` that loops over `tp × dp × pp` configs. But a grid sweep with generic formulas misses model-specific structure. The optimizer should use the *actual model* — its GEMM shapes, its GQA head counts, which layers are MoE, the expert count — to predict each config's cost and find the optimum.

### From generic formulas to model-specific shapes

Generic formula for Q projection compute: `2bsh²`. But with GQA (n_heads=64, n_kv_heads=8) at TP=8:

- Q per rank: `[bs, h] × [h, h/8]` → normal GEMM, ~89% of peak
- K per rank: `[bs, h] × [h, h/64]` → **tiny** GEMM, maybe ~40% of peak

The generic formula doesn't know this. The model-aware version extracts the actual shapes:

```python
def model_cost_profile(config: TransformerConfig) -> ModelCostProfile:
    """Extract everything the cost model needs from a model config."""
    layers = []
    for i in range(config.n_layers):
        if is_moe_layer(config, i):
            layers.append(MoELayerProfile(
                d_model=config.d_model,
                num_experts=config.num_experts,
                expert_d_ff=config.expert_d_ff,
                top_k=config.moe_top_k,
                n_heads=config.n_heads,
                n_kv_heads=config.n_kv_heads,
            ))
        else:
            layers.append(DenseLayerProfile(
                d_model=config.d_model,
                n_heads=config.n_heads,
                n_kv_heads=config.n_kv_heads,
                d_ff=config.d_ff,
            ))
    return ModelCostProfile(
        layers=layers,
        vocab_size=config.vocab_size,
        seq_len=config.seq_len,
    )
```

Each layer profile knows how to compute its GEMM shapes at a given TP degree:

```python
class DenseLayerProfile:
    def gemm_shapes(self, tp_size: int, batch_size: int) -> list[tuple[int,int,int,str]]:
        """Return (M, N, K, label) for every GEMM in this layer at this TP."""
        bs = batch_size * self.seq_len
        h = self.d_model
        kv_h = h * self.n_kv_heads // self.n_heads  # GQA: KV may be smaller

        return [
            (bs, h // tp_size, h,           "Q projection"),
            (bs, kv_h // tp_size, h,        "K projection"),   # small if GQA + high TP
            (bs, kv_h // tp_size, h,        "V projection"),
            (bs, h, h // tp_size,           "output projection"),
            (bs, self.d_ff // tp_size, h,   "FFN up"),
            (bs, h, self.d_ff // tp_size,   "FFN down"),
        ]

    def tp_message_bytes(self, plan: ParallelPlan, batch_size: int) -> int:
        """Bytes per TP collective for this layer."""
        bs = batch_size * self.seq_len
        # 2 reduce-scatters (or all-reduces) per layer, each over b*s*h elements
        return 2 * bs * self.d_model * 2  # × 2 for bf16 bytes
```

### The optimizer

```python
def optimal_parallelism(
    model_profile: ModelCostProfile,
    hardware_profile: HardwareProfile,
    num_gpus: int,
    batch_size: int,
    gpus_per_node: int = 8,
    constraints: ParallelismConstraints | None = None,
) -> list[RankedConfig]:
    """Find the best parallelism config for this specific model on this hardware."""

    # 1. Enumerate valid configs
    configs = enumerate_valid_configs(
        num_gpus=num_gpus,
        gpus_per_node=gpus_per_node,
        num_experts=model_profile.num_experts,
        constraints=constraints,
    )

    # 2. Project each config using model-specific shapes
    ranked = []
    for plan in configs:
        report = project_model_aware(model_profile, plan, hardware_profile, batch_size)
        ranked.append(RankedConfig(plan=plan, report=report))

    # 3. Sort by projected MFU
    ranked.sort(key=lambda r: r.report.predicted_mfu, reverse=True)
    return ranked
```

The `project_model_aware` function walks the actual layer list instead of using generic per-layer formulas:

```python
def project_model_aware(
    model: ModelCostProfile,
    plan: ParallelPlan,
    hw: HardwareProfile,
    batch_size: int,
) -> CostReport:
    """Project step time from the model's actual structure."""
    per_stage_compute = defaultdict(float)
    per_stage_tp_comm = defaultdict(float)
    total_ep_comm_ms = 0

    stage_assignments = assign_layers_to_stages(model, plan)

    for stage, layer_idx in stage_assignments.items():
        for i in layer_idx:
            layer = model.layers[i]

            # Compute: look up benchmarked throughput for each GEMM shape
            for m, n, k, label in layer.gemm_shapes(plan.tp_size, batch_size):
                tflops = hw.interpolate_matmul_tflops(m, n, k)
                per_stage_compute[stage] += (2 * m * n * k) / (tflops * 1e12) * 1000

            # TP comm: look up benchmarked bandwidth at actual message size
            tp_bytes = layer.tp_message_bytes(plan, batch_size)
            tp_bw = hw.interpolate_bandwidth("allreduce", tp_bytes)
            per_stage_tp_comm[stage] += tp_bytes / (tp_bw * 1e9) * 1000

            # EP comm: only for MoE layers
            if isinstance(layer, MoELayerProfile):
                ep_bytes = layer.ep_message_bytes(plan, batch_size)
                ep_bw = hw.interpolate_bandwidth("alltoall", ep_bytes)
                total_ep_comm_ms += ep_bytes / (ep_bw * 1e9) * 1000

    # PP bubble from the slowest stage
    slowest_stage_ms = max(
        per_stage_compute[s] + per_stage_tp_comm[s]
        for s in per_stage_compute
    )
    bubble_ms = compute_pp_bubble(plan, slowest_stage_ms)

    # FSDP comm
    fsdp_ms = compute_fsdp_comm(model, plan, hw)

    ...
```

### What this catches that generic formulas miss

**GQA + high TP:** With 8 KV heads and TP=8, each rank gets 1 KV head. The K/V GEMMs are `[bs, h/64] × [h/64, h]` — benchmarked at ~40% of peak. The optimizer might find TP=4 is better because the K/V GEMMs stay 2× larger and hit 70% of peak. Net effect: slightly more TP comm but much more compute efficiency.

**PP stage imbalance with MoE:**

```
PP stage balance (pp=4, 64 layers, MoE on layers 4,8,12,...):
  Stage 0 (layers  0–15):  3.2ms  (4 MoE layers + EP comm)
  Stage 1 (layers 16–31):  3.2ms  (4 MoE layers + EP comm)
  Stage 2 (layers 32–47):  3.2ms  (4 MoE layers + EP comm)
  Stage 3 (layers 48–63):  3.2ms  (4 MoE layers + EP comm)
  → balanced ✓ (MoE layers evenly distributed)

vs. model with MoE only in later layers:
  Stage 0 (layers  0–15):  2.1ms  (all dense)
  Stage 1 (layers 16–31):  2.1ms  (all dense)
  Stage 2 (layers 32–47):  4.3ms  (8 MoE layers)
  Stage 3 (layers 48–63):  4.3ms  (8 MoE layers)
  → 2× imbalance → 30% throughput loss from straggler
  → try: uneven split (20/20/12/12 layers per stage)
     Stage 0 (layers  0–19): 2.6ms
     Stage 1 (layers 20–39): 3.2ms (mixed dense+MoE)
     Stage 2 (layers 40–51): 3.0ms
     Stage 3 (layers 52–63): 3.0ms
     → much better balance
```

The optimizer can try different `layers_per_stage` splits and find the one that minimizes the slowest stage.

**Expert count vs EP degree:** With 64 experts and EP=32, each rank owns 2 experts. The expert GEMM is `[tokens_per_expert, h] × [h, expert_d_ff]` — but `tokens_per_expert` depends on routing decisions. The optimizer can estimate expected tokens per expert from `(batch_tokens × top_k) / num_experts` and check whether the resulting GEMM is large enough to be compute-bound.

### The full loop with the optimizer

```
1. model_profile = ModelCostProfile.from_config(transformer_config)
2. hw_profile = run_hardware_benchmarks(...)
3. ranked = optimal_parallelism(model_profile, hw_profile, num_gpus, batch_size)

   ┌─────────────────────────────────────────────────────────────┐
   │  Parallelism sweep for 70B model, 512 × H100 SXM           │
   │                                                             │
   │  tp  pp  dp  ep   MFU    bottleneck          fits?  notes  │
   │  ──────────────────────────────────────────────────────────  │
   │   4   4  32   1  52.1%  PP bubble              ✓           │
   │   8   2  32   1  49.3%  K/V GEMMs mem-bound    ✓   GQA!   │
   │   4   2  64   1  48.7%  FSDP comm              ✓           │
   │   8   4  16   1  47.2%  K/V + bubble            ✓   GQA!   │
   │   2   4  64   1  46.9%  PP bubble              ✓           │
   │   8   1  64   1    —    OOM                    ✗           │
   │                                                             │
   │  Note: tp=8 penalized because GQA KV heads (8) ÷ TP (8)   │
   │  = 1 head/rank → K/V GEMMs at 40% peak efficiency         │
   └─────────────────────────────────────────────────────────────┘

4. plan = ranked[0].plan  # tp=4, pp=4, dp=32
5. Train with plan → profiler produces measured breakdown
6. Gap report: predicted vs measured
7. If gap shows TP comm is better than predicted (overlap helps),
   maybe tp=8 moves up in the ranking → re-evaluate
```

The optimizer gives you a *starting point* calibrated to your specific model and hardware. The gap report tells you whether reality matches. If not, the gap explains why — and you either update the cost model's assumptions or change the system (add a better backend, enable overlap).

---

## Part 7: How It All Connects

```
    TransformerConfig ──► ModelCostProfile ──► Optimizer ──► Ranked plans
         │                       │                              │
         │                       │                              │ best plan
         │                       │                              ▼
         │               HardwareProfile ──────────────► Cost Model
         │              (from benchmarks)                   │
         │                                                  │ predict
         │                                                  ▼
         ├──► apply_parallelism(plan) ──► Train ──► Profiler
         │        │                                    │
         │        ▼                                    │ measure
         │    comm.py ─── measure(EventType.TP_COMM)   │
         │                                             ▼
         │                                        Gap Report
         │                                             │
         └─────────────────────────────────────────────┘
                                                   feedback:
                                              update cost model
                                              or try next plan
```

The `TransformerConfig` is the root — it defines the model that the `ModelCostProfile` extracts shapes from, the `ParallelPlan` parallelizes, and the profiler measures. The `HardwareProfile` (from benchmarks) calibrates the cost model's predictions. The gap report closes the loop by comparing prediction to reality and pointing at what to change.

---

## Directory structure

```
nanigpt/
├── benchmarks/
│   ├── hardware.py           # HardwareProfile, benchmark_matmul, benchmark_comm_sweep
│   ├── roofline.py           # Compute + comm roofline plotting
│   └── run_benchmarks.py     # CLI: uv run python -m nanigpt.benchmarks.run_benchmarks
│
├── cost_model/
│   ├── model.py              # CostModel: formulas from the parallelism strategies entry
│   ├── report.py             # CostReport: predicted breakdown
│   └── gap.py                # gap_report(): predicted vs measured comparison
│
├── profiling/
│   ├── timer.py              # StepMetrics (existing, gains comm breakdown)
│   ├── event_types.py        # EventType (existing, gains TP_COMM etc.)
│   ├── flop_counter.py       # Existing FLOP counting
│   ├── context.py            # Existing step context
│   └── torch_profiler.py     # Existing torch.profiler wrapper
│
├── parallel/
│   ├── comm.py               # Instrumented with measure(EventType.TP_COMM)
│   └── ...
└── ...
```

---

## Priorities

1. **Hardware micro-benchmarks.** Write `benchmarks/hardware.py`. Simple timed loops — benchmark matmul at model-relevant shapes, benchmark all-reduce at a range of message sizes. Output a `HardwareProfile`. This is immediately useful even without the cost model — you learn things about your hardware.

2. **Comm-aware profiling.** Add `TP_COMM`, `EP_COMM`, `FSDP_COMM` event types. Instrument `comm.py` with `measure()` calls. Now every training run breaks out compute vs comm.

3. **Cost model implementation.** The formulas are already derived in the [parallelism strategies entry](2026-02-24-parallelism-strategies.md) and the [cost model design](2026-02-25-cost-model-design.md). Make them executable, calibrated by `HardwareProfile`.

4. **Gap report.** Compare `CostReport` vs `StepMetrics`. Rank opportunities. This is mostly formatting once 1–3 exist.

---

## Open questions

1. **Should benchmarks run at training startup?** The full suite takes maybe 60 seconds. Running it once at startup means the cost model is always calibrated to the current hardware. Alternatively, cache profiles to disk keyed by GPU type + topology hash.

2. **How to handle overlapped comm and compute?** The analytical model assumes serialized (pessimistic). The profiler measures wall-clock per category. When comm overlaps with compute, the sum of categories exceeds wall time. The gap report needs to account for this — "predicted serial: 7.2ms, predicted with perfect overlap: 5.0ms, measured: 6.1ms → overlap efficiency: 50%."

3. **Memory-bound ops.** LayerNorm, softmax, GELU, residual adds are memory-bound, not compute-bound. They're FLOP-negligible but can eat wall-clock time. The cost model doesn't model them. Should it? Or should the gap report just identify "compute took longer than GEMM-only prediction → memory-bound ops account for the delta"?

4. **Scaling behavior.** Does the predict-measure-explain loop work at larger scale (hundreds of GPUs)? Network contention, stragglers, and load imbalance become factors. The benchmark suite measures ideal conditions — single-job, dedicated hardware. Multi-tenant contention is a runtime effect that benchmarks can't predict.
