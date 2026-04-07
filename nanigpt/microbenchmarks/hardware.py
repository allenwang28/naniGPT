"""Hardware micro-benchmarks: GEMM throughput and collective bandwidth.

Measures actual hardware capabilities at the shapes and sizes that matter
for training. Results are reported as utilization ratios (% of peak) so
they transfer across GPU generations.

Two benchmark types:
1. GEMM throughput at model-relevant matrix shapes (from TransformerConfig + ParallelPlan)
2. Collective bandwidth at a range of message sizes (bandwidth-vs-size curve)

Usage:
    uv run python -m nanigpt.microbenchmarks.hardware --config small-synthetic
    uv run python -m nanigpt.microbenchmarks.hardware --config small-synthetic-fsdp
"""

import json
import logging
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.distributed as dist

from nanigpt.config import ParallelConfig
from nanigpt.distributed.plan import ParallelPlan
from nanigpt.env import MASTER_ADDR, MASTER_PORT, enforce_defaults
from nanigpt.models.dense_transformer import TransformerConfig
from nanigpt.profiling.flop_counter import GPU_PEAK_TFLOPS, detect_gpu

log = logging.getLogger(__name__)

# Default message sizes for comm sweep: 1KB to 256MB in powers of 2
DEFAULT_COMM_SIZES = [2**i for i in range(10, 29)]

# Algorithmic bytes multiplier for each collective (ring algorithm).
# all-reduce: 2*(N-1)/N, all-gather: (N-1)/N, reduce-scatter: (N-1)/N
_ALGO_BYTES_FACTOR = {
    "allreduce": lambda n: 2 * (n - 1) / n,
    "allgather": lambda n: (n - 1) / n,
    "reduce_scatter": lambda n: (n - 1) / n,
}


def _run_allreduce(tensor: torch.Tensor, group: dist.ProcessGroup) -> None:
    dist.all_reduce(tensor, group=group)


def _run_allgather(tensor: torch.Tensor, group: dist.ProcessGroup) -> None:
    world = dist.get_world_size(group)
    output = torch.empty(tensor.numel() * world, dtype=tensor.dtype, device=tensor.device)
    dist.all_gather_into_tensor(output, tensor, group=group)


def _run_reduce_scatter(tensor: torch.Tensor, group: dist.ProcessGroup) -> None:
    world = dist.get_world_size(group)
    output = torch.empty(tensor.numel() // world, dtype=tensor.dtype, device=tensor.device)
    dist.reduce_scatter_tensor(output, tensor, group=group)


_COLLECTIVE_FN = {
    "allreduce": _run_allreduce,
    "allgather": _run_allgather,
    "reduce_scatter": _run_reduce_scatter,
}


@dataclass
class BenchmarkResult:
    """Result from a single micro-benchmark."""

    achieved: float  # TFLOPS or GB/s
    peak: float  # theoretical peak
    efficiency: float  # achieved / peak
    median_ms: float  # median trial time


@dataclass
class GemmResult(BenchmarkResult):
    """GEMM benchmark result with shape metadata."""

    shape: tuple[int, int, int]  # (M, N, K)
    label: str  # e.g. "Q/K/V projection"


@dataclass
class CommResult(BenchmarkResult):
    """Communication benchmark result with collective metadata."""

    size_bytes: int
    collective: str  # "allreduce", "allgather", etc.


def benchmark_matmul(
    m: int,
    n: int,
    k: int,
    label: str = "",
    dtype: torch.dtype = torch.bfloat16,
    warmup: int = 50,
    trials: int = 200,
) -> GemmResult:
    """Measure achieved TFLOPS for a specific GEMM shape."""
    a = torch.randn(m, k, dtype=dtype, device="cuda")
    b = torch.randn(k, n, dtype=dtype, device="cuda")

    for _ in range(warmup):
        torch.mm(a, b)
    torch.cuda.synchronize()

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
    gpu_name = detect_gpu()
    peak_tflops = GPU_PEAK_TFLOPS.get(gpu_name, 0.0)
    efficiency = achieved_tflops / peak_tflops if peak_tflops > 0 else 0.0

    return GemmResult(
        achieved=achieved_tflops,
        peak=peak_tflops,
        efficiency=efficiency,
        median_ms=median_ms,
        shape=(m, n, k),
        label=label,
    )


def model_gemm_shapes(
    config: TransformerConfig,
    plan: ParallelPlan,
    batch_size: int,
) -> list[tuple[int, int, int, str]]:
    """Extract the GEMM shapes that actually run during training.

    Returns (M, N, K, label) tuples for each unique GEMM in one transformer
    layer, accounting for tensor parallel splitting.
    """
    bs = batch_size * config.max_seq_len
    h = config.d_model
    tp = plan.tp_size
    ffn = config.d_ff

    if tp > 1:
        return [
            (bs, h // tp, h, "Q/K/V projection (col-parallel)"),
            (bs, h, h // tp, "Output projection (row-parallel)"),
            (bs, ffn // tp, h, "FFN up (col-parallel)"),
            (bs, h, ffn // tp, "FFN down (row-parallel)"),
        ]
    return [
        (bs, h, h, "Q/K/V projection"),
        (bs, h, h, "Output projection"),
        (bs, ffn, h, "FFN up"),
        (bs, h, ffn, "FFN down"),
    ]


def benchmark_model_gemms(
    config: TransformerConfig,
    plan: ParallelPlan,
    batch_size: int,
) -> list[GemmResult]:
    """Benchmark all GEMM shapes from a model config."""
    shapes = model_gemm_shapes(config, plan, batch_size)
    results = []
    for m, n, k, label in shapes:
        result = benchmark_matmul(m, n, k, label=label)
        results.append(result)
    return results


def benchmark_collective(
    collective: str,
    size_bytes: int,
    group: dist.ProcessGroup,
    warmup: int = 20,
    trials: int = 100,
) -> CommResult:
    """Measure achieved bandwidth for a collective at a given message size."""
    world_size = dist.get_world_size(group)
    numel = size_bytes // 2  # bf16 = 2 bytes per element

    # For reduce_scatter, input must be divisible by world_size
    if collective == "reduce_scatter":
        numel = (numel // world_size) * world_size
        size_bytes = numel * 2

    # For allgather, the input is size_bytes/world_size per rank
    if collective == "allgather":
        per_rank = numel // world_size
        numel = per_rank
        tensor = torch.randn(numel, dtype=torch.bfloat16, device="cuda")
    else:
        tensor = torch.randn(numel, dtype=torch.bfloat16, device="cuda")

    fn = _COLLECTIVE_FN[collective]

    for _ in range(warmup):
        fn(tensor, group)
    torch.cuda.synchronize()
    dist.barrier(group=group)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(trials):
        start.record()
        fn(tensor, group)
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))

    algo_factor = _ALGO_BYTES_FACTOR[collective](world_size)
    algo_bytes = algo_factor * size_bytes
    median_ms = sorted(samples)[len(samples) // 2]
    achieved_bw_gbs = algo_bytes / (median_ms / 1000) / 1e9

    # Use NVLink spec as peak — could be refined per topology
    peak_bw_gbs = 450.0  # GB/s, conservative NVLink estimate

    return CommResult(
        achieved=achieved_bw_gbs,
        peak=peak_bw_gbs,
        efficiency=achieved_bw_gbs / peak_bw_gbs if peak_bw_gbs > 0 else 0.0,
        median_ms=median_ms,
        size_bytes=size_bytes,
        collective=collective,
    )


def benchmark_comm_sweep(
    collective: str,
    group: dist.ProcessGroup,
    sizes: list[int] | None = None,
) -> list[CommResult]:
    """Sweep message sizes to build a bandwidth curve."""
    if sizes is None:
        sizes = DEFAULT_COMM_SIZES
    return [benchmark_collective(collective, s, group) for s in sizes]


def _format_bytes(n: int) -> str:
    """Format byte count as human-readable string."""
    if n >= 1 << 20:
        return f"{n / (1 << 20):.0f} MB"
    if n >= 1 << 10:
        return f"{n / (1 << 10):.0f} KB"
    return f"{n} B"


def print_gemm_table(results: list[GemmResult]) -> None:
    """Print GEMM benchmark results as a formatted table."""
    gpu_name = detect_gpu()
    peak = GPU_PEAK_TFLOPS.get(gpu_name, "?")
    log.info(f"GEMM benchmarks ({gpu_name}, bf16, peak={peak} TFLOPS):")
    log.info("")
    for r in results:
        m, n, k = r.shape
        log.info(
            f"  {r.label:<35s} ({m}, {n}, {k})"
            f"  {r.achieved:7.1f} TFLOPS  {r.efficiency * 100:5.1f}%"
            f"  [{r.median_ms:.3f} ms]"
        )
    log.info("")


def print_comm_table(results: list[CommResult], world_size: int) -> None:
    """Print communication benchmark results as a formatted table."""
    if not results:
        return
    collective = results[0].collective
    log.info(f"{collective} bandwidth sweep ({world_size} GPUs):")
    log.info(f"  {'Size':<10s} {'BW (GB/s)':>10s} {'Efficiency':>12s} {'Latency':>10s}")
    log.info(f"  {'─' * 10} {'─' * 10} {'─' * 12} {'─' * 10}")
    for r in results:
        log.info(
            f"  {_format_bytes(r.size_bytes):<10s}"
            f" {r.achieved:10.1f}"
            f" {r.efficiency * 100:11.1f}%"
            f" {r.median_ms:9.3f} ms"
        )
    log.info("")


def run_compute_benchmarks(
    config: TransformerConfig,
    plan: ParallelPlan,
    batch_size: int,
) -> list[GemmResult]:
    """Run GEMM benchmarks for model-relevant shapes and print results."""
    results = benchmark_model_gemms(config, plan, batch_size)
    print_gemm_table(results)
    return results


def run_comm_benchmarks(
    group: dist.ProcessGroup,
    collectives: list[str] | None = None,
) -> dict[str, list[CommResult]]:
    """Run comm bandwidth sweeps and print results."""
    if collectives is None:
        collectives = ["allreduce", "allgather", "reduce_scatter"]

    world_size = dist.get_world_size(group)
    all_results = {}
    for collective in collectives:
        results = benchmark_comm_sweep(collective, group)
        print_comm_table(results, world_size)
        all_results[collective] = results
    return all_results


def save_profile(
    gemm_results: list[GemmResult],
    comm_results: dict[str, list[CommResult]] | None = None,
    output_dir: str = "benchmarks",
) -> Path:
    """Save benchmark results to a JSON file.

    File is named by GPU and world size, e.g. benchmarks/A100-SXM_2gpu.json.
    Overwrites any existing file for the same configuration.
    """
    gpu_name = detect_gpu()
    world_size = 1
    if comm_results:
        first_collective = next(iter(comm_results.values()))
        if first_collective:
            world_size = dist.get_world_size()

    filename = f"{gpu_name}_{world_size}gpu.json"
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename

    profile = {
        "gpu_name": gpu_name,
        "world_size": world_size,
        "gemm": [asdict(r) for r in gemm_results],
    }
    if comm_results:
        profile["comm"] = {
            collective: [asdict(r) for r in results] for collective, results in comm_results.items()
        }

    out_path.write_text(json.dumps(profile, indent=2) + "\n")
    log.info(f"Saved hardware profile to {out_path}")
    return out_path


def run_all(
    config: TransformerConfig,
    plan: ParallelPlan,
    batch_size: int,
    group: dist.ProcessGroup | None = None,
    output_dir: str = "benchmarks",
) -> None:
    """Run all benchmarks, print results, and save to JSON."""
    gemm_results = run_compute_benchmarks(config, plan, batch_size)
    comm_results = None
    if group is not None:
        comm_results = run_comm_benchmarks(group)
    save_profile(gemm_results, comm_results, output_dir=output_dir)


# ---- CLI ----


def _parse_config():
    """Parse --config from CLI and return the TrainConfig."""
    args = sys.argv[1:]
    config_name = None
    i = 0
    while i < len(args):
        if args[i] == "--config" and i + 1 < len(args):
            config_name = args[i + 1]
            break
        i += 1

    if config_name is None:
        print("Usage: uv run python -m nanigpt.microbenchmarks.hardware --config <name>")
        print()
        from nanigpt.configs.registry import REGISTRY

        print(f"Available configs: {', '.join(sorted(REGISTRY.keys()))}")
        sys.exit(1)

    from nanigpt.configs.registry import REGISTRY

    if config_name not in REGISTRY:
        available = ", ".join(sorted(REGISTRY.keys()))
        print(f"Unknown config '{config_name}'. Available: {available}")
        sys.exit(1)

    return REGISTRY[config_name]()


def _plan_from_parallel(parallel: ParallelConfig) -> ParallelPlan:
    """Extract a ParallelPlan from a ParallelConfig."""
    return ParallelPlan(
        dp_replicate=parallel.dp_replicate,
        dp_shard=parallel.dp_shard,
        tp_size=parallel.tp_size,
    )


def _benchmark_worker(
    rank: int,
    world_size: int,
    model_config: TransformerConfig,
    plan: ParallelPlan,
    batch_size: int,
    output_dir: str = "benchmarks",
) -> None:
    """Run benchmarks for a single rank."""
    from nanigpt.distributed import cleanup_distributed, init_distributed

    logging.basicConfig(
        level=logging.INFO,
        format=f"\033[2m%(asctime)s [rank {rank}] %(levelname)s\033[0m %(message)s",
        datefmt="%H:%M:%S",
    )

    enforce_defaults()
    init_distributed(rank, world_size)

    group = torch.distributed.group.WORLD

    if rank == 0:
        run_all(model_config, plan, batch_size, group=group, output_dir=output_dir)
    else:
        run_comm_benchmarks(group)

    cleanup_distributed()


def main() -> None:
    config = _parse_config()
    model_config = config.model
    plan = _plan_from_parallel(config.parallel)
    batch_size = config.training.batch_size

    logging.basicConfig(
        level=logging.INFO,
        format="\033[2m%(asctime)s %(levelname)s\033[0m %(message)s",
        datefmt="%H:%M:%S",
    )

    # Resolve to absolute path before Ray workers change CWD
    output_dir = str(Path("benchmarks").resolve())

    log.info(f"Config: model d_model={model_config.d_model}, d_ff={model_config.d_ff}")
    log.info(f"Plan: tp={plan.tp_size}, dp_shard={plan.dp_shard}, dp_replicate={plan.dp_replicate}")
    log.info(f"Batch size: {batch_size}")

    if config.parallel.num_workers == 1:
        enforce_defaults()
        run_all(model_config, plan, batch_size, output_dir=output_dir)
    else:
        import ray

        ray.init(ignore_reinit_error=True)

        master_addr = MASTER_ADDR.get_value()
        master_port = MASTER_PORT.get_value()

        @ray.remote(num_gpus=1)
        class BenchmarkWorker:
            def __init__(self, rank, world_size, addr, port):
                self.rank = rank
                self.world_size = world_size
                MASTER_ADDR.set_value(addr)
                MASTER_PORT.set_value(port)

            def run(self, model_config, plan, batch_size, output_dir):
                _benchmark_worker(
                    self.rank, self.world_size, model_config, plan, batch_size, output_dir
                )

        num_workers = config.parallel.num_workers
        log.info(f"Launching {num_workers} workers via Ray")

        workers = [
            BenchmarkWorker.remote(rank, num_workers, master_addr, master_port)
            for rank in range(num_workers)
        ]
        futures = [w.run.remote(model_config, plan, batch_size, output_dir) for w in workers]
        ray.get(futures)
        ray.shutdown()


if __name__ == "__main__":
    main()
