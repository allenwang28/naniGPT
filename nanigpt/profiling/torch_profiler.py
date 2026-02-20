"""PyTorch Profiler integration with terminal-first output and optional wandb trace upload.

This Uses the step-end callback system to drive profiling automatically.

Usage:

    from nanigpt.profiling.torch_profiler import ProfilerConfig, init_profiler

    init_profiler(ProfilerConfig(enabled=True, start_step=10, end_step=12))

    for step in range(1, NUM_STEPS + 1):
        with step_context(step):
            ...  # training code

The profiler starts, warms up, records, and cleans up based on step numbers.
"""

import gzip
import logging
import shutil
import tempfile
import threading
from dataclasses import asdict, dataclass
from pathlib import Path

import torch.profiler

from nanigpt.profiling.context import get_step, register_step_end, unregister_step_end

log = logging.getLogger(__name__)


@dataclass
class ProfilerConfig:
    """Controls the PyTorch profiler window."""

    enabled: bool = True
    start_step: int = 10
    end_step: int = 12
    warmup_steps: int = 1
    top_n: int = 15
    export_trace: bool = True
    record_shapes: bool = True
    with_stack: bool = False
    with_flops: bool = True


def _print_kernel_table(averages: list, top_n: int) -> None:
    """Format key_averages() as a terminal table sorted by self GPU time."""
    # Filter to entries with non-zero self GPU time, sort descending
    gpu_entries = [e for e in averages if e.self_device_time_total > 0]
    gpu_entries.sort(key=lambda e: e.self_device_time_total, reverse=True)

    if not gpu_entries:
        log.info("No GPU kernels recorded.")
        return

    total_gpu_us = sum(e.self_device_time_total for e in gpu_entries)
    total_cpu_us = sum(e.self_cpu_time_total for e in averages)

    top = gpu_entries[:top_n]
    rest = gpu_entries[top_n:]

    # Header
    header = f"{'Kernel':<60} {'GPU%':>5} {'GPU ms':>8} {'CPU ms':>8} {'Calls':>6}"
    if any(e.flops for e in top):
        header += f" {'GFLOPS':>8}"
    log.info(header)
    log.info("-" * len(header))

    for e in top:
        gpu_pct = 100.0 * e.self_device_time_total / total_gpu_us if total_gpu_us else 0
        gpu_ms = e.self_device_time_total / 1000.0
        cpu_ms = e.self_cpu_time_total / 1000.0
        name = e.key[:60]
        line = f"{name:<60} {gpu_pct:5.1f} {gpu_ms:8.2f} {cpu_ms:8.2f} {e.count:6d}"
        if e.flops:
            gflops = e.flops / 1e9
            line += f" {gflops:8.1f}"
        log.info(line)

    # Summary for remaining ops
    if rest:
        rest_gpu_us = sum(e.self_device_time_total for e in rest)
        rest_pct = 100.0 * rest_gpu_us / total_gpu_us if total_gpu_us else 0
        log.info(f"\033[2m... {len(rest)} more ops ({rest_pct:.1f}% of GPU time)\033[0m")

    # CPU/GPU ratio as launch overhead indicator
    if total_gpu_us > 0:
        ratio = total_cpu_us / total_gpu_us
        log.info(f"\033[2mCPU/GPU time ratio: {ratio:.2f}x\033[0m")


def _export_and_upload_wandb_trace(
    prof: torch.profiler.profile, start_step: int, end_step: int
) -> None:
    """Export Chrome trace synchronously, then compress and upload to wandb in background.

    export_chrome_trace() must run before the profiler exits, so it happens inline.
    The gzip compression and wandb artifact upload run in a daemon thread to avoid
    blocking training.
    """
    try:
        import wandb

        if wandb.run is None:
            log.warning("No active wandb run — skipping trace upload.")
            return
    except Exception:
        log.warning("wandb not available — skipping trace upload.")
        return

    # Export synchronously (must happen before profiler.__exit__)
    tmpdir = tempfile.mkdtemp(prefix="nanigpt_trace_")
    trace_path = Path(tmpdir) / f"trace_steps_{start_step}_{end_step}.json"
    prof.export_chrome_trace(str(trace_path))

    def _compress_and_upload() -> None:
        try:
            import wandb

            gz_path = trace_path.with_suffix(".json.gz")
            with open(trace_path, "rb") as f_in, gzip.open(gz_path, "wb") as f_out:
                f_out.write(f_in.read())

            artifact = wandb.Artifact(
                name=f"profile-trace-steps-{start_step}-{end_step}",
                type="profile",
            )
            artifact.add_file(str(gz_path))
            wandb.log_artifact(artifact)
            log.debug(f"Uploaded trace artifact for steps {start_step}-{end_step}")
        except Exception:
            log.warning("Failed to upload wandb trace artifact.", exc_info=True)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    thread = threading.Thread(target=_compress_and_upload, daemon=True)
    thread.start()


_pending_results: list[tuple[list, int]] = []


def print_profiler_summary() -> None:
    """Print any stashed profiler results. Call at end of training."""
    for averages, top_n in _pending_results:
        _print_kernel_table(averages, top_n)
    _pending_results.clear()


def init_profiler(config: ProfilerConfig) -> None:
    """Register a step-end callback that manages the torch profiler lifecycle.

    The callback watches get_step() and:
    - Enters the profiler at the right step
    - Calls profiler.step() each step while active
    - Tears down and unregisters itself when the window completes
    """
    if not config.enabled:
        return

    active_steps = config.end_step - config.start_step

    schedule = torch.profiler.schedule(
        skip_first=config.start_step - 1,
        wait=0,
        warmup=config.warmup_steps,
        active=active_steps,
        repeat=1,
    )

    # Total steps the torch profiler needs to see before it's done:
    # skip_first + warmup + active
    total_profiler_steps = (config.start_step - 1) + config.warmup_steps + active_steps

    def on_trace_ready(prof: torch.profiler.profile) -> None:
        _pending_results.append((prof.key_averages(), config.top_n))
        if config.export_trace:
            _export_and_upload_wandb_trace(prof, config.start_step, config.end_step)

    profiler = torch.profiler.profile(
        schedule=schedule,
        on_trace_ready=on_trace_ready,
        record_shapes=config.record_shapes,
        with_stack=config.with_stack,
        with_flops=config.with_flops,
        acc_events=True,
    )

    # Mutable state for the callback
    state = {"entered": False, "steps_seen": 0}

    def _step_callback() -> None:
        step = get_step()
        if step is None:
            return

        if not state["entered"]:
            profiler.__enter__()
            state["entered"] = True

        profiler.step()
        state["steps_seen"] += 1

        if state["steps_seen"] >= total_profiler_steps:
            profiler.__exit__(None, None, None)
            unregister_step_end(_step_callback)

    register_step_end(_step_callback)


def profiler_config_dict(config: ProfilerConfig) -> dict:
    """Return config as a dict for wandb logging."""
    return asdict(config)
