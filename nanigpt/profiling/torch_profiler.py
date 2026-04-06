"""PyTorch Profiler integration with multi-window support.

Provides a callable TorchProfiler that plugs into the step-end callback system.
Supports multiple profiling windows to compare behavior across different phases
of training (e.g., before and after eval).

Usage:

    from nanigpt.profiling.context import register_step_end
    from nanigpt.profiling.torch_profiler import TorchProfiler

    # Single window
    profiler = TorchProfiler(TorchProfiler.Config(windows="10-12"))
    register_step_end(profiler)

    # Multiple windows (e.g., before/after eval at step 50)
    profiler = TorchProfiler(TorchProfiler.Config(windows="40-50,55-60"))
    register_step_end(profiler)

    for step in range(1, NUM_STEPS + 1):
        with step_context(step):
            ...  # training code

    profiler.print_summary()       # kernel table per window
    profiler.serve_perfetto()      # serve traces and print Perfetto links
"""

import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import torch.profiler

from nanigpt.configurable import Configurable
from nanigpt.profiling.context import get_step, unregister_step_end
from nanigpt.profiling.perfetto import export_trace, serve_traces
from nanigpt.profiling.trace_analysis import _is_comm_kernel, compute_exposed_comm

log = logging.getLogger(__name__)


def _parse_windows(windows_str: str) -> list[tuple[int, int]]:
    """Parse a comma-separated list of step ranges into (start, end) tuples.

    Examples:
        "10-12"       -> [(10, 12)]
        "40-50,55-60" -> [(40, 50), (55, 60)]
    """
    result = []
    for part in windows_str.split(","):
        part = part.strip()
        if "-" not in part:
            raise ValueError(
                f"Invalid window format '{part}'. Expected 'start-end' (e.g. '10-12')."
            )
        start_str, end_str = part.split("-", 1)
        start, end = int(start_str), int(end_str)
        if end <= start:
            raise ValueError(f"Window end ({end}) must be > start ({start}) in '{part}'.")
        result.append((start, end))

    # Sort by start step
    result.sort(key=lambda w: w[0])

    # Check for overlaps (including warmup — windows need enough gap)
    for i in range(len(result) - 1):
        if result[i][1] > result[i + 1][0]:
            raise ValueError(
                f"Windows overlap: {result[i][0]}-{result[i][1]} "
                f"and {result[i + 1][0]}-{result[i + 1][1]}."
            )

    return result


def _print_comm_overhead(exposed: dict, steps_label: str = "") -> None:
    """Format the communication overhead section from exposed comm analysis."""
    total = exposed["total_comm_ms"]
    blocking = exposed["exposed_comm_ms"]
    hidden = exposed["hidden_comm_ms"]
    n = exposed["num_nccl_kernels"]
    blocking_pct = 100.0 * blocking / total if total > 0 else 0
    hidden_pct = 100.0 * hidden / total if total > 0 else 0
    if steps_label:
        header = f"Communication overhead ({steps_label}):"
    else:
        header = "Communication overhead:"
    log.info(header)
    log.info(f"  Total NCCL time:          {total:7.2f} ms ({n} kernels)")
    log.info(f"  Blocking compute:         {blocking:7.2f} ms ({blocking_pct:.0f}%)")
    log.info(f"  Hidden behind compute:    {hidden:7.2f} ms ({hidden_pct:.0f}%)")
    log.info("")


def _print_kernel_table(averages: list, top_n: int, label: str = "") -> None:
    """Format key_averages() as a compact kernel table sorted by self GPU time."""
    gpu_entries = [e for e in averages if e.self_device_time_total > 0]
    gpu_entries.sort(key=lambda e: e.self_device_time_total, reverse=True)

    if label:
        log.info(label)

    if not gpu_entries:
        log.info("No GPU kernels recorded.")
        return

    total_gpu_us = sum(e.self_device_time_total for e in gpu_entries)

    top = gpu_entries[:top_n]
    rest = gpu_entries[top_n:]

    for e in top:
        gpu_pct = 100.0 * e.self_device_time_total / total_gpu_us if total_gpu_us else 0
        gpu_ms = e.self_device_time_total / 1000.0
        name = e.key[:40]
        line = f"  {name:<40s} {gpu_ms:7.2f} ms {gpu_pct:5.1f}%"
        if e.flops:
            gflops = e.flops / 1e9
            line += f"  {gflops:.0f} GFLOPS"
        log.info(line)

    if rest:
        rest_gpu_us = sum(e.self_device_time_total for e in rest)
        rest_pct = 100.0 * rest_gpu_us / total_gpu_us if total_gpu_us else 0
        log.info(f"  ... {len(rest)} more ops ({rest_pct:.1f}% of GPU time)")

    # Compute vs communication breakdown
    comm_us = sum(e.self_device_time_total for e in gpu_entries if _is_comm_kernel(e.key))
    compute_us = total_gpu_us - comm_us
    if comm_us > 0:
        comm_pct = 100.0 * comm_us / total_gpu_us
        compute_pct = 100.0 * compute_us / total_gpu_us
        log.info(
            f"  Compute: {compute_us / 1000:.2f} ms ({compute_pct:.1f}%)"
            f" | Communication: {comm_us / 1000:.2f} ms ({comm_pct:.1f}%)"
        )


@dataclass
class _WindowState:
    """Tracks the lifecycle of a single profiling window."""

    start_step: int
    end_step: int
    warmup_steps: int
    profiler: torch.profiler.profile
    entered: bool = False
    done: bool = False
    steps_called: int = 0

    @property
    def enter_step(self) -> int:
        """The step at which to enter the profiler (warmup begins)."""
        return self.start_step - self.warmup_steps

    @property
    def total_profiler_steps(self) -> int:
        """Total .step() calls needed: warmup + active."""
        return self.warmup_steps + (self.end_step - self.start_step)


class TorchProfiler(Configurable):
    """Step-end callback that manages the torch profiler lifecycle.

    Callable that conforms to the step-end callback interface. Manages one
    or more profiling windows, each with its own torch.profiler.profile
    instance. Windows are activated and torn down as training steps progress.

    Results are stashed and printed via print_summary() at end of training.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        """Controls the PyTorch profiler windows.

        Specify one or more step ranges as a comma-separated string:
            windows="10-12"         # single window, steps 10-12
            windows="40-50,55-60"   # two windows for comparison
        """

        enabled: bool = True
        windows: str = "10-12"
        warmup_steps: int = 1
        top_n: int = 15
        export_trace: bool = True
        trace_dir: str = "traces"
        perfetto: bool = False
        record_shapes: bool = True
        with_stack: bool = False
        with_flops: bool = True

        def __post_init__(self):
            # Validate by parsing — raises ValueError on bad input
            _parse_windows(self.windows)

    def __init__(self, config: Config):
        self._config = config
        self._results: list[tuple[list, int, str, dict | None]] = []
        self._trace_paths: list[Path] = []

        if not config.enabled:
            self._windows: list[_WindowState] = []
            return

        parsed = _parse_windows(config.windows)
        self._windows = []
        trace_dir = Path(config.trace_dir)

        for start, end in parsed:
            active_steps = end - start

            schedule = torch.profiler.schedule(
                skip_first=0,
                wait=0,
                warmup=config.warmup_steps,
                active=active_steps,
                repeat=1,
            )

            def make_on_trace_ready(ws: int, we: int, td: Path):
                def on_trace_ready(p: torch.profiler.profile) -> None:
                    label = f"Top kernels (steps {ws}-{we}):"
                    exposed = compute_exposed_comm(p)
                    self._results.append((p.key_averages(), config.top_n, label, exposed))
                    if config.export_trace:
                        gz_path = export_trace(p, ws, we, td)
                        self._trace_paths.append(gz_path)

                return on_trace_ready

            prof = torch.profiler.profile(
                schedule=schedule,
                on_trace_ready=make_on_trace_ready(start, end, trace_dir),
                record_shapes=config.record_shapes,
                with_stack=config.with_stack,
                with_flops=config.with_flops,
                acc_events=True,
            )

            self._windows.append(
                _WindowState(
                    start_step=start,
                    end_step=end,
                    warmup_steps=config.warmup_steps,
                    profiler=prof,
                )
            )

    def __call__(self) -> None:
        if not self._windows:
            return

        step = get_step()
        if step is None:
            return

        all_done = True
        for w in self._windows:
            if w.done:
                continue

            if step < w.enter_step:
                all_done = False
                continue

            # Enter the profiler context on the first relevant step
            if not w.entered:
                w.profiler.__enter__()
                w.entered = True

            w.profiler.step()
            w.steps_called += 1

            if w.steps_called >= w.total_profiler_steps:
                w.profiler.__exit__(None, None, None)
                w.done = True
            else:
                all_done = False

        if all_done:
            unregister_step_end(self)

    def print_summary(self) -> None:
        """Print exposed comm analysis and kernel table for each window."""
        for averages, top_n, label, exposed in self._results:
            if exposed is not None:
                # Extract "steps X-Y" from label like "Top kernels (steps 10-12):"
                m = re.search(r"steps (\d+-\d+)", label)
                steps_label = f"steps {m.group(1)}" if m else ""
                _print_comm_overhead(exposed, steps_label)
            _print_kernel_table(averages, top_n, label=label)

    def serve_perfetto(self) -> None:
        """Serve saved traces via local HTTP and print Perfetto UI links.

        Blocks until all traces have been fetched or the user presses Ctrl+C.
        Only runs if perfetto=True in config and traces were exported.
        """
        if self._config.perfetto and self._trace_paths:
            serve_traces(self._trace_paths)

    def config_dict(self) -> dict:
        """Return config as a dict for wandb logging."""
        return asdict(self._config)
