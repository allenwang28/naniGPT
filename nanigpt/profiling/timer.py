"""GPU-accurate timing for training regions.

Architecture:

    measure (context manager)
      → CUDATimer (records CUDA events on a stream)
        → background thread (polls event.query() for completion)
      → StepMetrics (collects resolved timings per step)

The key constraint: CUDA event timing requires both events to have completed
before calling elapsed_time(), but calling event.synchronize() blocks the CPU
and can stall the GPU pipeline. We avoid this by polling event.query() in a
background daemon thread. By the time StepMetrics.step() runs at the step
boundary, the polling threads have almost always finished — so joining them
is effectively free.

Data flow for a single timed region:

    1. measure.__enter__  → CUDATimer.record_start() inserts a start marker
                            into the GPU stream
    2. (GPU work happens)
    3. measure.__exit__   → CUDATimer.record_end() inserts an end marker and
                            spawns a daemon thread that polls end_event.query()
                          → StepMetrics.record_pending() stashes the timer
    4. step_context exit  → StepMetrics.step() joins each timer's thread,
                            reads elapsed_ms(), and moves timings into history

The stream parameter on measure/CUDATimer defaults to the current stream but
can be overridden for timing work on non-default streams.
"""

import logging
import threading
import time
from collections import defaultdict
from contextlib import ContextDecorator

import torch

from nanigpt.profiling.context import get_step
from nanigpt.profiling.event_types import EventType

logger = logging.getLogger(__name__)

# Polling interval for background event completion threads
_POLL_INTERVAL_S = 0.001


class CUDATimer:
    """Thin wrapper around CUDA events for timing GPU operations.

    Accepts an optional stream (defaults to the current stream at creation time).
    On record_end(), spawns a background thread that polls event.query() for
    completion. elapsed_ms() joins the thread and returns the result — never
    blocking the main thread unless the GPU is still running.
    """

    def __init__(self, stream: torch.cuda.Stream | None = None):
        self.stream = stream or torch.cuda.current_stream()
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)
        self._elapsed: float | None = None
        self._thread: threading.Thread | None = None

    def record_start(self) -> None:
        self.start_event.record(self.stream)

    def record_end(self) -> None:
        """Record the end event and spawn a background thread to poll for completion."""
        self.end_event.record(self.stream)
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def _poll(self) -> None:
        while not self.end_event.query():
            time.sleep(_POLL_INTERVAL_S)
        self._elapsed = self.start_event.elapsed_time(self.end_event)

    def elapsed_ms(self) -> float:
        """Returns elapsed time in milliseconds.

        Joins the background polling thread if it hasn't finished yet.
        By step boundary this is typically a no-op.
        """
        if self._thread is not None:
            self._thread.join()
        if self._elapsed is None:
            raise RuntimeError("CUDATimer: call record_start/record_end before elapsed_ms")
        return self._elapsed


class StepMetrics:
    """Accumulates region timings per step and provides reporting.

    Regions are stashed as pending (event_type, CUDATimer) pairs during the step.
    On step(), pending timers are resolved (joining their background threads)
    and timings are moved into history.
    """

    def __init__(self):
        self._pending: list[tuple[EventType, CUDATimer]] = []
        self._current: dict[str, float] = {}
        self._history: dict[str, list[float]] = defaultdict(list)

    def record_pending(self, event_type: EventType, timer: CUDATimer) -> None:
        """Stash a timer for resolution at step boundary."""
        self._pending.append((event_type, timer))

    def record(self, event_type: EventType, elapsed_ms: float) -> None:
        """Record a region's elapsed time directly (for testing or CPU timing)."""
        key = event_type.value
        self._current[key] = self._current.get(key, 0.0) + elapsed_ms

    def step(self) -> None:
        """Finalize the current step: resolve pending timers and move into history."""
        for event_type, timer in self._pending:
            elapsed = timer.elapsed_ms()
            key = event_type.value
            self._current[key] = self._current.get(key, 0.0) + elapsed
        self._pending.clear()

        for key, ms in self._current.items():
            self._history[key].append(ms)
        self._current.clear()

    def last_step_ms(self) -> dict[str, float]:
        """Return timings from the most recent step."""
        return {name: times[-1] for name, times in self._history.items() if times}

    def mean_ms(self) -> dict[str, float]:
        """Return mean timings across all recorded steps."""
        return {name: sum(times) / len(times) for name, times in self._history.items() if times}

    def report(self, last_n: int = 1) -> str:
        """Format a profiling report for the last N steps."""
        if last_n == 1:
            timings = self.last_step_ms()
        else:
            timings = {}
            for name, times in self._history.items():
                recent = times[-last_n:]
                timings[name] = sum(recent) / len(recent)

        total = sum(timings.values())
        if total == 0:
            return "No timing data recorded."

        lines = []
        for name, ms in timings.items():
            pct = 100.0 * ms / total
            lines.append(f"  {name:>12s}: {ms:7.2f} ms ({pct:5.1f}%)")

        lines.append(f"  {'total':>12s}: {total:7.2f} ms")
        return "\n".join(lines)


_global_metrics: StepMetrics | None = None


def get_global_metrics() -> StepMetrics:
    """Return the global StepMetrics instance, creating it on first access."""
    global _global_metrics
    if _global_metrics is None:
        _global_metrics = StepMetrics()
    return _global_metrics


class measure(ContextDecorator):
    """Standalone context manager / decorator for timing GPU regions.

    Records CUDA events on enter/exit without synchronizing the main thread.
    The CUDATimer polls for completion in a background thread. Timings are
    guaranteed available after StepMetrics.step().

    Usage:
        with measure(EventType.FORWARD):
            logits = model(x)

        with measure(EventType.FORWARD, stream=my_stream):
            logits = model(x)

        @measure(EventType.BACKWARD)
        def backward_pass():
            loss.backward()
    """

    def __init__(
        self,
        event_type: EventType,
        description: str | None = None,
        stream: torch.cuda.Stream | None = None,
    ):
        self.event_type = event_type
        self.description = description or event_type.value.capitalize()
        self._stream = stream

    def __enter__(self):
        self._timer = CUDATimer(stream=self._stream)
        step = get_step()
        step_prefix = f"[step {step}] " if step is not None else ""
        logger.debug(f"{step_prefix}{self.description} {self.event_type.value}_start")
        self._timer.record_start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._timer.record_end()
        get_global_metrics().record_pending(self.event_type, self._timer)

        step = get_step()
        step_prefix = f"[step {step}] " if step is not None else ""
        logger.debug(f"{step_prefix}{self.description} {self.event_type.value}_end")
        return False
