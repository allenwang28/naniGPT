"""GPU-accurate timing for training steps.

StepMetrics is the central collector. Timing sources stash data during a
step; StepMetrics.step() resolves everything at the step boundary and
moves it into history.

Current timing sources:

    1. CUDATimer / measure — CUDA events on the compute stream, resolved
       via background polling (no CPU sync). Used for forward, backward,
       optimizer, etc.

    2. CUDA event pairs (via NaniProcessGroup) — recorded around each
       collective dispatch, resolved via start.elapsed_time(end).
       Measures how long the compute stream is occupied by each collective.
       Used for per-parallelism comm breakdown (TP, FSDP, DDP).

New timing sources plug in by adding a stash method + resolution in step().
"""

import logging
import threading
import time
from contextlib import ContextDecorator
from dataclasses import dataclass

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

    Two sources of timing data:
    1. CUDATimer (via record_pending): for compute regions timed on the
       current CUDA stream. Resolved by joining background polling threads.
    2. CUDA event pairs (via record_events): for comm collectives timed by
       NaniProcessGroup. Resolved via start.elapsed_time(end).

    On step(), both sources are resolved and timings are moved into history.
    """

    def __init__(self):
        self._pending: list[tuple[EventType, CUDATimer]] = []
        # (type, start, end)
        self._pending_events: list[tuple[EventType, torch.cuda.Event, torch.cuda.Event]] = []
        self._current: dict[str, float] = {}
        self._history: list[dict[str, float]] = []

    def record_pending(self, event_type: EventType, timer: CUDATimer) -> None:
        """Stash a timer for resolution at step boundary."""
        self._pending.append((event_type, timer))

    def record_events(
        self,
        event_type: EventType,
        start: torch.cuda.Event,
        end: torch.cuda.Event,
    ) -> None:
        """Stash a CUDA event pair for resolution at step boundary."""
        self._pending_events.append((event_type, start, end))

    def record(self, event_type: EventType, elapsed_ms: float) -> None:
        """Record a region's elapsed time directly (for testing or CPU timing)."""
        key = event_type.value
        self._current[key] = self._current.get(key, 0.0) + elapsed_ms

    def step(self) -> None:
        """Finalize the current step: resolve pending timers and move into history."""
        # Resolve CUDATimer-based regions
        for event_type, timer in self._pending:
            elapsed = timer.elapsed_ms()
            key = event_type.value
            self._current[key] = self._current.get(key, 0.0) + elapsed
        self._pending.clear()

        # Resolve CUDA event pairs (from CommTimer)
        for event_type, start, end in self._pending_events:
            end.synchronize()
            elapsed = start.elapsed_time(end)
            key = event_type.value
            self._current[key] = self._current.get(key, 0.0) + elapsed
        self._pending_events.clear()

        self._history.append(dict(self._current))
        self._current.clear()

    def last_step_ms(self) -> dict[str, float]:
        """Return timings from the most recent step only."""
        if not self._history:
            return {}
        return dict(self._history[-1])

    def mean_ms(self) -> dict[str, float]:
        """Return mean timings across all recorded steps."""
        if not self._history:
            return {}
        all_keys = {k for step in self._history for k in step}
        return {
            key: sum(step.get(key, 0.0) for step in self._history)
            / sum(1 for step in self._history if key in step)
            for key in all_keys
        }

    def report(self, last_n: int = 1) -> str:
        """Format a profiling report for the last N steps."""
        if last_n == 1:
            timings = self.last_step_ms()
        else:
            recent = self._history[-last_n:]
            all_keys = {k for step in recent for k in step}
            timings = {
                key: sum(step.get(key, 0.0) for step in recent)
                / sum(1 for step in recent if key in step)
                for key in all_keys
            }

        total = sum(timings.values())
        if total == 0:
            return "No timing data recorded."

        lines = []
        for name, ms in timings.items():
            pct = 100.0 * ms / total
            lines.append(f"  {name:>12s}: {ms:7.2f} ms ({pct:5.1f}%)")

        lines.append(f"  {'total':>12s}: {total:7.2f} ms")
        return "\n".join(lines)


_PHASE_ORDER = [
    EventType.FORWARD,
    EventType.BACKWARD,
    EventType.OPTIMIZER,
    EventType.DATA,
]

_COMM_ORDER = [EventType.TP_COMM, EventType.FSDP_COMM, EventType.DP_COMM]
_COMM_DISPLAY = {"tp_comm": "tp", "fsdp_comm": "fsdp", "dp_comm": "dp"}


@dataclass(slots=True)
class StepTimings:
    """Structured timings for a single step (or averaged across steps).

    Separates compute phases (which determine step duration) from
    communication volume (overlapped with compute).
    """

    step_timings: dict[str, float]
    """Compute phase breakdown: forward, backward, optimizer, data, etc."""

    comm_timings: dict[str, float]
    """Communication volume per parallelism dimension (overlapped with compute)."""

    @property
    def step_ms(self) -> float:
        """Total step duration (compute phases only, excludes comm)."""
        return sum(self.step_timings.values())

    def phase_breakdown(self) -> list[tuple[str, float, float]]:
        """Return (name, ms, pct) for each compute phase in display order.

        Known phases (forward, backward, optimizer, data) come first,
        followed by any unexpected phases.
        """
        total = self.step_ms
        ordered = [
            (e.value, self.step_timings[e.value])
            for e in _PHASE_ORDER
            if e.value in self.step_timings
        ]
        known = {e.value for e in _PHASE_ORDER}
        other = [(k, v) for k, v in self.step_timings.items() if k not in known]
        return [
            (name, ms, 100.0 * ms / total if total > 0 else 0.0) for name, ms in ordered + other
        ]

    def comm_breakdown(self) -> list[tuple[str, float]]:
        """Return (display_name, ms) for each comm dimension in display order.

        Returns empty list if no comm timings recorded.
        """
        items = [
            (_COMM_DISPLAY.get(e.value, e.value), self.comm_timings[e.value])
            for e in _COMM_ORDER
            if e.value in self.comm_timings
        ]
        if EventType.COMMUNICATION.value in self.comm_timings:
            items.append(
                (
                    EventType.COMMUNICATION.value,
                    self.comm_timings[EventType.COMMUNICATION.value],
                )
            )
        return items

    @property
    def comm_total_ms(self) -> float:
        """Total communication volume across all dimensions."""
        return sum(ms for _, ms in self.comm_breakdown())


def _split_timings(timings: dict[str, float]) -> StepTimings:
    """Separate raw timings dict into step vs comm."""
    from nanigpt.profiling.event_types import COMM_EVENTS

    step = {k: v for k, v in timings.items() if k != EventType.EVAL.value and k not in COMM_EVENTS}
    comm = {k: v for k, v in timings.items() if k in COMM_EVENTS}
    return StepTimings(step_timings=step, comm_timings=comm)


def last_step_timings() -> StepTimings:
    """Return structured timings from the most recent step."""
    return _split_timings(get_global_metrics().last_step_ms())


def mean_step_timings() -> StepTimings:
    """Return structured timings averaged across all recorded steps."""
    return _split_timings(get_global_metrics().mean_ms())


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
