"""Global step context for training instrumentation.

Provides a lightweight global that tracks which training step is currently
executing. Any code can call get_step() to annotate logs or metrics with
the current step number without needing a reference to the training loop.

Usage:

    init_context()                     # once at startup

    for step in range(1, N + 1):
        with step_context(step):       # sets step, clears tags on exit,
            ...                        # finalizes StepMetrics

        # after the with block, metrics for this step are available
        get_global_metrics().last_step_ms()

Tags are optional per-step annotations (e.g. "warmup", "eval") that get
cleared automatically between steps.
"""

import contextlib
import dataclasses
from collections.abc import Callable, Generator
from dataclasses import field


@dataclasses.dataclass
class StepContext:
    step: int | None = None
    step_tags: list[str] = field(default_factory=list)


GLOBAL_CONTEXT: StepContext | None = None

_STEP_END_CALLBACKS: list[Callable[[], None]] = []


def register_step_end(fn: Callable[[], None]) -> None:
    """Register a callback to run at the end of each step (after StepMetrics.step())."""
    _STEP_END_CALLBACKS.append(fn)


def unregister_step_end(fn: Callable[[], None]) -> None:
    """Remove a previously registered step-end callback."""
    _STEP_END_CALLBACKS.remove(fn)


def init_context() -> None:
    global GLOBAL_CONTEXT
    GLOBAL_CONTEXT = StepContext()


@contextlib.contextmanager
def step_context(step: int) -> Generator[None]:
    """Scope a training step: sets the step on entry, finalizes metrics and clears tags on exit."""
    if GLOBAL_CONTEXT is None:
        raise RuntimeError("Call init_context() before step_context()")
    GLOBAL_CONTEXT.step = step
    GLOBAL_CONTEXT.step_tags.clear()
    try:
        yield
    finally:
        # Import here to avoid circular import (timer.py imports context.py)
        from nanigpt.profiling.timer import get_global_metrics

        get_global_metrics().step()
        for cb in _STEP_END_CALLBACKS:
            cb()
        GLOBAL_CONTEXT.step_tags.clear()


def get_step() -> int | None:
    if GLOBAL_CONTEXT is None:
        return None
    return GLOBAL_CONTEXT.step


def add_step_tag(tag: str) -> None:
    if GLOBAL_CONTEXT is None:
        raise RuntimeError("Call init_context() before add_step_tag()")
    GLOBAL_CONTEXT.step_tags.append(tag)


def clear_step_tags() -> None:
    if GLOBAL_CONTEXT is None:
        raise RuntimeError("Call init_context() before clear_step_tags()")
    GLOBAL_CONTEXT.step_tags.clear()
