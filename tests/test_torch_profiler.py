"""Tests for PyTorch profiler integration: config, callbacks, TorchProfiler, and kernel table."""

import logging

import pytest

import nanigpt.profiling.context as ctx
from nanigpt.profiling.context import (
    init_context,
    register_step_end,
    step_context,
    unregister_step_end,
)
from nanigpt.profiling.torch_profiler import TorchProfiler, _parse_windows


@pytest.fixture(autouse=True)
def _reset_context():
    """Reset global context and callbacks before each test."""
    ctx.GLOBAL_CONTEXT = None
    ctx._STEP_END_CALLBACKS.clear()
    yield
    ctx.GLOBAL_CONTEXT = None
    ctx._STEP_END_CALLBACKS.clear()


def test_profiler_config_defaults():
    config = TorchProfiler.Config()
    assert config.enabled is True
    assert config.windows == "10-12"
    assert config.warmup_steps == 1
    assert config.top_n == 15
    assert config.export_trace is True
    assert config.record_shapes is True
    assert config.with_stack is False
    assert config.with_flops is True


def test_torch_profiler_disabled():
    """enabled=False should be a complete no-op — no callback effect."""
    profiler = TorchProfiler(TorchProfiler.Config(enabled=False))
    register_step_end(profiler)
    assert profiler._windows == []
    # Calling it does nothing
    profiler()


def test_step_end_callbacks():
    """Verify register/unregister and that callbacks fire from step_context."""
    init_context()
    calls = []

    def cb():
        calls.append("fired")

    register_step_end(cb)
    assert cb in ctx._STEP_END_CALLBACKS

    with step_context(1):
        pass
    assert calls == ["fired"]

    with step_context(2):
        pass
    assert calls == ["fired", "fired"]

    unregister_step_end(cb)
    assert cb not in ctx._STEP_END_CALLBACKS

    with step_context(3):
        pass
    assert calls == ["fired", "fired"]  # no new call


def test_step_end_multiple_callbacks():
    """Multiple callbacks fire in registration order."""
    init_context()
    order = []

    def cb_a():
        order.append("a")

    def cb_b():
        order.append("b")

    register_step_end(cb_a)
    register_step_end(cb_b)

    with step_context(1):
        pass
    assert order == ["a", "b"]


def test_kernel_table_format_no_gpu(caplog):
    """CPU-only profiler has no GPU kernels, so the table reports that."""
    import torch
    import torch.profiler

    from nanigpt.profiling.torch_profiler import _print_kernel_table

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU],
    ) as prof:
        x = torch.randn(64, 64)
        _ = x @ x

    with caplog.at_level(logging.INFO, logger="nanigpt.profiling.torch_profiler"):
        _print_kernel_table(prof.key_averages(), top_n=5)

    assert "No GPU kernels recorded" in caplog.text


def test_kernel_table_format_with_entries(caplog):
    """Verify the table header and row formatting with mock data."""
    from unittest.mock import MagicMock

    from nanigpt.profiling.torch_profiler import _print_kernel_table

    # Create a mock entry
    entry = MagicMock()
    entry.key = "aten::mm"
    entry.self_device_time_total = 5000  # 5ms in us
    entry.self_cpu_time_total = 1000
    entry.count = 10
    entry.flops = 0

    with caplog.at_level(logging.INFO, logger="nanigpt.profiling.torch_profiler"):
        _print_kernel_table([entry], top_n=5)

    log_text = caplog.text
    assert "aten::mm" in log_text
    assert "ms" in log_text
    assert "%" in log_text


# ---- Window parsing tests ----


def test_parse_single_window():
    assert _parse_windows("10-12") == [(10, 12)]


def test_parse_multiple_windows():
    assert _parse_windows("40-50,55-60") == [(40, 50), (55, 60)]


def test_parse_windows_sorts():
    """Windows are sorted by start step regardless of input order."""
    assert _parse_windows("55-60,40-50") == [(40, 50), (55, 60)]


def test_parse_windows_with_spaces():
    assert _parse_windows("40-50, 55-60") == [(40, 50), (55, 60)]


def test_parse_windows_reversed_raises():
    with pytest.raises(ValueError, match="must be > start"):
        _parse_windows("50-40")


def test_parse_windows_equal_raises():
    with pytest.raises(ValueError, match="must be > start"):
        _parse_windows("10-10")


def test_parse_windows_overlap_raises():
    with pytest.raises(ValueError, match="overlap"):
        _parse_windows("10-20,15-25")


def test_parse_windows_bad_format_raises():
    with pytest.raises(ValueError, match="Invalid window format"):
        _parse_windows("10")


def test_config_validates_windows():
    """Config.__post_init__ catches bad window strings."""
    with pytest.raises(ValueError):
        TorchProfiler.Config(windows="50-40")


def test_profiler_creates_multiple_windows():
    """Multiple windows produce multiple _WindowState entries."""
    profiler = TorchProfiler(TorchProfiler.Config(windows="10-12,20-22"))
    assert len(profiler._windows) == 2
    assert profiler._windows[0].start_step == 10
    assert profiler._windows[0].end_step == 12
    assert profiler._windows[1].start_step == 20
    assert profiler._windows[1].end_step == 22


def test_profiler_config_perfetto_defaults():
    """Perfetto and trace_dir have sensible defaults."""
    config = TorchProfiler.Config()
    assert config.perfetto is False
    assert config.trace_dir == "traces"
    assert config.export_trace is True


def test_serve_perfetto_no_traces():
    """serve_perfetto is a no-op when there are no trace paths."""
    from nanigpt.profiling.perfetto import serve_traces

    # Should not raise
    serve_traces([])
