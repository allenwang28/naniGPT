"""Tests for FLOP counter math: achieved TFLOPS, MFU, and edge cases."""

from nanigpt.profiling.flop_counter import achieved_tflops, mfu


def test_achieved_tflops():
    # 1e12 FLOPs in 1000ms (1s) = 1.0 TFLOPS
    assert achieved_tflops(1e12, 1000.0) == 1.0


def test_achieved_tflops_zero_time():
    assert achieved_tflops(1e12, 0.0) == 0.0


def test_achieved_tflops_negative_time():
    assert achieved_tflops(1e12, -1.0) == 0.0


def test_mfu_known_gpu():
    # 100 TFLOPS achieved on A100-SXM (312 TFLOPS peak)
    utilization = mfu(100.0, "A100-SXM")
    assert abs(utilization - 100.0 / 312.0) < 1e-6


def test_mfu_unknown_gpu():
    assert mfu(100.0, "unknown") == 0.0


def test_mfu_zero_peak():
    # Shouldn't happen with real data, but guard against division by zero
    assert mfu(100.0, "nonexistent") == 0.0
