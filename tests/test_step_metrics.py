"""Tests for StepMetrics: record/step lifecycle, accumulation, averaging, and report formatting."""

from nanigpt.profiling.event_types import EventType
from nanigpt.profiling.timer import StepMetrics


def test_record_and_step():
    m = StepMetrics()
    m.record(EventType.FORWARD, 10.0)
    m.record(EventType.BACKWARD, 20.0)
    m.step()

    last = m.last_step_ms()
    assert last == {"forward": 10.0, "backward": 20.0}


def test_accumulates_within_step():
    m = StepMetrics()
    m.record(EventType.FORWARD, 5.0)
    m.record(EventType.FORWARD, 3.0)
    m.step()

    assert m.last_step_ms() == {"forward": 8.0}


def test_mean_across_steps():
    m = StepMetrics()
    m.record(EventType.FORWARD, 10.0)
    m.step()
    m.record(EventType.FORWARD, 20.0)
    m.step()

    assert m.mean_ms() == {"forward": 15.0}


def test_last_step_returns_most_recent():
    m = StepMetrics()
    m.record(EventType.DATA, 1.0)
    m.step()
    m.record(EventType.DATA, 99.0)
    m.step()

    assert m.last_step_ms() == {"data": 99.0}


def test_report_format():
    m = StepMetrics()
    m.record(EventType.FORWARD, 30.0)
    m.record(EventType.BACKWARD, 70.0)
    m.step()

    report = m.report()
    assert "forward" in report
    assert "backward" in report
    assert "total" in report
    assert "100.00 ms" in report


def test_report_no_data():
    m = StepMetrics()
    assert m.report() == "No timing data recorded."


def test_report_last_n():
    m = StepMetrics()
    m.record(EventType.FORWARD, 10.0)
    m.step()
    m.record(EventType.FORWARD, 20.0)
    m.step()
    m.record(EventType.FORWARD, 30.0)
    m.step()

    report = m.report(last_n=2)
    # Average of last 2 steps: (20 + 30) / 2 = 25
    assert "25.00 ms" in report
