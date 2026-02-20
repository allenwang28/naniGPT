"""Tests for global step context: init, scoping, tag management, and error handling."""

import pytest

import nanigpt.profiling.context as ctx
from nanigpt.profiling.context import (
    add_step_tag,
    get_step,
    init_context,
    step_context,
)


@pytest.fixture(autouse=True)
def _reset_context():
    """Reset global context before each test."""
    ctx.GLOBAL_CONTEXT = None
    yield
    ctx.GLOBAL_CONTEXT = None


def test_get_step_before_init():
    assert get_step() is None


def test_init_and_get_step():
    init_context()
    assert get_step() is None


def test_step_context_sets_step():
    init_context()
    with step_context(5):
        assert get_step() == 5


def test_step_context_clears_tags_on_exit():
    init_context()
    with step_context(1):
        add_step_tag("warmup")
        assert ctx.GLOBAL_CONTEXT.step_tags == ["warmup"]
    assert ctx.GLOBAL_CONTEXT.step_tags == []


def test_step_context_without_init_raises():
    with pytest.raises(RuntimeError, match="init_context"):
        with step_context(1):
            pass


def test_nested_step_contexts():
    init_context()
    with step_context(1):
        assert get_step() == 1
        with step_context(2):
            assert get_step() == 2
        # After inner exits, step is still 2 (no restore)
        assert get_step() == 2


def test_tags_cleared_between_steps():
    init_context()
    with step_context(1):
        add_step_tag("first")
    with step_context(2):
        assert ctx.GLOBAL_CONTEXT.step_tags == []
