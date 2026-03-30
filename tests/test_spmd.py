"""Tests for the SPMD type system (nanigpt/distributed/spmd.py).

Unit tests verify the enum and assert_type logic. Multi-GPU tests verify
that type assertions correctly detect type violations on real tensors
across ranks.
"""

import os

from tests.distributed_utils import distributed_test

# ---- Unit tests (no GPU required) ----


def test_spmd_type_enum_values():
    """SPMDType enum should have the four expected types."""
    from nanigpt.distributed.spmd import SPMDType

    assert SPMDType.INVARIANT.value == "invariant"
    assert SPMDType.REPLICATE.value == "replicate"
    assert SPMDType.VARYING.value == "varying"
    assert SPMDType.PARTIAL.value == "partial"
    assert len(SPMDType) == 4


def test_spmd_checks_disabled_by_default():
    """spmd_checks_enabled() should return False by default."""
    from nanigpt.distributed.spmd import spmd_checks_enabled

    old = os.environ.pop("NANIGPT_SPMD_CHECKS", None)
    try:
        assert not spmd_checks_enabled()
    finally:
        if old is not None:
            os.environ["NANIGPT_SPMD_CHECKS"] = old


def test_spmd_checks_enabled_with_env():
    """spmd_checks_enabled() should return True when NANIGPT_SPMD_CHECKS=1."""
    from nanigpt.distributed.spmd import spmd_checks_enabled

    old = os.environ.get("NANIGPT_SPMD_CHECKS")
    os.environ["NANIGPT_SPMD_CHECKS"] = "1"
    try:
        assert spmd_checks_enabled()
    finally:
        if old is None:
            del os.environ["NANIGPT_SPMD_CHECKS"]
        else:
            os.environ["NANIGPT_SPMD_CHECKS"] = old


# ---- Multi-GPU tests ----


@distributed_test(world_size=2)
def test_assert_invariant_passes_for_identical_tensors(rank, world_size):
    """assert_type(INVARIANT) should pass when all ranks have the same tensor."""
    import torch

    from nanigpt.distributed.spmd import SPMDType, assert_type

    group = torch.distributed.group.WORLD
    x = torch.ones(4, 8, device="cuda") * 42.0
    assert_type(x, SPMDType.INVARIANT, group)


@distributed_test(world_size=2)
def test_assert_replicate_passes_for_identical_tensors(rank, world_size):
    """assert_type(REPLICATE) should pass when all ranks have the same tensor."""
    import torch

    from nanigpt.distributed.spmd import SPMDType, assert_type

    group = torch.distributed.group.WORLD
    x = torch.ones(4, 8, device="cuda") * 42.0
    assert_type(x, SPMDType.REPLICATE, group)


@distributed_test(world_size=2)
def test_assert_invariant_fails_for_different_tensors(rank, world_size):
    """assert_type(INVARIANT) should fail when ranks have different tensors."""
    import torch

    from nanigpt.distributed.spmd import SPMDType, assert_type

    group = torch.distributed.group.WORLD
    x = torch.ones(4, 8, device="cuda") * (rank + 1)

    # Rank 0 compares to itself (broadcast src=0), so only non-zero ranks raise
    if rank == 0:
        assert_type(x, SPMDType.INVARIANT, group)  # passes on rank 0
    else:
        try:
            assert_type(x, SPMDType.INVARIANT, group)
            raise RuntimeError("assert_type should have raised AssertionError")
        except AssertionError:
            pass  # expected


@distributed_test(world_size=2)
def test_assert_varying_passes_for_different_tensors(rank, world_size):
    """assert_type(VARYING) should pass when ranks have different tensors."""
    import torch

    from nanigpt.distributed.spmd import SPMDType, assert_type

    group = torch.distributed.group.WORLD
    x = torch.ones(4, 8, device="cuda") * (rank + 1)
    assert_type(x, SPMDType.VARYING, group)


@distributed_test(world_size=2)
def test_assert_varying_fails_for_identical_tensors(rank, world_size):
    """assert_type(VARYING) should fail when all ranks have identical tensors."""
    import torch

    from nanigpt.distributed.spmd import SPMDType, assert_type

    group = torch.distributed.group.WORLD
    x = torch.ones(4, 8, device="cuda") * 42.0

    try:
        assert_type(x, SPMDType.VARYING, group)
        raise RuntimeError("assert_type should have raised AssertionError")
    except AssertionError:
        pass  # expected


@distributed_test(world_size=2)
def test_assert_partial_always_passes(rank, world_size):
    """assert_type(PARTIAL) should always pass (no verification possible)."""
    import torch

    from nanigpt.distributed.spmd import SPMDType, assert_type

    group = torch.distributed.group.WORLD
    # Different tensors — partial is a no-op
    x = torch.ones(4, 8, device="cuda") * (rank + 1)
    assert_type(x, SPMDType.PARTIAL, group)

    # Same tensors — still should not raise
    y = torch.ones(4, 8, device="cuda") * 42.0
    assert_type(y, SPMDType.PARTIAL, group)


@distributed_test(world_size=2)
def test_enter_produces_replicate_type(rank, world_size):
    """After enter_parallel_region, tensor should be R@tp (identical on all ranks)."""
    import torch

    from nanigpt.distributed.comm import enter_parallel_region
    from nanigpt.distributed.spmd import SPMDType, assert_type

    group = torch.distributed.group.WORLD
    x = torch.ones(4, 8, device="cuda") * 42.0
    y = enter_parallel_region(x, group)
    assert_type(y, SPMDType.REPLICATE, group)


@distributed_test(world_size=2)
def test_exit_produces_invariant_type(rank, world_size):
    """After exit_parallel_region, tensor should be I@tp (all-reduced)."""
    import torch

    from nanigpt.distributed.comm import exit_parallel_region
    from nanigpt.distributed.spmd import SPMDType, assert_type

    group = torch.distributed.group.WORLD
    x = torch.ones(4, 8, device="cuda") * (rank + 1)
    y = exit_parallel_region(x, group)
    assert_type(y, SPMDType.INVARIANT, group)


@distributed_test(world_size=2)
def test_gather_output_is_replicate(rank, world_size):
    """After gather_into_parallel_region, output should be R@tp (full tensor on all)."""
    import torch

    from nanigpt.distributed.comm import gather_into_parallel_region
    from nanigpt.distributed.spmd import SPMDType, assert_type

    group = torch.distributed.group.WORLD
    x = torch.ones(4, 8, device="cuda") * (rank + 1)
    y = gather_into_parallel_region(x, group)
    assert_type(y, SPMDType.REPLICATE, group)

