"""SPMD type system for verifying tensor parallelism correctness.

Inspired by ezyang's spmd_types (https://github.com/ezyang/spmd_types), which
builds a full type checker with automatic propagation through PyTorch's dispatch
system. Our version is deliberately simpler — two complementary layers instead
of one sophisticated one:

    1. Type comments: `# V@tp`, `# P@tp → I@tp` inline in TP functions.
       Zero cost, always present. The code documents its own invariants.

    2. assert_type(): Runtime value verification that broadcasts rank 0's
       tensor and compares across ranks. Enabled by NANIGPT_SPMD_CHECKS=1.
       This catches numerical divergence (different RNG seeds, non-deterministic
       ops) that type-level checking alone cannot.

ezyang's spmd_types propagates types automatically through an op registry and
catches errors at the operation level (e.g., nonlinear(P) is rejected before
it runs). We don't do that — our type comments are just comments, and
assert_type() only checks where you place it. The tradeoff: we're 138 lines
instead of 37 files, and the inline comments survive even if the checking
code is removed entirely.

Both systems share the same four types:

    | Type | Meaning                                    | Gradient |
    |------|--------------------------------------------|----------|
    | I    | Invariant: same value, same computation     | I        |
    | R    | Replicate: same value, may differ in compute | P        |
    | V    | Varying: different values per rank          | V        |
    | P    | Partial: pending sum across ranks           | R        |

Key rules:
    - nonlinear(P) is FORBIDDEN (most valuable catch — GELU on a partial
      sum gives wrong gradients with no error message)
    - linear(P) → P (matmul distributes over addition)
    - Forward types determine backward types: R↔P swap, I and V fixed

Usage:
    from nanigpt.distributed.spmd import SPMDType, assert_type

    # In TP functions:
    input_ = enter_parallel_region(input_, group)  # I@tp → R@tp
    if spmd_checks_enabled():
        assert_type(input_, SPMDType.REPLICATE, group)

    output = F.linear(input_, weight, bias)  # R@tp × V@tp → V@tp
    if spmd_checks_enabled():
        assert_type(output, SPMDType.VARYING, group)
"""

import logging
from enum import Enum

import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed import ProcessGroup

log = logging.getLogger(__name__)


class SPMDType(Enum):
    """SPMD tensor type on a single mesh axis."""

    INVARIANT = "invariant"  # Same value, same computation on all ranks
    REPLICATE = "replicate"  # Same value, ranks may compute differently
    VARYING = "varying"  # Different values per rank
    PARTIAL = "partial"  # Pending sum across ranks


def spmd_checks_enabled() -> bool:
    """Return True if SPMD type checking is enabled via environment variable."""
    from nanigpt.env import SPMD_CHECKS

    return SPMD_CHECKS.get_value()


def assert_type(
    tensor: Tensor,
    expected: SPMDType,
    group: ProcessGroup,
    dim_name: str = "tp",
    rtol: float = 1e-5,
    atol: float = 1e-5,
) -> None:
    """Debug assertion: verify a tensor's SPMD type by checking values across ranks.

    This is a synchronization point — only use in debug/test mode, never in
    the hot path of training.

    Verification logic:
        I/R: all ranks must have identical values (all-equal check)
        V:   at least one rank must differ (not-all-equal check)
        P:   no direct check possible (partial sums need the full value as reference)

    Args:
        tensor: The tensor to check.
        expected: Expected SPMD type.
        group: Process group to check across.
        dim_name: Mesh dimension name for error messages.
        rtol: Relative tolerance for floating point comparison.
        atol: Absolute tolerance for floating point comparison.

    Raises:
        AssertionError: If the tensor doesn't match the expected type.
    """
    world_size = dist.get_world_size(group)
    if world_size == 1:
        return  # All types are trivially valid with 1 rank

    rank = dist.get_rank(group)

    if expected in (SPMDType.INVARIANT, SPMDType.REPLICATE):
        # All ranks should have identical values
        # Broadcast rank 0's tensor and compare
        reference = tensor.clone()
        dist.broadcast(reference, src=0, group=group)
        all_equal = torch.allclose(tensor, reference, rtol=rtol, atol=atol)

        if not all_equal:
            max_diff = (tensor - reference).abs().max().item()
            raise AssertionError(
                f"SPMD type check failed: expected {expected.value}@{dim_name} "
                f"(identical across ranks), but rank {rank} differs from rank 0. "
                f"Max absolute difference: {max_diff}"
            )

    elif expected == SPMDType.VARYING:
        # At least one rank should differ from rank 0
        reference = tensor.clone()
        dist.broadcast(reference, src=0, group=group)

        local_equal = torch.allclose(tensor, reference, rtol=rtol, atol=atol)

        # Gather equality flags to rank 0
        equal_flag = torch.tensor([local_equal], dtype=torch.bool, device=tensor.device)
        all_flags = [
            torch.empty(1, dtype=torch.bool, device=tensor.device) for _ in range(world_size)
        ]
        dist.all_gather(all_flags, equal_flag, group=group)

        all_same = all(f.item() for f in all_flags)
        if all_same:
            raise AssertionError(
                f"SPMD type check failed: expected {expected.value}@{dim_name} "
                f"(different across ranks), but all ranks have identical values."
            )

    elif expected == SPMDType.PARTIAL:
        # Partial sums can't be verified without knowing the expected full value.
        # This is a no-op — the type comment is the documentation.
        pass
