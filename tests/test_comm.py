"""Tests for communication primitives (nanigpt/distributed/comm.py).

Each primitive is an autograd.Function with a forward collective and its
mathematically dual backward collective (see the table in comm.py).
TP correctness depends on both directions being right — a correct forward
with a wrong backward silently produces wrong gradients — so every test
pair verifies both directions independently.
"""

from tests.distributed_utils import distributed_test


@distributed_test(world_size=2)
def test_enter_forward_is_identity(rank, world_size):
    """Enter forward should be identity — output equals input."""
    import torch

    from nanigpt.distributed.comm import enter_parallel_region

    group = torch.distributed.group.WORLD
    x = torch.randn(4, 8, device="cuda") * (rank + 1)
    x_clone = x.clone()

    y = enter_parallel_region(x, group)
    assert torch.equal(y, x_clone), "Enter forward should be identity"


@distributed_test(world_size=2)
def test_enter_backward_is_all_reduce(rank, world_size):
    """Enter backward should all-reduce the gradient."""
    import torch

    from nanigpt.distributed.comm import enter_parallel_region

    group = torch.distributed.group.WORLD
    x = torch.randn(4, 8, device="cuda", requires_grad=True)

    y = enter_parallel_region(x, group)
    # Create rank-specific gradient
    grad = torch.ones_like(y) * (rank + 1)
    y.backward(grad)

    # Backward all-reduces: grad should be sum of all ranks' grads
    expected_grad = torch.ones_like(x) * sum(r + 1 for r in range(world_size))
    assert torch.allclose(x.grad, expected_grad), (
        f"Enter backward should all-reduce. "
        f"Got {x.grad[0, 0].item()}, expected {expected_grad[0, 0].item()}"
    )


@distributed_test(world_size=2)
def test_exit_forward_is_all_reduce(rank, world_size):
    """Exit forward should all-reduce the input."""
    import torch

    from nanigpt.distributed.comm import exit_parallel_region

    group = torch.distributed.group.WORLD
    x = torch.ones(4, 8, device="cuda") * (rank + 1)

    y = exit_parallel_region(x, group)
    expected = torch.ones(4, 8, device="cuda") * sum(r + 1 for r in range(world_size))
    assert torch.allclose(y, expected), (
        f"Exit forward should all-reduce. Got {y[0, 0].item()}, expected {expected[0, 0].item()}"
    )


@distributed_test(world_size=2)
def test_exit_backward_is_identity(rank, world_size):
    """Exit backward should be identity — grad passes through unchanged."""
    import torch

    from nanigpt.distributed.comm import exit_parallel_region

    group = torch.distributed.group.WORLD
    x = (torch.ones(4, 8, device="cuda") * (rank + 1)).requires_grad_(True)

    y = exit_parallel_region(x, group)
    grad = torch.ones_like(y) * 7.0
    y.backward(grad)

    # Backward is identity: grad should pass through unchanged
    assert torch.allclose(x.grad, grad), (
        f"Exit backward should be identity. Got {x.grad[0, 0].item()}, expected 7.0"
    )


@distributed_test(world_size=2)
def test_gather_forward_is_all_gather(rank, world_size):
    """Gather forward should all-gather along dim 0."""
    import torch

    from nanigpt.distributed.comm import gather_into_parallel_region

    group = torch.distributed.group.WORLD
    # Each rank has a chunk of size 4
    x = torch.ones(4, 8, device="cuda") * (rank + 1)

    y = gather_into_parallel_region(x, group)
    assert y.shape == (4 * world_size, 8), f"Expected shape {(4 * world_size, 8)}, got {y.shape}"

    # Verify content: first chunk from rank 0, second from rank 1
    for r in range(world_size):
        chunk = y[r * 4 : (r + 1) * 4]
        expected = torch.ones(4, 8, device="cuda") * (r + 1)
        assert torch.allclose(chunk, expected), f"Chunk {r} mismatch"


@distributed_test(world_size=2)
def test_gather_backward_is_reduce_scatter(rank, world_size):
    """Gather backward should reduce-scatter the gradient."""
    import torch

    from nanigpt.distributed.comm import gather_into_parallel_region

    group = torch.distributed.group.WORLD
    x = torch.ones(4, 8, device="cuda", requires_grad=True)

    y = gather_into_parallel_region(x, group)
    # Grad is all ones — reduce-scatter should give each rank a chunk summed across ranks
    grad = torch.ones_like(y)
    y.backward(grad)

    # Reduce-scatter: each rank gets its chunk summed = world_size * ones
    expected = torch.ones(4, 8, device="cuda") * world_size
    assert torch.allclose(x.grad, expected), (
        f"Gather backward should reduce-scatter. "
        f"Got {x.grad[0, 0].item()}, expected {expected[0, 0].item()}"
    )


@distributed_test(world_size=2)
def test_scatter_forward_is_reduce_scatter(rank, world_size):
    """Scatter forward should reduce-scatter along dim 0."""
    import torch

    from nanigpt.distributed.comm import scatter_from_parallel_region

    group = torch.distributed.group.WORLD
    # Input is (world_size * 4, 8) on each rank, with rank-specific values
    x = torch.ones(world_size * 4, 8, device="cuda") * (rank + 1)

    y = scatter_from_parallel_region(x, group)
    assert y.shape == (4, 8), f"Expected shape (4, 8), got {y.shape}"

    # Each chunk is summed across ranks: sum(r+1 for r in range(world_size))
    expected_val = sum(r + 1 for r in range(world_size))
    expected = torch.ones(4, 8, device="cuda") * expected_val
    assert torch.allclose(y, expected), (
        f"Scatter forward mismatch. Got {y[0, 0].item()}, expected {expected_val}"
    )


@distributed_test(world_size=2)
def test_scatter_backward_is_all_gather(rank, world_size):
    """Scatter backward should all-gather the gradient."""
    import torch

    from nanigpt.distributed.comm import scatter_from_parallel_region

    group = torch.distributed.group.WORLD
    x = torch.ones(world_size * 4, 8, device="cuda", requires_grad=True)

    y = scatter_from_parallel_region(x, group)
    # Each rank gets a chunk of size 4; use rank-specific grad
    grad = torch.ones(4, 8, device="cuda") * (rank + 1)
    y.backward(grad)

    # Backward is all-gather: reconstruct full gradient from all ranks' chunks
    assert x.grad.shape == (world_size * 4, 8)
    for r in range(world_size):
        chunk = x.grad[r * 4 : (r + 1) * 4]
        expected = torch.ones(4, 8, device="cuda") * (r + 1)
        assert torch.allclose(chunk, expected), f"Backward chunk {r} mismatch"
