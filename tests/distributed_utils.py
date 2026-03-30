"""Distributed test utilities using Ray for multi-GPU testing.

Provides a `distributed_test` decorator that runs test functions across
multiple GPU workers with real NCCL collectives.

When Ray is already running (e.g. from conftest.py session fixture), reuses
the existing cluster. Otherwise spins up a fresh one per test.

Usage:

    from tests.distributed_utils import distributed_test

    @distributed_test(world_size=2)
    def test_all_reduce(rank, world_size):
        import torch.distributed as dist
        x = torch.ones(4, device="cuda") * rank
        dist.all_reduce(x)
        expected = sum(range(world_size))
        assert torch.allclose(x, torch.full((4,), expected, device="cuda", dtype=x.dtype))
"""

import os
import traceback

import pytest
import torch


def _worker_fn(rank, world_size, test_fn, master_addr, master_port):
    """Run inside a Ray actor: init process group, run test, cleanup."""
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port

    from nanigpt.distributed import cleanup_distributed, init_distributed

    try:
        init_distributed(rank, world_size, backend="nccl")
        test_fn(rank, world_size)
        return None  # success
    except Exception:
        return traceback.format_exc()
    finally:
        cleanup_distributed()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def distributed_test(world_size: int = 2):
    """Decorator that runs a test function across multiple GPU workers via Ray.

    The decorated function receives (rank, world_size) as arguments. Process
    groups are initialized before the function runs and cleaned up after.

    If Ray is already running (from session fixture), reuses it and creates
    ephemeral task actors (fast — no cluster startup). Otherwise starts
    Ray from scratch (slow — ~20s).

    Skips the test if fewer GPUs are available than requested.
    """

    def decorator(fn):
        def wrapper():
            num_gpus = torch.cuda.device_count()
            if num_gpus < world_size:
                pytest.skip(f"Need {world_size} GPUs, only {num_gpus} available")

            import ray

            # If Ray is already running (from session fixture), reuse it.
            # Otherwise start fresh.
            we_started_ray = False
            if not ray.is_initialized():
                ray.init(ignore_reinit_error=True, num_gpus=world_size)
                we_started_ray = True

            master_addr = os.environ.get("MASTER_ADDR", "localhost")
            master_port = os.environ.get("MASTER_PORT", "29500")

            try:

                @ray.remote(num_gpus=1)
                def run_worker(rank, ws, fn_ref, addr, port):
                    return _worker_fn(rank, ws, fn_ref, addr, port)

                futures = [
                    run_worker.remote(rank, world_size, fn, master_addr, master_port)
                    for rank in range(world_size)
                ]

                results = ray.get(futures)

                errors = [(rank, err) for rank, err in enumerate(results) if err is not None]
                if errors:
                    msg_parts = [f"Rank {rank} failed:\n{err}" for rank, err in errors]
                    pytest.fail("\n\n".join(msg_parts))
            finally:
                if we_started_ray:
                    ray.shutdown()

        # Preserve test name for pytest discovery, but not the signature
        # (pytest would interpret rank/world_size params as fixture names)
        wrapper.__name__ = fn.__name__
        wrapper.__qualname__ = fn.__qualname__
        wrapper.__module__ = fn.__module__
        wrapper.__doc__ = fn.__doc__
        return wrapper

    return decorator
