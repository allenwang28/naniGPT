"""Session-scoped fixtures for distributed GPU testing.

Starts Ray once per pytest session so distributed_test() can reuse
the running cluster instead of spinning up/down per test.
"""

import pytest
import torch


def pytest_configure(config):
    """Register the 'gpu' marker."""
    config.addinivalue_line("markers", "gpu: test requires GPU(s)")


@pytest.fixture(scope="session", autouse=True)
def _ray_session():
    """Start Ray once for the entire test session.

    distributed_test() checks ray.is_initialized() and skips its own
    init/shutdown when Ray is already running. This cuts per-test
    overhead from ~20s to ~2s.
    """
    num_gpus = torch.cuda.device_count()
    if num_gpus < 2:
        return  # Don't start Ray if not enough GPUs; tests will skip

    import ray

    ray.init(ignore_reinit_error=True, num_gpus=num_gpus)
    yield
    ray.shutdown()
