"""Theoretical FLOP counting and hardware utilization measurement.

Computes analytical FLOP counts from model architecture.

Uses a FlopCountable Protocol so any model that implements flop_count()
and num_non_embedding_params() works without importing concrete model classes.

Key metrics:
    - achieved_tflops: actual TFLOPS from measured step time
    - mfu: Model FLOP Utilization = achieved / hardware peak
    - GPU_PEAK_TFLOPS: reference table for common GPUs (bf16 tensor core)
"""

from typing import Protocol


class FlopCountable(Protocol):
    """Interface for models that can report their theoretical FLOP count."""

    def flop_count(self, batch_size: int, seq_len: int) -> int: ...
    def num_non_embedding_params(self) -> int: ...


# Peak TFLOPS for common GPUs (bf16/fp16 tensor core)
GPU_PEAK_TFLOPS = {
    "A100-SXM": 312.0,
    "A100-PCIe": 312.0,
    "H100-SXM": 989.0,
    "H100-PCIe": 756.0,
    "L40S": 362.0,
    "A10G": 125.0,
}


def forward_flops(model: FlopCountable, batch_size: int, seq_len: int) -> int:
    """Theoretical forward-pass FLOPs."""
    return model.flop_count(batch_size, seq_len)


def backward_flops(model: FlopCountable, batch_size: int, seq_len: int) -> int:
    """Theoretical backward-pass FLOPs (≈ 2x forward)."""
    return 2 * forward_flops(model, batch_size, seq_len)


def total_flops(model: FlopCountable, batch_size: int, seq_len: int) -> int:
    """Total FLOPs for one training step (forward + backward)."""
    fwd = forward_flops(model, batch_size, seq_len)
    return 3 * fwd  # forward + backward (2x forward)


def achieved_tflops(flops: int, elapsed_ms: float) -> float:
    """Compute achieved TFLOPS from FLOP count and wall time."""
    if elapsed_ms <= 0:
        return 0.0
    elapsed_s = elapsed_ms / 1000.0
    return flops / elapsed_s / 1e12


def mfu(achieved: float, gpu_name: str) -> float:
    """Model FLOP Utilization: achieved TFLOPS / hardware peak TFLOPS."""
    peak = GPU_PEAK_TFLOPS.get(gpu_name)
    if peak is None or peak == 0:
        return 0.0
    return achieved / peak


def detect_gpu() -> str:
    """Best-effort detection of current GPU type. Returns key into GPU_PEAK_TFLOPS."""
    try:
        import torch

        name = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
    except Exception:
        return "unknown"

    name_upper = name.upper()

    # Try matching by name string first
    if "H100" in name_upper:
        return "H100-SXM" if "SXM" in name_upper else "H100-PCIe"
    if "A100" in name_upper:
        return "A100-SXM" if "SXM" in name_upper else "A100-PCIe"
    if "L40S" in name_upper:
        return "L40S"
    if "A10G" in name_upper:
        return "A10G"

    # Fall back to SM architecture + memory heuristics for OEM/board variants
    # (e.g. "NVIDIA PG509-210" is an A100-SXM 80GB)
    sm = (props.major, props.minor)
    mem_gb = props.total_memory / (1024**3)
    if sm == (8, 0) and mem_gb > 35:
        return "A100-SXM"
    if sm == (9, 0) and mem_gb > 35:
        return "H100-SXM"

    return "unknown"
