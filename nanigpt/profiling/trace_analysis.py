"""Post-processing analysis on profiler trace data.

Operates on raw Kineto events from torch.profiler to compute metrics
that aren't available from key_averages() alone — e.g. exposed vs
hidden communication time relative to compute.

Algorithm approach:
merge compute kernel intervals on the GPU timeline, then for each
NCCL kernel compute how much falls outside merged compute.
"""

import bisect
import logging

import torch.profiler

log = logging.getLogger(__name__)

_NCCL_PREFIXES = ("nccl", "ncclKernel", "ncclDevKernel")


def _is_comm_kernel(name: str) -> bool:
    """Return True if the kernel name is an NCCL communication op."""
    return any(name.startswith(p) for p in _NCCL_PREFIXES)


def _merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge overlapping/adjacent intervals. Returns sorted non-overlapping list."""
    if not intervals:
        return []
    sorted_ivs = sorted(intervals)
    merged = [sorted_ivs[0]]
    for start, end in sorted_ivs[1:]:
        if start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _exposed_duration(start: int, end: int, merged_compute: list[tuple[int, int]]) -> int:
    """Compute the portion of [start, end) NOT covered by merged compute intervals.

    Uses binary search to find the first relevant compute interval, then
    walks forward. All values in nanoseconds.
    """
    if not merged_compute or start >= end:
        return end - start

    # Find the first compute interval that could overlap: its end > start
    # bisect on the end values of merged_compute
    ends = [iv[1] for iv in merged_compute]
    idx = bisect.bisect_right(ends, start)

    covered = 0
    for i in range(idx, len(merged_compute)):
        c_start, c_end = merged_compute[i]
        if c_start >= end:
            break
        overlap_start = max(start, c_start)
        overlap_end = min(end, c_end)
        if overlap_start < overlap_end:
            covered += overlap_end - overlap_start

    return (end - start) - covered


def compute_exposed_comm(prof: torch.profiler.profile) -> dict | None:
    """Compute NCCL kernel time not overlapped with compute kernels.

    Uses the Kineto events API to get raw GPU kernel intervals, partitions
    into NCCL vs compute, merges compute intervals, then computes exposed
    (non-overlapped) duration for each NCCL kernel.

    Returns dict with total/exposed/hidden comm in ms, or None if no NCCL kernels.
    """
    try:
        kineto_results = prof.profiler.kineto_results
        if kineto_results is None:
            log.warning("Exposed comm: kineto_results is None")
            return None
        events = kineto_results.events()
    except (AttributeError, RuntimeError) as e:
        log.warning(f"Exposed comm: could not access kineto events: {e}")
        return None

    from torch.autograd import DeviceType

    nccl_intervals: list[tuple[int, int]] = []
    compute_intervals: list[tuple[int, int]] = []

    for evt in events:
        if evt.device_type() != DeviceType.CUDA:
            continue
        start_ns = evt.start_ns()
        dur_ns = evt.duration_ns()
        if dur_ns <= 0:
            continue
        end_ns = start_ns + dur_ns
        name = evt.name()
        if _is_comm_kernel(name):
            nccl_intervals.append((start_ns, end_ns))
        else:
            compute_intervals.append((start_ns, end_ns))

    if not nccl_intervals:
        return None

    merged_compute = _merge_intervals(compute_intervals)

    total_nccl_ns = 0
    exposed_ns = 0
    for start, end in nccl_intervals:
        total_nccl_ns += end - start
        exposed_ns += _exposed_duration(start, end, merged_compute)

    hidden_ns = total_nccl_ns - exposed_ns

    return {
        "total_comm_ms": total_nccl_ns / 1e6,
        "exposed_comm_ms": exposed_ns / 1e6,
        "hidden_comm_ms": hidden_ns / 1e6,
        "num_nccl_kernels": len(nccl_intervals),
    }
