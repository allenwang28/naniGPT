"""Custom process group: wraps a real ProcessGroup for DeviceMesh integration.

NaniProcessGroup wraps a real ProcessGroup (typically NCCL) so that
DeviceMesh.from_group() can accept our groups. The Python group_name
property is required because C++ pybind11 properties don't dispatch to
Python subclasses (same pattern as torchft's ProcessGroup).

When comm timing is enabled, CUDA events are recorded around each
collective and stashed into StepMetrics for resolution at step boundary.

Future: fault tolerance hooks (abort, errored, shutdown) are stubs
for the FT branch to fill in.
"""

import logging

import torch
from torch.distributed import ProcessGroup, Work
from torch.distributed.distributed_c10d import (
    AllgatherOptions,
    AllreduceCoalescedOptions,
    AllreduceOptions,
    AllToAllOptions,
    BarrierOptions,
    BroadcastOptions,
    ReduceScatterOptions,
)

from nanigpt.profiling.event_types import MESH_DIM_TO_EVENT, EventType
from nanigpt.profiling.timer import get_global_metrics

_log = logging.getLogger(__name__)


class NaniProcessGroup(ProcessGroup):
    """ProcessGroup wrapper for DeviceMesh integration and future FT.

    Delegates all collectives to the inner ProcessGroup. Overrides
    group_name as a Python property so DeviceMesh.from_group() can
    read it correctly.

    When comm_timing is enabled, records CUDA events around each
    collective dispatch for per-dimension communication breakdown.

    Args:
        pg: The real ProcessGroup to delegate to.
        mesh_dim: Mesh dimension name ("tp", "dp_shard", "dp_replicate").
    """

    def __init__(self, pg: ProcessGroup, mesh_dim: str) -> None:
        super().__init__(pg.rank(), pg.size())
        self._pg = pg
        self._mesh_dim = mesh_dim
        self._group_name: str = pg.group_name
        self._comm_timing = False
        self._event_type: EventType = MESH_DIM_TO_EVENT.get(mesh_dim, EventType.COMMUNICATION)

        # Copy the NCCL backend registration from the inner PG so that
        # code resolving backends by device type (e.g. FSDP) works.
        cuda_device = torch.device("cuda")
        backend = pg._get_backend(cuda_device)
        self._register_backend(cuda_device, ProcessGroup.BackendType.NCCL, backend)

    @property  # type: ignore[override]
    def group_name(self) -> str:
        return self._group_name

    def enable_comm_timing(self) -> None:
        self._comm_timing = True

    def disable_comm_timing(self) -> None:
        self._comm_timing = False

    def _timed(self, fn, *args) -> Work:
        """Record CUDA events around a collective if timing is enabled."""
        if not self._comm_timing:
            return fn(*args)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        work = fn(*args)
        end.record()
        get_global_metrics().record_events(self._event_type, start, end)
        return work

    # ---- Collective overrides ----

    def allreduce(
        self,
        tensors: list[torch.Tensor],
        opts: AllreduceOptions,
    ) -> Work:
        return self._timed(self._pg.allreduce, tensors, opts)

    def allreduce_coalesced(
        self,
        tensors: list[torch.Tensor],
        opts: AllreduceCoalescedOptions,
    ) -> Work:
        return self._timed(self._pg.allreduce_coalesced, tensors, opts)

    def allgather(
        self,
        output_tensors: list[list[torch.Tensor]],
        input_tensor: list[torch.Tensor],
        opts: AllgatherOptions,
    ) -> Work:
        return self._timed(self._pg.allgather, output_tensors, input_tensor, opts)

    # _allgather_base and _reduce_scatter_base are the C++ methods that
    # dist.all_gather_into_tensor() and dist.reduce_scatter_tensor() dispatch
    # to. FSDP calls these directly, not the Python-level allgather().
    def _allgather_base(  # type: ignore[override]
        self,
        output: torch.Tensor,
        input: torch.Tensor,
        opts: AllgatherOptions | None = None,
    ) -> Work:
        if opts is None:
            opts = AllgatherOptions()
        return self._timed(self._pg._allgather_base, output, input, opts)

    def allgather_into_tensor_coalesced(
        self,
        output_tensors: list[torch.Tensor],
        input_tensors: list[torch.Tensor],
        opts: AllgatherOptions,
    ) -> Work:
        return self._timed(
            self._pg.allgather_into_tensor_coalesced, output_tensors, input_tensors, opts
        )

    def reduce_scatter(
        self,
        output_tensors: list[torch.Tensor],
        input_tensors: list[list[torch.Tensor]],
        opts: ReduceScatterOptions,
    ) -> Work:
        return self._timed(self._pg.reduce_scatter, output_tensors, input_tensors, opts)

    def _reduce_scatter_base(  # type: ignore[override]
        self,
        output: torch.Tensor,
        input: torch.Tensor,
        opts: ReduceScatterOptions | None = None,
    ) -> Work:
        if opts is None:
            opts = ReduceScatterOptions()
        return self._timed(self._pg._reduce_scatter_base, output, input, opts)

    def reduce_scatter_tensor_coalesced(
        self,
        output_tensors: list[torch.Tensor],
        input_tensors: list[torch.Tensor],
        opts: ReduceScatterOptions,
    ) -> Work:
        return self._timed(
            self._pg.reduce_scatter_tensor_coalesced, output_tensors, input_tensors, opts
        )

    def broadcast(
        self,
        tensor_list: list[torch.Tensor],
        opts: BroadcastOptions,
    ) -> Work:
        return self._timed(self._pg.broadcast, tensor_list, opts)

    def alltoall_base(
        self,
        output_buffer: torch.Tensor,
        input_buffer: torch.Tensor,
        output_split_sizes: list[int],
        input_split_sizes: list[int],
        opts: AllToAllOptions,
    ) -> Work:
        return self._timed(
            self._pg.alltoall_base,
            output_buffer,
            input_buffer,
            output_split_sizes,
            input_split_sizes,
            opts,
        )

    def barrier(self, opts: BarrierOptions) -> Work:
        return self._pg.barrier(opts)

    def send(self, tensors: list[torch.Tensor], dst_rank: int, tag: int) -> Work:
        return self._pg.send(tensors, dst_rank, tag)

    def recv(self, tensors: list[torch.Tensor], src_rank: int, tag: int) -> Work:
        return self._pg.recv(tensors, src_rank, tag)

    def size(self) -> int:
        return self._pg.size()

    def getBackendName(self) -> str:
        return self._pg.getBackendName()

    # ---- Fault tolerance stubs ----

    def abort(self) -> None:
        """Abort in-flight operations. Stub for fault tolerance."""
        pass

    def errored(self) -> Exception | None:
        """Whether an async error occurred that requires reconfiguration."""
        return None

    def shutdown(self) -> None:
        """Graceful teardown. Stub for fault tolerance."""
        pass

    def __repr__(self) -> str:
        return f"NaniProcessGroup(mesh_dim={self._mesh_dim!r}, pg={self._pg})"
