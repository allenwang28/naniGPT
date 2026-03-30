"""Communication primitives: autograd.Function conjugate pairs.

Each class pairs a forward collective with its mathematically correct
backward dual. These are the atoms that TP, EP, and PP build on.

Adapted from Megatron's mappings.py (megatron/core/tensor_parallel/mappings.py,
commit 32efeffd). The original names were _CopyToModelParallelRegion,
_ReduceFromModelParallelRegion, etc. Renamed here for readability — see the
mapping below.

    | Megatron name                    | Our name                         |
    |----------------------------------|----------------------------------|
    | _CopyToModelParallelRegion       | _EnterParallelRegion             |
    | _ReduceFromModelParallelRegion   | _ExitParallelRegion              |
    | _GatherFromModelParallelRegion   | _GatherIntoParallelRegion        |
    | _ReduceScatterToModelParallelReg | _ScatterFromParallelRegion       |

    | Class                       | Forward        | Backward       |
    |-----------------------------|----------------|----------------|
    | EnterParallelRegion         | identity       | all-reduce     |
    | ExitParallelRegion          | all-reduce     | identity       |
    | GatherIntoParallelRegion    | all-gather     | reduce-scatter |
    | ScatterFromParallelRegion   | reduce-scatter | all-gather     |

SPMD type transitions (see spmd.py):
    Enter:   I@tp → R@tp  (or R@tp → R@tp)
    Exit:    P@tp → I@tp
    Gather:  V@tp → R@tp
    Scatter: P@tp → V@tp

The first pair (Enter/Exit) is for basic TP without sequence parallelism.
The second pair (Gather/Scatter) replaces them when sequence parallelism
is enabled — same communication volume, but activations between TP regions
are sharded on the sequence dim, saving memory.
"""

import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed import ProcessGroup

# ---- Conjugate pair 1: Enter / Exit ----
# Used for basic TP without sequence parallelism.
# Column-parallel uses Enter before the matmul (identity fwd, all-reduce bwd).
# Row-parallel uses Exit after the matmul (all-reduce fwd, identity bwd).


class _EnterParallelRegion(torch.autograd.Function):
    """Forward: identity. Backward: all-reduce. SPMD: I@tp → R@tp.

    Wraps the input before a column-parallel matmul (see tensor_parallel.py).

    Forward is identity because the input is already replicated — each rank
    has the full x and just multiplies by its local weight shard W_i.

    Backward needs all-reduce because each rank computed grad_x = grad_y @ W_i,
    which is only a partial gradient (rank 0 doesn't know rank 1's weights).
    The true grad_x = sum(grad_y @ W_i) across all ranks.
    """

    @staticmethod
    def forward(ctx, input_: Tensor, group: ProcessGroup) -> Tensor:
        ctx.group = group
        return input_

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> tuple[Tensor, None]:
        dist.all_reduce(grad_output, group=ctx.group)
        return grad_output, None


class _ExitParallelRegion(torch.autograd.Function):
    """Forward: all-reduce. Backward: identity. SPMD: P@tp → I@tp.

    Wraps the output after a row-parallel matmul (see tensor_parallel.py).

    Forward needs all-reduce because each rank computed z_i = y_i @ W_row_i^T,
    which is a partial sum — the true output is z = z_0 + z_1 + ... but no
    single rank has it yet. The all-reduce sums them.

    Backward is identity because the upstream gradient flows back through each
    rank's own rows of W — no cross-rank information needed.
    """

    @staticmethod
    def forward(ctx, input_: Tensor, group: ProcessGroup) -> Tensor:
        ctx.group = group
        dist.all_reduce(input_, group=group)
        return input_

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> tuple[Tensor, None]:
        return grad_output, None


# ---- Conjugate pair 2: Gather / Scatter ----
# Used for TP with sequence parallelism. More bandwidth-efficient than
# Enter/Exit because they move smaller tensors.


class _GatherIntoParallelRegion(torch.autograd.Function):
    """Forward: all-gather (dim 0). Backward: reduce-scatter (dim 0). SPMD: V@tp → R@tp.

    Sequence-parallel replacement for Enter. Each rank holds a different
    chunk of the sequence; the all-gather reconstructs the full sequence
    before the column-parallel matmul.

    Backward uses reduce-scatter (not all-reduce) because the gradient is
    partial (each rank computed with its weight shard) AND we only need each
    rank's sequence chunk back — reduce-scatter does both in one op.
    """

    @staticmethod
    def forward(ctx, input_: Tensor, group: ProcessGroup) -> Tensor:
        ctx.group = group
        ctx.input_shape = input_.shape

        world_size = dist.get_world_size(group)
        if world_size == 1:
            return input_

        # All-gather along dim 0
        gathered = [torch.empty_like(input_) for _ in range(world_size)]
        dist.all_gather(gathered, input_.contiguous(), group=group)
        return torch.cat(gathered, dim=0)

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> tuple[Tensor, None]:
        group = ctx.group
        world_size = dist.get_world_size(group)
        if world_size == 1:
            return grad_output, None

        # Reduce-scatter along dim 0
        chunk_size = grad_output.shape[0] // world_size
        input_grad = torch.empty(
            chunk_size,
            *grad_output.shape[1:],
            dtype=grad_output.dtype,
            device=grad_output.device,
        )
        dist.reduce_scatter_tensor(input_grad, grad_output.contiguous(), group=group)
        return input_grad, None


class _ScatterFromParallelRegion(torch.autograd.Function):
    """Forward: reduce-scatter (dim 0). Backward: all-gather (dim 0). SPMD: P@tp → V@tp.

    Sequence-parallel replacement for Exit. Each rank has a partial sum
    from the row-parallel matmul (same as Exit), but instead of
    all-reducing to give every rank the full result, reduce-scatter sums AND
    splits — each rank gets only its sequence chunk. This saves activation
    memory between TP regions.

    Backward uses all-gather to reconstruct the full gradient before it flows
    back into the row-parallel matmul.
    """

    @staticmethod
    def forward(ctx, input_: Tensor, group: ProcessGroup) -> Tensor:
        ctx.group = group
        ctx.input_shape = input_.shape

        world_size = dist.get_world_size(group)
        if world_size == 1:
            return input_

        # Reduce-scatter along dim 0
        chunk_size = input_.shape[0] // world_size
        output = torch.empty(
            chunk_size,
            *input_.shape[1:],
            dtype=input_.dtype,
            device=input_.device,
        )
        dist.reduce_scatter_tensor(output, input_.contiguous(), group=group)
        return output

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> tuple[Tensor, None]:
        group = ctx.group
        world_size = dist.get_world_size(group)
        if world_size == 1:
            return grad_output, None

        # All-gather along dim 0
        gathered = [torch.empty_like(grad_output) for _ in range(world_size)]
        dist.all_gather(gathered, grad_output.contiguous(), group=group)
        return torch.cat(gathered, dim=0), None


# ---- Convenience wrappers ----
# These hide the .apply() call and make the SPMD type transitions readable.


def enter_parallel_region(input_: Tensor, group: ProcessGroup) -> Tensor:
    """Identity forward, all-reduce backward. SPMD: I@tp → R@tp."""
    return _EnterParallelRegion.apply(input_, group)


def exit_parallel_region(input_: Tensor, group: ProcessGroup) -> Tensor:
    """All-reduce forward, identity backward. SPMD: P@tp → I@tp."""
    return _ExitParallelRegion.apply(input_, group)


def gather_into_parallel_region(input_: Tensor, group: ProcessGroup) -> Tensor:
    """All-gather forward (dim 0), reduce-scatter backward. SPMD: V@tp → R@tp."""
    return _GatherIntoParallelRegion.apply(input_, group)


def scatter_from_parallel_region(input_: Tensor, group: ProcessGroup) -> Tensor:
    """Reduce-scatter forward (dim 0), all-gather backward. SPMD: P@tp → V@tp."""
    return _ScatterFromParallelRegion.apply(input_, group)
