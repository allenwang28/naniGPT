"""Parallelism plan: declares *what* gets parallelized *how*.

The plan is a data description — it doesn't do any work. Mesh creation,
model wrapping, and weight sharding read the plan to decide what to do.

Each dimension is an independent degree. The product of all degrees must
equal world_size. Dimensions with degree 1 are inactive (no communication).

    | Dimension     | What it does                        | Communication     |
    |---------------|-------------------------------------|-------------------|
    | dp_replicate  | Gradient all-reduce (DDP)           | all-reduce        |
    | dp_shard      | Parameter sharding (FSDP)           | all-gather + RS   |
    | tp            | Weight sharding within a layer      | all-reduce per layer |
    | pp            | (future) Pipeline stages            | point-to-point    |
    | ep            | (future) Expert routing             | all-to-all        |
    | cp            | (future) Context/sequence parallel  | ring attention    |

Composition examples (8 GPUs):
    dp_shard=8                           → pure FSDP
    dp_replicate=8                       → pure DDP
    dp_replicate=4, dp_shard=2           → HSDP
    dp_shard=4, tp=2                     → FSDP + TP
    dp_replicate=2, dp_shard=2, tp=2     → HSDP + TP

Future: pp_size, ep_size, cp_size.
"""

from dataclasses import dataclass


@dataclass(kw_only=True, slots=True)
class ParallelPlan:
    """Declares the parallelism strategy for a training run.

    dp_replicate: DDP degree (gradient all-reduce). 1 = disabled.
    dp_shard: FSDP degree (parameter sharding). 1 = disabled.
    tp_size: Tensor parallel degree. Must divide n_heads and d_ff. 1 = disabled.
    """

    dp_replicate: int = 1
    dp_shard: int = 1
    tp_size: int = 1

    def degree_for(self, mesh_dim: str) -> int:
        """Return the parallelism degree for a named mesh dimension.

        Maps mesh dimension names (as used in DeviceMesh and ModuleSharding)
        to the corresponding plan field. Unknown dimensions return 1.
        """
        _MESH_DIM_TO_FIELD = {
            "tp": self.tp_size,
            "dp_shard": self.dp_shard,
            "dp_replicate": self.dp_replicate,
        }
        return _MESH_DIM_TO_FIELD.get(mesh_dim, 1)
