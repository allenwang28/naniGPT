"""Parallelism plan: declares *what* gets parallelized *how*.

The plan is a data description — it doesn't do any work. Mesh creation,
model wrapping, and weight sharding read the plan to decide what to do.

Current: DDP and FSDP only (dp_strategy).
Future: tp_size, pp_size, ep_size, cp_size, sequence_parallel, tp_plan, expert_plan.
"""

from dataclasses import dataclass
from typing import Literal


@dataclass(kw_only=True, slots=True)
class ParallelPlan:
    """Declares the parallelism strategy for a training run.

    For now this only covers data parallelism. As TP/PP/EP are added,
    the plan grows to include tp_size, pp_size, ep_size, and sub-plans
    like TensorParallelPlan and ExpertParallelPlan.
    """

    dp_strategy: Literal["ddp", "fsdp"] = "fsdp"
    """Data parallelism strategy."""
