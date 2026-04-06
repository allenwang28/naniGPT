"""Enum of training phases used as keys for timing regions.

Each EventType maps to a string value used in logs and StepMetrics history.
Add new members here as the training loop grows (e.g. COMMUNICATION, EVAL).
No _START/_END variants — those are derived by the measure context manager.
"""

from enum import StrEnum


class EventType(StrEnum):
    STEP = "step"
    DATA = "data"
    FORWARD = "forward"
    BACKWARD = "backward"
    OPTIMIZER = "optimizer"
    EVAL = "eval"
    COMMUNICATION = "communication"

    # Per-parallelism communication breakdown
    TP_COMM = "tp_comm"
    FSDP_COMM = "fsdp_comm"
    DP_COMM = "dp_comm"


# Map mesh dimension names to comm event types
MESH_DIM_TO_EVENT: dict[str, EventType] = {
    "tp": EventType.TP_COMM,
    "dp_shard": EventType.FSDP_COMM,
    "dp_replicate": EventType.DP_COMM,
}

# All communication event types — used to separate comm volume from step time
COMM_EVENTS: frozenset[str] = frozenset(
    e.value
    for e in (
        EventType.TP_COMM,
        EventType.FSDP_COMM,
        EventType.DP_COMM,
        EventType.COMMUNICATION,
    )
)
