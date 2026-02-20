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
