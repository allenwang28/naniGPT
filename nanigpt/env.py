"""Centralized environment variable declarations.

Adapted from forge's env.py. Each env var is declared once with its name,
default, and description. Type conversion is automatic based on the default.

    from nanigpt.env import SPMD_CHECKS
    if SPMD_CHECKS.get_value():
        ...
"""

import os
from dataclasses import dataclass
from typing import Any


@dataclass
class EnvVar:
    """Typed environment variable with default and description."""

    name: str
    default: Any
    description: str

    def get_value(self) -> Any:
        """Read from os.environ, auto-converting to the default's type.

        bool default: "true", "1", "yes" (case-insensitive) → True
        int default:  int(value)
        float default: float(value)
        other/None:   raw string
        """
        value = os.environ.get(self.name)

        if value is None:
            return self.default

        if isinstance(self.default, bool):
            return value.lower() in ("true", "1", "yes")
        elif isinstance(self.default, int):
            return int(value)
        elif isinstance(self.default, float):
            return float(value)
        else:
            return value

    def set_value(self, value: Any) -> None:
        """Write to os.environ."""
        os.environ[self.name] = str(value)

    def set_default(self, value: Any) -> None:
        """Set in os.environ only if not already set."""
        os.environ.setdefault(self.name, str(value))


# ---- Environment variable declarations ----

MASTER_ADDR = EnvVar(
    name="MASTER_ADDR",
    default="localhost",
    description="Address of rank 0 for distributed rendezvous.",
)

MASTER_PORT = EnvVar(
    name="MASTER_PORT",
    default="29500",
    description="Port of rank 0 for distributed rendezvous.",
)

SPMD_CHECKS = EnvVar(
    name="NANIGPT_SPMD_CHECKS",
    default=False,
    description="Enable runtime SPMD type verification (broadcasts rank 0's tensor and compares). Adds sync points — use for debugging, not training.",
)


def all_env_vars() -> list[EnvVar]:
    """Return all registered EnvVar instances in this module."""
    return [v for v in globals().values() if isinstance(v, EnvVar)]
