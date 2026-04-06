"""Root configuration for training runs.

Defines the config tree (TrainConfig and its sub-configs), the DataConfig
type alias, and parse_config() for CLI parsing via tyro.
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Literal

import torch
import tyro

from nanigpt.data.synthetic import SyntheticData
from nanigpt.data.tokenized import TokenizedData
from nanigpt.models.dense_transformer import TransformerConfig
from nanigpt.profiling.torch_profiler import TorchProfiler

DataConfig = (
    Annotated[SyntheticData.Config, tyro.conf.subcommand(name="synthetic")]
    | Annotated[TokenizedData.Config, tyro.conf.subcommand(name="tokenized")]
)


@dataclass(kw_only=True, slots=True)
class TrainingConfig:
    """Training loop hyperparameters."""

    batch_size: int = 32
    lr: float = 3e-4
    num_steps: int = 200
    device: str = "cuda"
    dtype: Literal["float32", "float16", "bfloat16"] = "bfloat16"
    seed: int = 42

    @property
    def torch_dtype(self) -> torch.dtype:
        return {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[
            self.dtype
        ]

    def __post_init__(self):
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.num_steps <= 0:
            raise ValueError(f"num_steps must be positive, got {self.num_steps}")
        if self.lr <= 0:
            raise ValueError(f"lr must be positive, got {self.lr}")


@dataclass(kw_only=True, slots=True)
class EvalConfig:
    """Validation evaluation settings."""

    val_interval: int = 50
    """Evaluate every N training steps."""

    val_steps: int = 20
    """Number of batches to average for validation loss."""

    def __post_init__(self):
        if self.val_interval <= 0:
            raise ValueError(f"val_interval must be positive, got {self.val_interval}")
        if self.val_steps <= 0:
            raise ValueError(f"val_steps must be positive, got {self.val_steps}")


@dataclass(kw_only=True, slots=True)
class LoggingConfig:
    """Logging and experiment tracking."""

    log_interval: int = 10
    """Print training metrics every N steps."""

    wandb_project: str = "nanigpt"
    """Weights & Biases project name."""

    def __post_init__(self):
        if self.log_interval <= 0:
            raise ValueError(f"log_interval must be positive, got {self.log_interval}")


@dataclass(kw_only=True, slots=True)
class ParallelConfig:
    """Parallelism settings.

    Each dimension is an independent degree. Set them individually and they
    compose via the DeviceMesh. The product of all degrees must equal
    num_workers. dp_shard=-1 (default) auto-fills with remaining ranks.

    See nanigpt/distributed/__init__.py for how these dimensions compose,
    mesh layout, and application order.
    """

    dp_replicate: int = 1
    dp_shard: int = -1
    tp_size: int = 1
    num_workers: int = 1
    backend: str = "nccl"
    comm_timing: bool = True

    def __post_init__(self):
        # Auto-fill dp_shard with remaining ranks
        if self.dp_shard == -1:
            self.dp_shard = self.num_workers // (self.dp_replicate * self.tp_size)

        # Validate product constraint
        total = self.dp_replicate * self.dp_shard * self.tp_size
        if total != self.num_workers:
            raise ValueError(
                f"dp_replicate ({self.dp_replicate}) × dp_shard ({self.dp_shard}) "
                f"× tp_size ({self.tp_size}) = {total}, "
                f"but num_workers = {self.num_workers}"
            )


@dataclass(kw_only=True, slots=True)
class TrainConfig:
    """Root configuration for a training run."""

    model: TransformerConfig = field(default_factory=TransformerConfig)
    data: DataConfig = field(default_factory=SyntheticData.Config)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    profiler: TorchProfiler.Config = field(default_factory=TorchProfiler.Config)
    parallel: ParallelConfig = field(default_factory=ParallelConfig)

    def validate(self) -> None:
        """Run preflight checks that catch problems before heavy initialization.

        Called once on the launcher process before spawning workers, so errors
        surface immediately instead of minutes later inside a Ray actor.
        """
        errors: list[str] = []

        # Check data files exist for tokenized data configs.
        # Resolve to absolute path so Ray workers (which run from /tmp/ray/...)
        # can find the files regardless of working directory.
        if isinstance(self.data, TokenizedData.Config):
            data_path = Path(self.data.data_dir).resolve()
            self.data.data_dir = str(data_path)
            for split in ("train", "val"):
                bin_path = data_path / f"{split}.bin"
                if not bin_path.exists():
                    errors.append(
                        f"Data file not found: {bin_path}\n"
                        f"  Run: uv run python -m nanigpt.data.prepare "
                        f"--num-tokens 1_000_000 --output-dir {self.data.data_dir}"
                    )

        # Check GPU availability (only on the local node — multi-host
        # setups may have GPUs elsewhere that we can't see here)
        if self.training.device == "cuda" and not torch.cuda.is_available():
            errors.append("Config requires CUDA but no GPU is available")

        if errors:
            raise RuntimeError(
                "Config validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
            )


def parse_config(args: list[str] | None = None) -> TrainConfig:
    """Parse config from CLI args.

    Usage:
        # Use a registry preset:
        python -m nanigpt.train --config small-fineweb

        # Override specific fields:
        python -m nanigpt.train --config small-fineweb --training.lr 1e-4

        # No preset — all defaults, override what you want:
        python -m nanigpt.train --training.num-steps 500 --model.d-model 512
    """
    import tyro

    if args is None:
        args = sys.argv[1:]

    # Extract --config if present (handled before tyro sees the args)
    config_name = None
    remaining = []
    i = 0
    while i < len(args):
        if args[i] == "--config" and i + 1 < len(args):
            config_name = args[i + 1]
            i += 2
        else:
            remaining.append(args[i])
            i += 1

    # Load base config from registry or use defaults
    if config_name is not None:
        from nanigpt.configs.registry import REGISTRY

        if config_name not in REGISTRY:
            available = ", ".join(sorted(REGISTRY.keys()))
            raise ValueError(f"Unknown config '{config_name}'. Available: {available}")
        base_config = REGISTRY[config_name]()
    else:
        base_config = TrainConfig()

    # tyro applies CLI overrides on top of the base config
    return tyro.cli(TrainConfig, args=remaining, default=base_config)
