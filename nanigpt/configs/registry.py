"""Experiment preset registry.

Each function returns a complete TrainConfig. Use --config <name> on the CLI
to select a preset, then override individual fields with tyro flags.
"""

from collections.abc import Callable

from nanigpt.config import EvalConfig, ParallelConfig, TrainConfig, TrainingConfig
from nanigpt.data.synthetic import SyntheticData
from nanigpt.data.tokenized import TokenizedData
from nanigpt.models.dense_transformer import MODEL_PRESETS

# ---- Base configs ----


def small_synthetic() -> TrainConfig:
    """Quick iteration config: small model, synthetic data, 200 steps."""
    return TrainConfig(
        model=MODEL_PRESETS["small"],
        data=SyntheticData.Config(seq_len=256, distribution="zipf"),
        training=TrainingConfig(batch_size=32, lr=3e-4, num_steps=200),
    )


def small_fineweb() -> TrainConfig:
    """Small model on real data for loss curve validation."""
    return TrainConfig(
        model=MODEL_PRESETS["small"],
        data=TokenizedData.Config(data_dir="data/fineweb-1M", seq_len=256),
        training=TrainingConfig(batch_size=32, lr=3e-4, num_steps=200),
        eval=EvalConfig(val_interval=50, val_steps=20),
    )


# ---- Variants modify a base ----


def small_synthetic_long() -> TrainConfig:
    """Extended synthetic run for convergence testing."""
    config = small_synthetic()
    config.training.num_steps = 2000
    config.logging.log_interval = 100
    return config


def medium_fineweb() -> TrainConfig:
    """Medium model on real data."""
    config = small_fineweb()
    config.model = MODEL_PRESETS["medium"]
    config.data.seq_len = 512
    config.training.num_steps = 2000
    return config


def small_synthetic_ddp() -> TrainConfig:
    """Small synthetic with DDP on 2 GPUs."""
    config = small_synthetic()
    config.parallel = ParallelConfig(plan="ddp", num_workers=2)
    return config


def small_synthetic_fsdp() -> TrainConfig:
    """Small synthetic with FSDP on 2 GPUs."""
    config = small_synthetic()
    config.parallel = ParallelConfig(plan="fsdp", num_workers=2)
    return config


def medium_fineweb_fsdp() -> TrainConfig:
    """Medium model on real data with FSDP on 8 GPUs."""
    config = medium_fineweb()
    config.parallel = ParallelConfig(plan="fsdp", num_workers=8)
    return config


# ---- Registry map ----

REGISTRY: dict[str, Callable[[], TrainConfig]] = {
    "small-synthetic": small_synthetic,
    "small-fineweb": small_fineweb,
    "small-synthetic-long": small_synthetic_long,
    "medium-fineweb": medium_fineweb,
    "medium-fineweb-fsdp": medium_fineweb_fsdp,
    "small-synthetic-ddp": small_synthetic_ddp,
    "small-synthetic-fsdp": small_synthetic_fsdp,
}
