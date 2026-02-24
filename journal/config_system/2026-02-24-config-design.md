# Config System Design Spec

**Date:** 2026-02-24
**Context:** The [config system research entry](config_system/2026-02-20-config-system-research.md) concluded with a decision to adopt the torchtitan v2 pattern: `Configurable` base class with `build()`, registry functions for experiment definition, and `tyro` for CLI generation. This entry makes that abstract decision concrete — showing exact implementations, mapping every current parameter, and presenting the open design decisions with tradeoffs.

## 1. The Configurable Base Class

This is settled from the research entry. Every component that owns runtime state inherits from `Configurable` and defines a nested `Config` dataclass. `config.build()` constructs the owning component.

```python
from dataclasses import dataclass
from typing import ClassVar


class Configurable:
    """Base class for components that are constructed from configuration.

    Subclasses define a nested Config dataclass. The Config's build() method
    constructs the owning component, so the config tree mirrors the object tree.

    Usage:
        class MyComponent(Configurable):
            @dataclass(kw_only=True, slots=True)
            class Config(Configurable.Config):
                learning_rate: float = 3e-4

            def __init__(self, config: Config):
                self.lr = config.learning_rate

        config = MyComponent.Config(learning_rate=1e-4)
        component = config.build()  # returns MyComponent(config)
    """

    @dataclass(kw_only=True, slots=True)
    class Config:
        _owner: ClassVar[type | None] = None

        def build(self, **kwargs):
            """Construct the owning component from this config."""
            if self._owner is None:
                raise TypeError(
                    f"{type(self).__name__} has no owner class — "
                    f"it must be defined as a nested class inside a Configurable subclass"
                )
            return self._owner(config=self, **kwargs)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if "Config" in cls.__dict__:
            config_cls = cls.__dict__["Config"]
            if not hasattr(config_cls, "__slots__"):
                raise TypeError(
                    f"{cls.__name__}.Config must use @dataclass(kw_only=True, slots=True)"
                )
            config_cls._owner = cls
```

The `__init_subclass__` hook enforces two things at class definition time (not runtime): every `Config` uses `slots=True` (for typo detection — setting an unknown attribute raises `AttributeError`), and the `_owner` backlink is set automatically so `build()` knows what to construct.

## 2. Config Tree — The Full Hierarchy

Every current constant and parameter mapped to a typed field in a two-level dataclass tree:

```
TrainConfig (root — passed to tyro, logged to wandb)
├── model: TransformerConfig
│   ├── vocab_size: int = 50257
│   ├── d_model: int = 768
│   ├── n_heads: int = 12
│   ├── n_layers: int = 12
│   ├── d_ff: int = 3072
│   ├── max_seq_len: int = 1024
│   └── dropout: float = 0.0
├── data: DataConfig (= SyntheticData.Config | TokenizedData.Config)  [Configurable]
│   │  SyntheticData.Config → build() returns SyntheticData:
│   ├── seq_len: int = 256
│   ├── distribution: str = "zipf"
│   ├── zipf_exponent: float = 1.0
│   └── num_workers: int = 2
│   │  TokenizedData.Config → build() returns TokenizedData:
│   ├── data_dir: str = "data/fineweb-1M"
│   ├── seq_len: int = 256
│   └── num_workers: int = 2
├── training: TrainingConfig
│   ├── batch_size: int = 32
│   ├── lr: float = 3e-4
│   ├── num_steps: int = 200
│   ├── device: str = "cuda"
│   ├── dtype: str = "bfloat16"
│   └── seed: int = 42
├── eval: EvalConfig
│   ├── val_interval: int = 50
│   └── val_steps: int = 20
├── logging: LoggingConfig
│   ├── log_interval: int = 10
│   └── wandb_project: str = "nanigpt"
└── profiler: TorchProfiler.Config → build() returns TorchProfiler  [Configurable]
    ├── enabled: bool = True
    ├── start_step: int = 10
    ├── end_step: int = 12
    ├── warmup_steps: int = 1
    ├── top_n: int = 15
    ├── export_trace: bool = True
    ├── record_shapes: bool = True
    ├── with_stack: bool = False
    └── with_flops: bool = True
```

This is every knob that currently exists in the codebase, either as a module-level constant in `train.py` or as a field in the existing `TransformerConfig`/`ProfilerConfig` dataclasses. One current constant is intentionally absent: `MODEL_PRESET` (the string key into `PRESET_CONFIGS`). It's replaced by registry function selection — `--config small-synthetic` replaces `MODEL_PRESET = "small"`. The preset name is no longer a config field; it's the choice of which registry function to call.

## 3. Per-Component Config Definitions

### TransformerConfig (already exists, minimal changes)

```python
@dataclass(kw_only=True, slots=True)
class TransformerConfig:
    """Dense transformer architecture parameters."""

    vocab_size: int = 50257
    d_model: int = 768
    n_heads: int = 12
    n_layers: int = 12
    d_ff: int = 3072
    max_seq_len: int = 1024
    dropout: float = 0.0

    def __post_init__(self):
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
            )
        if self.d_ff <= 0 or self.d_model <= 0:
            raise ValueError("d_ff and d_model must be positive")
```

### Data Configs (Configurable)

Data uses a tagged union of `Configurable` components — each data source's `Config.build()` constructs its dataset. See Decision Point below for rationale on the union approach.

```python
class SyntheticData(Configurable):
    """Synthetic random token data for controlled experiments."""

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        seq_len: int = 256
        """Sequence length (tokens per sample)."""

        distribution: Literal["uniform", "zipf"] = "zipf"
        """Token distribution."""

        zipf_exponent: float = 1.0
        """Exponent for Zipfian distribution. Only used when distribution='zipf'."""

        num_workers: int = 2
        """DataLoader worker processes."""

        def __post_init__(self):
            if self.seq_len <= 0:
                raise ValueError(f"seq_len must be positive, got {self.seq_len}")
            if self.zipf_exponent <= 0:
                raise ValueError(f"zipf_exponent must be positive, got {self.zipf_exponent}")
            if self.num_workers < 0:
                raise ValueError(f"num_workers must be non-negative, got {self.num_workers}")

    def __init__(self, config: Config, *, vocab_size: int, num_samples: int):
        self.dataset = RandomTokenDataset(
            vocab_size=vocab_size, seq_len=config.seq_len,
            num_samples=num_samples, distribution=config.distribution,
            zipf_exponent=config.zipf_exponent,
        )
        self.num_workers = config.num_workers


class TokenizedData(Configurable):
    """Pre-tokenized memmap data produced by prepare.py."""

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        data_dir: str = "data/fineweb-1M"
        """Path to directory containing {split}.bin and meta.json."""

        seq_len: int = 256
        """Sequence length (tokens per sample)."""

        num_workers: int = 2
        """DataLoader worker processes."""

        def __post_init__(self):
            if self.seq_len <= 0:
                raise ValueError(f"seq_len must be positive, got {self.seq_len}")
            if not self.data_dir:
                raise ValueError("data_dir must be non-empty for TokenizedData.Config")
            if self.num_workers < 0:
                raise ValueError(f"num_workers must be non-negative, got {self.num_workers}")

    def __init__(self, config: Config, *, split: str, num_samples: int):
        self.dataset = TokenizedDataset(
            config.data_dir, split, config.seq_len, num_samples,
        )
        self.num_workers = config.num_workers


# Type alias — the "registry" of data config types.
DataConfig = SyntheticData.Config | TokenizedData.Config
```

### TrainingConfig

```python
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
        return {"float32": torch.float32, "float16": torch.float16,
                "bfloat16": torch.bfloat16}[self.dtype]

    def __post_init__(self):
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.num_steps <= 0:
            raise ValueError(f"num_steps must be positive, got {self.num_steps}")
        if self.lr <= 0:
            raise ValueError(f"lr must be positive, got {self.lr}")
```

### EvalConfig

```python
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
```

### LoggingConfig

```python
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
```

### TorchProfiler (Configurable)

`TorchProfiler` becomes a `Configurable` component. Its `Config` is a nested dataclass; `config.profiler.build()` constructs the profiler.

```python
class TorchProfiler(Configurable):
    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        """Controls the PyTorch profiler window."""

        enabled: bool = True
        start_step: int = 10
        end_step: int = 12
        warmup_steps: int = 1
        top_n: int = 15
        export_trace: bool = True
        record_shapes: bool = True
        with_stack: bool = False
        with_flops: bool = True

        def __post_init__(self):
            if self.end_step <= self.start_step:
                raise ValueError(
                    f"end_step ({self.end_step}) must be > start_step ({self.start_step})"
                )

    def __init__(self, config: Config):
        # ... existing TorchProfiler.__init__ logic, unchanged
```

### TrainConfig (root)

```python
@dataclass(kw_only=True, slots=True)
class TrainConfig:
    """Root configuration for a training run."""

    model: TransformerConfig = field(default_factory=TransformerConfig)
    data: DataConfig = field(default_factory=SyntheticData.Config)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    profiler: TorchProfiler.Config = field(default_factory=TorchProfiler.Config)
```

## 4. Decision Points

### Decision: Model config pattern

The model is the most architecturally sensitive component. As we add MoE and potentially other architectures, how should model config interact with model construction?

**Option A: Plain dataclass + model_registry (torchtitan v2 style)**

`TransformerConfig` stays a plain dataclass. A `model_registry` maps preset names to config instances. The training loop calls `DenseTransformer(config)` directly.

```python
# models/dense_transformer.py
@dataclass(kw_only=True, slots=True)
class TransformerConfig:
    vocab_size: int = 50257
    d_model: int = 768
    # ...

MODEL_PRESETS: dict[str, TransformerConfig] = {
    "small": TransformerConfig(d_model=256, n_heads=8, n_layers=8, d_ff=1024, max_seq_len=512),
    "medium": TransformerConfig(d_model=512, n_heads=8, n_layers=12, d_ff=2048),
    "large": TransformerConfig(d_model=768, n_heads=12, n_layers=12, d_ff=3072),
}

# In registry function:
def small_synthetic() -> TrainConfig:
    return TrainConfig(model=MODEL_PRESETS["small"])

# In train.py — the training loop picks the model class:
model = DenseTransformer(config.model).to(config.training.device)
```

When MoE arrives, a separate `MoEConfig` dataclass and `MoETransformer` class get their own presets. The registry function picks which class to use:

```python
def small_moe() -> TrainConfig:
    return TrainConfig(
        model=MoEConfig(d_model=256, n_experts=8, ...),
    )

# train.py needs to dispatch:
if isinstance(config.model, MoEConfig):
    model = MoETransformer(config.model)
elif isinstance(config.model, TransformerConfig):
    model = DenseTransformer(config.model)
```

**Option B: Configurable + nn.Module multiple inheritance (AXLearn style)**

`DenseTransformer` inherits from both `nn.Module` and `Configurable`. Its `Config` knows how to `build()` itself into the right model.

```python
class DenseTransformer(nn.Module, Configurable):
    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        vocab_size: int = 50257
        d_model: int = 768
        # ...

    def __init__(self, config: Config):
        nn.Module.__init__(self)
        # ... build layers from config

# In registry:
def small_synthetic() -> TrainConfig:
    return TrainConfig(model=DenseTransformer.Config(d_model=256, ...))

# In train.py — model construction is polymorphic:
model = config.model.build().to(config.training.device)
```

When MoE arrives, `MoETransformer` defines its own `Config(Configurable.Config)` and the registry function just uses that config. `train.py` doesn't change — `config.model.build()` dispatches automatically via `_owner`.

**Tradeoffs:**

| Dimension | Option A (plain dataclass) | Option B (Configurable model) |
|-----------|---------------------------|-------------------------------|
| Simplicity now | Simpler — models stay as plain `nn.Module`, no multiple inheritance | Adds `Configurable` to model classes |
| Adding architectures | Requires `isinstance` dispatch in train.py | Polymorphic — train.py doesn't change |
| Type narrowing | `config.model` has a concrete type, IDE knows the fields | `config.model` typed as `Configurable.Config` base — fields require type narrowing to access architecture-specific params |
| Mental model | Config and model are separate things | Config *is* the model blueprint |
| MRO complexity | None | `nn.Module` + `Configurable` MRO is manageable but adds a concept |
| tyro CLI | Straightforward — one model config type per registry function | Works identically — tyro sees the concrete `Config` subclass |

**Recommendation:** Option A for now. We have one model architecture. The `isinstance` dispatch cost is zero with one branch, and we avoid the MRO complexity. When MoE lands and we have two architectures, the dispatch in train.py is a 3-line `if/elif` — not worth adding multiple inheritance to avoid. If we reach three+ architectures, revisit.

---

### Decision: Data config representation

Synthetic data and tokenized data have different parameters (`distribution`/`zipf_exponent` vs `data_dir`). How should DataConfig handle this?

**Option A: Single DataConfig with optional fields**

One dataclass with all fields. Synthetic-only fields are ignored when `data_dir` is set.

```python
@dataclass(kw_only=True, slots=True)
class DataConfig:
    data_dir: str = ""
    seq_len: int = 256
    distribution: Literal["uniform", "zipf"] = "zipf"
    zipf_exponent: float = 1.0
    num_workers: int = 2

    @property
    def is_synthetic(self) -> bool:
        return not self.data_dir
```

**Option B: Tagged union / subclass configs**

Separate dataclasses for each data source. A `DataConfig` type alias collects all data config types into a union; `TrainConfig` uses the alias so it never changes when new sources are added.

```python
@dataclass(kw_only=True, slots=True)
class SyntheticDataConfig:
    seq_len: int = 256
    distribution: Literal["uniform", "zipf"] = "zipf"
    zipf_exponent: float = 1.0
    num_workers: int = 2

@dataclass(kw_only=True, slots=True)
class TokenizedDataConfig:
    data_dir: str = "data/fineweb-1M"
    seq_len: int = 256
    num_workers: int = 2

# Type alias — the "registry" of data config types.
# Adding a new source: define the dataclass, add it here. TrainConfig doesn't change.
DataConfig = SyntheticDataConfig | TokenizedDataConfig

# In TrainConfig:
data: DataConfig = field(default_factory=SyntheticDataConfig)
```

The type alias is preferable to a dynamic registry (e.g. a `register_data_config()` decorator that builds a union at runtime) because static type checkers and tyro both need the concrete types visible at import time. A dynamic union breaks mypy/pyright and tyro's CLI subcommand generation. The alias scales fine — even 10 data sources is a one-line definition.

**Tradeoffs:**

| Dimension | Option A (single dataclass) | Option B (tagged union) |
|-----------|----------------------------|-------------------------|
| Simplicity | One class, trivial | Two classes, union type annotation |
| Invalid states | Can set `data_dir` and `distribution` simultaneously — meaningless but harmless | Impossible to set synthetic params on tokenized config |
| CLI UX with tyro | Flat flags: `--data.data-dir`, `--data.distribution` | tyro handles unions via subcommands — `--data:synthetic` vs `--data:tokenized`. More explicit but more to type |
| Adding data sources | Add more optional fields (gets messy at 4+ sources) | Add a new dataclass (scales cleanly) |
| Current scale | 2 data sources, ~3 source-specific fields | Same |

**Recommendation:** Option B. Even with only two data sources now, new data sources are expected soon and the union approach scales cleanly — each new source is a new dataclass, no accumulation of unrelated optional fields. The tyro subcommand UX (`--data:synthetic`, `--data:tokenized`) also makes the CLI self-documenting about which data source is in use.

---

### Decision: Config ownership — nested vs co-located vs centralized

Where do config dataclasses live in the file tree?

**Option A: Co-located with owning module**

Each module defines its own config next to its implementation.

```
nanigpt/
  models/
    dense_transformer.py    → TransformerConfig, MODEL_PRESETS
  data/
    config.py               → DataConfig (or in synthetic.py / tokenized.py)
  profiling/
    torch_profiler.py       → ProfilerConfig  (already here)
  config.py                 → TrainConfig, TrainingConfig, EvalConfig, LoggingConfig
```

**Option B: All configs centralized**

Everything in one file or one package.

```
nanigpt/
  config.py                 → TrainConfig, TransformerConfig, DataConfig,
                              TrainingConfig, EvalConfig, LoggingConfig, ProfilerConfig
  models/
    dense_transformer.py    → imports TransformerConfig from config.py
  profiling/
    torch_profiler.py       → imports ProfilerConfig from config.py
```

**Option C: Hybrid — Configurable components own their Config as a nested class, plain configs centralized**

If we adopt `Configurable` for some components, their configs are nested classes (co-located by definition). Plain dataclass configs live in a shared location.

```
nanigpt/
  config.py                 → TrainConfig, TrainingConfig, EvalConfig, LoggingConfig, DataConfig
  models/
    dense_transformer.py    → TransformerConfig (plain dataclass, co-located with model)
  profiling/
    torch_profiler.py       → ProfilerConfig (plain dataclass, co-located with profiler)
```

**Tradeoffs:**

| Dimension | Option A (co-located) | Option B (centralized) | Option C (hybrid) |
|-----------|----------------------|----------------------|-------------------|
| Find a config | Look in the module that uses it | Look in `config.py` | Depends on the config — need to know the convention |
| Circular imports | Risk if modules import each other's configs | None — single source | Low risk — same as A for co-located pieces |
| Discoverability | Scattered — need to know the module | One file to read | Moderate |
| Cohesion | Config is next to the code it parameterizes | Config is separated from code it parameterizes | Best of both: infrastructure-like configs centralized, component-specific configs co-located |
| Scale | Works well — each module is self-contained | `config.py` grows large with many components | Grows moderately |

**Recommendation:** Option C. This is what torchtitan v2 does — component-specific configs live with their components (`ProfilingConfig` in `tools/profiling.py`, `CheckpointManager.Config` nested inside `CheckpointManager`), while ownerless configs (`ParallelismConfig`, `TrainingConfig`) live in a shared `configs.py`.

For us: `TransformerConfig` naturally lives in `dense_transformer.py` (it parameterizes that specific model). `ProfilerConfig` naturally lives in `torch_profiler.py` (it already does). Generic training parameters (`TrainingConfig`, `EvalConfig`, `LoggingConfig`, `DataConfig`) and the root `TrainConfig` live in `nanigpt/config.py`. This matches what we already have — we're just formalizing it and adding the missing configs.

**Convention: configs go at the top of the file, before the implementation they parameterize.** This is already the pattern — `TransformerConfig` is at the top of `dense_transformer.py` (before `DenseTransformer`), `ProfilerConfig` is at the top of `torch_profiler.py` (before `TorchProfiler`). Making this explicit: when you open a module file, the config dataclass is the first thing you see, making it immediately discoverable without scrolling.

---

### Decision: What gets Configurable vs plain dataclass

Every component that constructs runtime state gets `Configurable` from day one. The pattern should be baked in when modules are written, not retrofitted later. Migrating a plain dataclass to `Configurable` after the fact touches every call site — doing it upfront costs almost nothing.

**Gets Configurable now:**
- `TorchProfiler` — `TorchProfiler.Config` becomes a nested `Configurable.Config`. Construction via `config.profiler.build()`.
- `SyntheticDataConfig` / `TokenizedDataConfig` — Each data config's `build()` constructs the corresponding dataset. This gives us polymorphic dataset construction through the union type for free.

**Stays plain dataclass:**
- `TransformerConfig` — Per the model config decision above (Option A), model configs stay plain. Model construction is externally orchestrated (device placement, parallelism wrapping, compilation) and doesn't fit `build()` cleanly.
- `TrainingConfig`, `EvalConfig`, `LoggingConfig` — Pure parameter bags with no associated component to construct. These parameterize the training loop, not a buildable object.

**Rule going forward:** Any new module that owns runtime state (future `Trainer` class, optimizers, schedulers, checkpoint managers, etc.) uses `Configurable` from the start. If it constructs something, it gets `build()`. If it's just a bag of parameters, it stays a plain dataclass.

## 5. Registry — Experiment Functions

Registry functions replace the module-level constants in `train.py`. Each function returns a complete `TrainConfig`.

```python
# nanigpt/configs/registry.py

from nanigpt.config import TrainConfig, TrainingConfig, EvalConfig, LoggingConfig
from nanigpt.data.synthetic import SyntheticData
from nanigpt.data.tokenized import TokenizedData
from nanigpt.models.dense_transformer import TransformerConfig, MODEL_PRESETS


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
    config.training.num_steps = 1000
    return config


# ---- Registry map ----

REGISTRY: dict[str, Callable[[], TrainConfig]] = {
    "small-synthetic": small_synthetic,
    "small-fineweb": small_fineweb,
    "small-synthetic-long": small_synthetic_long,
    "medium-fineweb": medium_fineweb,
}
```

**Design notes:**

- **Functions, not instances.** The registry maps names to functions, not pre-built configs. This means each call gets a fresh copy — no accidental mutation across experiments.
- **Variants compose from bases.** `small_synthetic_long()` calls `small_synthetic()` and modifies it. When the base changes, all variants inherit the change.
- **Flat namespace.** No hierarchy in registry keys. `"small-synthetic"` not `"models/small/data/synthetic"`. At our scale (<20 experiments), flat is fine.
- **MODEL_PRESETS stays.** The existing preset dict in `dense_transformer.py` maps model size names to `TransformerConfig` instances. Registry functions reference these rather than duplicating architecture params.

## 6. CLI via tyro

### Integration

```python
# nanigpt/config.py (alongside the dataclass definitions)

import sys
import tyro
from nanigpt.configs.registry import REGISTRY


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
    if args is None:
        args = sys.argv[1:]

    # Extract --config if present
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
        if config_name not in REGISTRY:
            available = ", ".join(sorted(REGISTRY.keys()))
            raise ValueError(f"Unknown config '{config_name}'. Available: {available}")
        base_config = REGISTRY[config_name]()
    else:
        base_config = TrainConfig()

    # tyro applies CLI overrides on top of the base config
    return tyro.cli(TrainConfig, args=remaining, default=base_config)
```

### UX

```bash
# Run with a named preset
uv run python -m nanigpt.train --config small-fineweb

# Override learning rate
uv run python -m nanigpt.train --config small-fineweb --training.lr 1e-4

# Override model and training params
uv run python -m nanigpt.train --config small-synthetic \
    --model.d-model 512 --model.n-layers 6 \
    --training.num-steps 1000

# Disable profiler
uv run python -m nanigpt.train --config small-fineweb --profiler.enabled False

# See all available flags and defaults
uv run python -m nanigpt.train --help
```

tyro auto-generates `--help` output from the dataclass tree. Each nested dataclass becomes a section. Field docstrings become help text. `Literal` types become restricted choices. Booleans become `--flag`/`--no-flag` pairs.

### New dependency

```bash
Add `tyro` to `pyproject.toml`
```

tyro is lightweight (pure Python, no C extensions) and has no transitive dependencies beyond standard typing libraries. It's actively maintained by the same people who contribute to nerfstudio.

## 7. Serialization

Replace the hand-built wandb config dict with `dataclasses.asdict()`.

**Current state** (`train.py:100-116`):
```python
wandb.init(
    project="nanigpt",
    config={
        "model_preset": MODEL_PRESET,
        "model": asdict(config),
        "batch_size": BATCH_SIZE,
        "seq_len": SEQ_LEN,
        "lr": LR,
        # ... manually assembled
    },
)
```

**After:**
```python
config = parse_config()

wandb.init(
    project=config.logging.wandb_project,
    config=dataclasses.asdict(config),
)
```

`dataclasses.asdict()` recursively converts the entire config tree to a dict of dicts. wandb flattens nested dicts with dot separators in the UI (`model.d_model`, `training.lr`), so the logged config is both human-readable and filterable.

One edge case: `torch.dtype` is not JSON-serializable. That's why `TrainingConfig.dtype` is a `str` (`"bfloat16"`) with a `torch_dtype` property for runtime use. The serialized config is pure primitives — `dataclasses.asdict()` produces a dict that `json.dumps()` handles without custom encoders.

## 8. What train.py Becomes

The current `main()` function is 200 lines that mixes config definition, dataset construction, model construction, training loop, evaluation, and reporting. After the refactor:

```python
"""Single-GPU training loop with profiling.

Run:
    uv run python -m nanigpt.train --config small-fineweb
    uv run python -m nanigpt.train --help
"""

import dataclasses
import logging
import math
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import wandb
from nanigpt.config import parse_config
from nanigpt.models.dense_transformer import DenseTransformer
from nanigpt.profiling import flop_counter
from nanigpt.profiling.context import init_context, register_step_end, step_context
from nanigpt.profiling.event_types import EventType
from nanigpt.profiling.timer import get_global_metrics, measure


def main():
    config = parse_config()

    logging.basicConfig(
        level=logging.INFO,
        format="\033[2m%(asctime)s %(levelname)s\033[0m %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger("train")

    torch.manual_seed(config.training.seed)
    init_context()

    gpu_name = flop_counter.detect_gpu()

    wandb.init(
        project=config.logging.wandb_project,
        config=dataclasses.asdict(config),
    )

    model = DenseTransformer(config.model).to(config.training.device)

    # Data and profiler constructed via build() — config dispatches to the right type
    train_data = config.data.build(
        vocab_size=config.model.vocab_size,
        num_samples=config.training.num_steps * config.training.batch_size,
        split="train",
    )
    profiler = config.profiler.build()
    # ... rest of training loop uses config.training.*, config.eval.*, etc.
```

**What changes:**
- Module-level constants disappear entirely
- `parse_config()` is the single entry point for all configuration
- `wandb.init` config is a one-liner
- Dataset and profiler constructed via `config.data.build()` and `config.profiler.build()` — polymorphic dispatch, no `isinstance` or `if/else` in train.py
- Every reference to a bare constant (`BATCH_SIZE`, `LR`, etc.) becomes `config.training.batch_size`, `config.training.lr`, etc.

**What doesn't change:**
- The training loop structure (forward, backward, optimizer step)
- The evaluation function
- The end-of-training summary

## 9. Migration Path

Incremental steps from current state to the design above. Each step is a self-contained commit that doesn't break `uv run python -m nanigpt.train`.

**Step 1: Add `kw_only=True, slots=True` to existing dataclasses**

Update `TransformerConfig` and `ProfilerConfig` to use `@dataclass(kw_only=True, slots=True)`. Update `PRESET_CONFIGS` to use keyword arguments (already does). This is a mechanical change — if anything passes positional args, it breaks loudly.

**Step 2: Add the new config dataclasses**

Create `nanigpt/config.py` with `DataConfig`, `TrainingConfig`, `EvalConfig`, `LoggingConfig`, and root `TrainConfig`. Add `__post_init__` validation. This file imports `TransformerConfig` and `ProfilerConfig` from their existing locations.

**Step 3: Add `Configurable` base class and adopt it**

Add `nanigpt/configurable.py` with the `Configurable` base class. Convert `TorchProfiler` and the data sources (`SyntheticData`, `TokenizedData`) to `Configurable` components with nested `Config` dataclasses and `build()` construction.

**Step 4: Add registry and tyro CLI**

`Add `tyro` to `pyproject.toml``. Create `nanigpt/configs/registry.py` with the initial presets. Add `parse_config()` to `nanigpt/config.py`. At this point, both the old style (module-level constants) and new style (`--config`) work.

**Step 5: Refactor train.py**

Replace module-level constants with `config = parse_config()`. Replace all bare constant references with `config.*` access. Replace the hand-built wandb config dict with `dataclasses.asdict(config)`. Delete the old constants block.

**Step 6: Add validation and clean up**

Add cross-field validation in `__post_init__` (e.g., profiler `end_step > start_step`, seq_len > 0). Review the full config surface for anything missed.
