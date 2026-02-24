# Config System Research

**Date:** 2026-02-20
**Context:** The training loop is accumulating module-level constants (`DATA_DIR`, `VAL_INTERVAL`, `VAL_STEPS`, etc.) with conditional branches like `if DATA_DIR:`. Time to design a config system before this gets worse.

## First Principles: What Does a Config System Actually Do?

A training run is fully determined by its configuration: model architecture, data, training hyperparameters, and infrastructure. You need to pin all of these down to launch a run, vary any one independently between runs, and reconstruct the full specification of any past run. That's the whole problem — and it's the scientific method applied to training: precise specification of experimental conditions, controlled variation of one variable at a time, and a complete record for reproduction. The config system is what makes or breaks all three.

Everything else is implementation detail, but those details matter. A config system that makes it painful to vary one knob, or that can't tell you what ran last Tuesday, or that lets you launch a 1000-GPU job with a typo in a field name, is worse than module-level constants.

Decomposing the problem, a config system handles these sub-problems:

1. **Definition** — How do you declare what knobs exist and what their types/defaults are?
2. **Construction** — How do you build a complete config for a specific experiment?
3. **Override** — How do you tweak a config without rewriting it? (CLI, env vars, programmatic)
4. **Validation** — How do you catch bad configs before they waste GPU hours?
5. **Composition** — How do you share config fragments across experiments without copy-paste?
6. **Serialization** — How do you save/restore configs for reproducibility?

## What I Looked At

I looked at training frameworks, standalone config libraries, and various internal frameworks I've seen in the industry. The goal was to see where the community has converged and where genuine disagreements remain.

| Name | Description |
|------|-------------|
| [**Megatron-LM**](https://github.com/NVIDIA/Megatron-LM/tree/32efeffd2) | NVIDIA. Earliest large-scale LLM training framework. Pure argparse. |
| [**torchtitan v1**](https://github.com/pytorch/torchtitan/tree/73a0e697) | PyTorch/Meta. Reference distributed training, TOML-based config. |
| [**torchtitan v2**](https://github.com/pytorch/torchtitan/tree/1ce7f761) | PyTorch/Meta. Same framework, mid-[refactor](https://github.com/pytorch/torchtitan/pull/2386) to programmatic Python config. |
| [**torchtune**](https://github.com/pytorch/torchtune/tree/6f2aa725) | PyTorch/Meta. Finetuning-focused, pure YAML + OmegaConf. |
| [**AXLearn**](https://github.com/apple/axlearn/tree/5602a78) | Apple. JAX-based, production training, pure Python config. |
| [**Fiddle**](https://github.com/google/fiddle) | Google DeepMind. Config = deferred function call. `fdl.Config` + `fdl.build()`. |
| [**ml_collections**](https://github.com/google/ml_collections) | Google. `ConfigDict` — typed dict with locking and reactive references. |

torchtitan is split into v1 and v2 because it's actively undergoing a BC-breaking config refactor that changes the fundamental philosophy — from declarative TOML files to programmatic Python. These represent two genuinely different opinions on how config should work.

---

## Approach 1: CLI Flags (argparse)

Almost every training framework starts the same way:

```python
parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--num-steps", type=int, default=1000)
args = parser.parse_args()

model = build_model()
train(model, args)
```

It's stdlib, zero-dependency, universally understood. For a 10-knob training script, it's the right answer. The question is what happens when the knob count grows.

Megatron-LM is the best case study because it's the oldest and most battle-tested framework here, and it never moved off argparse. It shows exactly where this starting point leads to at scale:

- **Definition:** 500+ hand-written [`parser.add_argument()` calls](https://github.com/NVIDIA/Megatron-LM/blob/32efeffd2/megatron/training/arguments.py). No types beyond what argparse provides. No grouping.
- **Construction:** Shell scripts with bash arrays ([`MODEL_ARGS`, `TRAINING_ARGS`, `DATA_ARGS`](https://github.com/NVIDIA/Megatron-LM/blob/32efeffd2/examples/llama/train_llama3_8b_h100_fp8.sh)) concatenated at launch time.
- **Override:** CLI flags. That's it. Shell scripts *are* the config files.
- **Validation:** [1,150 lines of hand-written assertions](https://github.com/NVIDIA/Megatron-LM/blob/32efeffd2/megatron/training/arguments.py#L293) in `validate_args()`. Manually maintained, manually ordered.
- **Composition:** Copy-paste between shell scripts. No inheritance, no shared fragments.
- **Serialization:** The namespace can be saved/loaded, but it's unstructured.

This actually works! Megatron has trained some of the largest models in existence with this system. But the pain points are clear: the flat namespace makes it impossible to know which args belong to which subsystem, `get_args()` everywhere creates implicit coupling, and the validation function is a 1,150-line block of assertions that someone has to manually maintain.

Megatron itself is now migrating toward dataclasses with [`ArgumentGroupFactory`](https://github.com/NVIDIA/Megatron-LM/blob/32efeffd2/megatron/training/argument_utils.py) auto-generating argparse from field metadata — acknowledging that the original approach doesn't scale.

### Why not just a dict?

Before reaching for YAML files or dataclass hierarchies below, there's another simple option: a plain Python dictionary.

```python
config = {
    "lr": 3e-4,
    "batch_size": 32,
    "model": {
        "hidden_dim": 512,
        "num_layers": 6,
    },
}
```

No imports, no class definitions, trivially serializable (`json.dumps(config)`), nestable, and everyone knows how to use one. For a quick experiment script, this is fine.

The problem is that dicts fail silently. `config["leraning_rate"]` raises `KeyError` at runtime — but `config["leraning_rate"] = 1e-4` succeeds silently, and now you have a config with both `lr` and `leraning_rate` and no error until you notice your learning rate didn't change. There's no schema — nothing says what keys are valid, what types they should be, or what the defaults are. You can't autocomplete `config["` in an IDE. You can't write `__post_init__` to validate cross-field constraints. And as the config grows, every access is a string lookup that could be wrong.

This is the exact gap that dataclasses fill: a dict with a declared schema, typed fields, defaults, and `__post_init__` for validation. Every framework that starts with dicts eventually moves to something with a schema — whether that's dataclasses, `ConfigDict`, `attrs`, or a custom config class. The question is which one.

---

## Approach 2: Config-as-Data (YAML / TOML)

The most natural first step away from argparse: put your config in a structured file. This is what torchtune and torchtitan v1 do, with different file formats but the same philosophy — the config is data, not code.

### torchtune — YAML + OmegaConf

[`recipes/configs/llama3/8B_full.yaml`](https://github.com/pytorch/torchtune/blob/6f2aa725/recipes/configs/llama3/8B_full.yaml):
```yaml
tokenizer:
  _component_: torchtune.models.llama3.llama3_tokenizer
  path: /tmp/Meta-Llama-3-8B-Instruct/original/tokenizer.model

model:
  _component_: torchtune.models.llama3.llama3_8b

optimizer:
  _component_: torch.optim.AdamW
  lr: 2e-5
  fused: True

dataset:
  _component_: torchtune.datasets.alpaca_dataset
  packed: False
```

The `_component_` field is the key idea — it's a Python dotpath that tells torchtune which class/function to instantiate. The [recursive instantiation](https://github.com/pytorch/torchtune/blob/6f2aa725/torchtune/config/_instantiate.py) walks the YAML tree, resolves each `_component_` to a callable, and passes the remaining keys as kwargs:

```python
def _instantiate_node(obj, *args, caller_globals=None):
    if isinstance(obj, dict) or isinstance(obj, DictConfig):
        if "_component_" not in obj:
            return {k: _instantiate_node(v, ...) for k, v in obj.items()}
        else:
            _component_ = _get_component_from_path(obj["_component_"], ...)
            kwargs = {k: _instantiate_node(v, ...) for k, v in obj.items()
                      if k != "_component_"}
            return _create_component(_component_, args, kwargs)
```

CLI overrides use [OmegaConf dotlist merging](https://github.com/pytorch/torchtune/blob/6f2aa725/torchtune/config/_utils.py) — `key=value` pairs merged onto the YAML-loaded config:

```python
def _merge_yaml_and_cli_args(yaml_args, cli_args) -> DictConfig:
    cli_dotlist = []
    for arg in cli_args:
        k, v = arg.split("=")
        if k in yaml_kwargs and _has_component(yaml_kwargs[k]):
            k += "._component_"
        cli_dotlist.append(f"{k}={v}")
    return OmegaConf.merge(OmegaConf.create(yaml_kwargs), OmegaConf.from_dotlist(cli_dotlist))
```

Validation uses [`inspect.signature.bind()`](https://github.com/pytorch/torchtune/blob/6f2aa725/torchtune/config/_validate.py) to check that the YAML kwargs match the callable's actual signature — catching mismatches before training starts.

### torchtitan v1 — TOML + dataclasses

[`train_configs/llama3_8b.toml`](https://github.com/pytorch/torchtitan/blob/73a0e697/torchtitan/models/llama3/train_configs/llama3_8b.toml):
```toml
[model]
name = "llama3"
flavor = "8B"

[optimizer]
name = "AdamW"
lr = 3e-4

[training]
local_batch_size = 1
seq_len = 8192
steps = 1000

[parallelism]
data_parallel_shard_degree = -1
tensor_parallel_degree = 1
```

Each TOML section maps 1:1 to a Python dataclass. [`JobConfig`](https://github.com/pytorch/torchtitan/blob/73a0e697/torchtitan/config/job_config.py) composes them:

```python
@dataclass
class JobConfig:
    job: Job = field(default_factory=Job)
    model: Model = field(default_factory=Model)
    optimizer: Optimizer = field(default_factory=Optimizer)
    training: Training = field(default_factory=Training)
    parallelism: Parallelism = field(default_factory=Parallelism)
    checkpoint: Checkpoint = field(default_factory=Checkpoint)
    # ... 18 sections total
```

CLI overrides use `tyro`, which auto-generates a CLI parser from the dataclass tree — add a dataclass field, get a CLI flag for free.

### The Hydra question

The config-as-data approach has a canonical implementation: [Hydra](https://hydra.cc/) (Meta/Facebook). It's built on OmegaConf and provides the full package — YAML config files, recursive instantiation via `_target_` fields, config groups for composition, multirun sweeps for hyperparameter search, and a plugin system for launchers, sweepers, and logging. It's the most widely adopted config framework in ML research: fairseq, Detectron2, and NeMo all use it.

torchtune's design is explicitly derived from Hydra — the [`instantiate` docstring](https://github.com/pytorch/torchtune/blob/6f2aa725/torchtune/config/_instantiate.py#L83) says "based on Hydra's `instantiate` utility." They use `_component_` instead of Hydra's `_target_`, but the pattern is identical: a YAML field names a Python callable, the remaining fields become kwargs, and a recursive walk builds the object tree. torchtune took the one piece of Hydra it needed and reimplemented it in ~100 lines, skipping config groups, sweepers, and the plugin system.

This is a common pattern — teams adopt pieces of Hydra's design without adopting Hydra itself. The reasons are consistent:

- **Debugging is painful.** Hydra wraps your application's entry point, changes the working directory (by default), and manages its own output folders. When something goes wrong, the stack trace passes through Hydra's internals, and figuring out what config values actually reached your code requires understanding OmegaConf's resolution order.
- **Config groups add indirection.** A single run's config is assembled from multiple YAML files across a directory tree (`config/model/llama3.yaml`, `config/optimizer/adamw.yaml`, `config/data/c4.yaml`). This is powerful for composition but means you can't look at one file and know what a run will do — you have to mentally assemble the pieces.
- **Overrides have their own syntax.** `++key=value` (force add), `~key` (delete), `key@pkg` (package directive) — Hydra's override grammar is its own mini-language that everyone on the team has to learn.
- **It's a framework, not a library.** Hydra wants to own your `main()`. Training frameworks already own `main()`, so adopting Hydra means negotiating who controls application startup, logging, and output directories.

Hydra is maximally flexible, and everyone complains about it, and they use it anyway — because the alternatives (rolling your own YAML loading, override merging, and instantiation) end up reimplementing half of Hydra less carefully. torchtune's approach is an honest acknowledgment of this: take the good idea (`_target_`-based instantiation), reimplement it simply, skip the rest.

### What config-as-data gets right

There's a reason declarative specs have won in so many domains — SQL, Terraform, Kubernetes manifests, Dockerfiles, CI configs. The common thread is separating "what I want" from "how to build it," which gives you properties that imperative code can't guarantee.

- **The config can't have side effects.** A YAML file can't `import os`, can't read environment variables (unless interpolation is explicitly supported), can't branch on runtime conditions. This is a feature — the config is a pure specification of intent. You can reason about what it will produce without executing it.
- **The config IS the record.** With config-as-code, you need a serialization step (`dataclasses.asdict()` → JSON) to get an auditable record of what ran, and you have to trust that the serialized form captures everything. With YAML, the file you checked in is the record. No serialization needed.
- **Content-addressable reproducibility.** You can hash a YAML file. Two identical files guarantee identical configs. Two identical Python functions don't — they might call other functions that changed, read different env vars, or depend on import-time state.
- **Readable and diffable.** You can diff two config files and see exactly what changed between experiments. No need to trace through function calls or inheritance chains.
- **Universally accessible.** Not everyone who needs to tweak configs is fluent in Python. YAML and TOML are readable by anyone, editable without understanding the framework's config abstractions.

### Where it breaks down

The strengths above are real, and for projects with a modest number of experiments they may be sufficient. The pain points emerge as the experiment matrix grows:

- **Composition is painful.** If you have 5 model sizes and 3 datasets, you need 15 YAML files that are 90% identical. When you change the default optimizer, you update every file.
- **Dependencies between fields are clumsy.** `seq_len` might constrain `batch_size` which determines `gradient_accumulation_steps`. In YAML you either duplicate the math in comments, add a post-processing step, or use OmegaConf interpolation (`${...}`) which only handles simple references, not arithmetic.
- **Stringly typed.** A typo in a YAML field name is only caught at runtime (or never). No IDE autocomplete, no type checking. The `_component_` pattern makes this especially fragile — you're writing Python dotpaths as strings.
- **Validation requires separate machinery.** torchtune needs `inspect.signature.bind()` to verify that YAML kwargs match the callable. Dataclass-backed TOML (torchtitan v1) is better — `__post_init__` catches some errors — but cross-section constraints still need manual code.

The torchtitan v1→v2 migration is the most interesting data point here: a framework that *started* with config-as-data and is actively moving away from it. But that migration is driven by torchtitan's specific needs — a reference framework that must generate many config variants programmatically across model sizes, parallelism strategies, and hardware targets. A project that runs a handful of experiments from hand-written configs wouldn't feel the same pressure.

---

## Approach 3: Config-as-Code (Python dataclasses + registry)

The programmatic approach: the config is Python code. No YAML, no TOML — just dataclasses and functions that return them. This is where torchtitan v2 and AXLearn have converged, and it's the direction internal frameworks at big labs I've seen tend to go.

### torchtitan v2 — dataclasses + registry functions

[`config_registry.py`](https://github.com/pytorch/torchtitan/blob/1ce7f761/torchtitan/models/llama3/config_registry.py) defines named experiments as functions returning typed `Trainer.Config` instances:

```python
def llama3_debugmodel() -> Trainer.Config:
    return Trainer.Config(
        model_spec=model_registry("debugmodel"),
        optimizer=OptimizersContainer.Config(lr=8e-4),
        training=TrainingConfig(
            local_batch_size=8, seq_len=2048, steps=10,
        ),
        dataloader=HuggingFaceTextDataLoader.Config(dataset="c4_test"),
        checkpoint=CheckpointManager.Config(interval=10),
    )

# Variants compose from the base:
def llama3_debugmodel_float8() -> Trainer.Config:
    config = llama3_debugmodel()
    config.model_converters = ModelConvertersContainer.Config(
        converters=[Float8LinearConverter.Config(...)],
    )
    return config
```

Config dataclasses use [`Literal` types](https://github.com/pytorch/torchtitan/blob/1ce7f761/torchtitan/config/configs.py) for constrained fields and `__post_init__` for validation:

```python
@dataclass(kw_only=True, slots=True)
class ParallelismConfig:
    fsdp_reshard_after_forward: Literal["default", "always", "never"] = "default"
    context_parallel_rotate_method: Literal["allgather", "alltoall"] = "allgather"
    context_parallel_load_balancer: str | None = None

    def __post_init__(self):
        if self.context_parallel_load_balancer == "":
            raise ValueError(
                "context_parallel_load_balancer cannot be an empty string. "
                "Use None to disable load balancing."
            )
```

CLI overrides use [`tyro`](https://github.com/brentyi/tyro) as a thin layer on top of the registry. [`ConfigManager`](https://github.com/pytorch/torchtitan/blob/1ce7f761/torchtitan/config/manager.py) loads a config from the registry, then passes it as `default=` to `tyro.cli()`:

```python
class ConfigManager:
    """Configuration precedence: CLI args > config_registry function defaults"""
    def parse_args(self, args: list[str] = sys.argv[1:]):
        loaded_config, args = self._load_config(args)
        self.config = tyro.cli(
            type(loaded_config), args=args, default=loaded_config, registry=custom_registry
        )
        self._validate_config()
        return self.config
```

tyro walks the nested dataclass tree and auto-generates a CLI with dot-separated flags for every field. You never write `add_argument()` — adding a field to a dataclass automatically creates a CLI flag:

```bash
python train.py --module llama3 --config llama3_debugmodel \
    --optimizer.lr 1e-4 \
    --training.seq-len 4096 \
    --profiling.enable-profiling \
    --checkpoint.interval 500
```

Each nested dataclass becomes a section in `--help`, with types, defaults, and docstrings pulled directly from the dataclass definitions. `Literal` types become restricted choices, booleans become `--flag`/`--no-flag` pairs, and `Optional` fields accept `None`. The entire CLI is derived from the type annotations — no manual parser code to maintain.

The backbone of v2 is the [`Configurable`](https://github.com/pytorch/torchtitan/blob/1ce7f761/torchtitan/config/configurable.py) base class. It's not optional infrastructure — every component in the framework inherits from it: [`Trainer`](https://github.com/pytorch/torchtitan/blob/1ce7f761/torchtitan/trainer.py#L55), `CheckpointManager`, `OptimizersContainer`, `BaseDataLoader`, `BaseValidator`, `MetricsProcessor`, `Float8LinearConverter`, and more. Each component owns a nested `Config` dataclass, and `config.build()` constructs the owning component:

```python
class Configurable:
    @dataclass(kw_only=True, slots=True)
    class Config:
        _owner: ClassVar[type | None] = None
        def build(self, **kwargs):
            return self._owner(config=self, **kwargs)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if "Config" in cls.__dict__:
            config_cls = cls.__dict__["Config"]
            if "__slots__" not in config_cls.__dict__:
                raise TypeError(f"{cls.__name__}.Config must use @dataclass(kw_only=True, slots=True)")
            config_cls._owner = cls
```

Here's what it looks like for a concrete component. [`CheckpointManager`](https://github.com/pytorch/torchtitan/blob/1ce7f761/torchtitan/components/checkpoint.py#L125) inherits `Configurable` and defines its `Config` as a nested dataclass. The docstrings on each field become the `--help` text in tyro's auto-generated CLI:

```python
class CheckpointManager(Configurable):
    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        enable: bool = False
        """Whether to enable checkpoint"""

        interval: int = 500
        """Checkpointing interval in steps"""

        last_save_model_only: bool = False
        """Whether to only save the model in the last checkpoint"""

        # ... more fields
```

The config tree in a registry function is a tree of these buildable components. Construction cascades through `.build()` calls. At the top level, [`train.py`](https://github.com/pytorch/torchtitan/blob/1ce7f761/torchtitan/train.py#L41) is just:

```python
config_manager = ConfigManager()
config = config_manager.parse_args()
trainer = config.build()
trainer.train()
```

`config.build()` constructs a `Trainer` (because `config._owner` is `Trainer`). Inside [`Trainer.__init__`](https://github.com/pytorch/torchtitan/blob/1ce7f761/torchtitan/trainer.py#L230), each nested config builds its own component, passing whatever runtime arguments it needs:

```python
# Inside Trainer.__init__(self, config):
self.tokenizer = config.tokenizer.build(tokenizer_path=config.hf_assets_path)
self.dataloader = config.dataloader.build(
    dp_world_size=batch_degree, dp_rank=batch_rank,
    tokenizer=self.tokenizer, seq_len=config.training.seq_len,
)
model = model_config.build()
self.optimizers = config.optimizer.build(model_parts=self.model_parts)
self.lr_schedulers = config.lr_scheduler.build(optimizers=self.optimizers)
```

So the config tree mirrors the object tree. Each config knows which class it constructs, and construction flows top-down through `.build()` — the `Trainer` builds its components, each component builds its sub-components. The config is a complete blueprint; `.build()` turns it into a running system.

Because every component owns its config, definitions are [distributed across the codebase](https://github.com/pytorch/torchtitan/blob/1ce7f761/torchtitan/config/configs.py), not centralized. `ProfilingConfig` lives in `tools/profiling.py`, `CheckpointManager.Config` lives near `CheckpointManager`. Only configs without a clear owner (like `ParallelismConfig`, `TrainingConfig`) live in the shared `configs.py`.

### AXLearn — deep hierarchy + registry

torchtitan v2's `Configurable` pattern didn't appear in a vacuum — AXLearn has been doing this for years. The structural parallels are striking: both have a `Configurable` base class with a nested `Config`, both use `config.instantiate()` / `config.build()` to construct the owning class, and both distribute config definitions across the codebase so each component owns its config.

AXLearn's [`Configurable`](https://github.com/apple/axlearn/blob/5602a78/axlearn/common/config.py#L796) is the base. Every component inherits from it and defines a nested `Config`:

```python
class Configurable:
    @config_class
    class Config(InstantiableConfig):
        klass: type  # set automatically by default_config()

        def instantiate(self, **kwargs):
            _validate_required_fields(self)
            return self.klass(self, **kwargs)

    @classmethod
    def default_config(cls):
        return cls.Config(klass=cls)
```

The API differs from torchtitan in a few ways. Configs are constructed via `default_config()` and mutated with a chainable `.set()` method rather than constructor kwargs:

```python
# torchtitan v2 style:
CheckpointManager.Config(interval=10, enable=True)

# AXLearn style:
Checkpointer.default_config().set(keep_last_n=3, keep_every_n_steps=50_000)
```

`default_config()` creates a `Config` with `klass` automatically set to the owning class. `.set()` is just `setattr` in a loop that returns `self` for chaining.

[`Module`](https://github.com/apple/axlearn/blob/5602a78/axlearn/common/module.py#L745) extends `Configurable` and adds a parent-child tree. Children are added via `_add_child(name, child_config)`, which calls `child_config.instantiate(parent=self)` — so the config tree turns into a module tree:

```python
class Module(Configurable):
    @config_class
    class Config(Configurable.Config):
        name: Required[str] = REQUIRED

    def _add_child(self, name, child_config, **kwargs):
        child_config.name = name
        module = child_config.instantiate(parent=self, **kwargs)
        self._children[name] = module
        return module
```

The [trainer](https://github.com/apple/axlearn/blob/5602a78/axlearn/common/trainer.py#L224) uses this to construct itself from its config — same cascade pattern as torchtitan, but with explicit parent-child relationships:

```python
class SpmdTrainer(Module):
    @config_class
    class Config(Module.Config):
        model: Required[BaseModel.Config] = REQUIRED
        learner: Required[Learner.Config] = REQUIRED
        checkpointer: BaseCheckpointer.Config = Checkpointer.default_config()
        evalers: dict[str, SpmdEvaler.Config] = {}
        mesh_shape: Required[MeshShape] = REQUIRED

    def __init__(self, cfg, *, parent):
        super().__init__(cfg, parent=parent)
        # Each _add_child call instantiates the child config:
        self._add_child("model", cfg.model)
        self._add_child("learner", cfg.learner)
        self._add_child("checkpointer", cfg.checkpointer)
        for name, evaler_cfg in cfg.evalers.items():
            self._add_child(name, evaler_cfg, model=self.model)
```

`trainer.model.decoder.attention.num_heads` is a real access path — the config nesting goes 4+ levels deep, mirroring the module tree exactly.

Experiments are organized as [`named_trainer_configs()`](https://github.com/apple/axlearn/blob/5602a78/axlearn/experiments/text/gpt/c4_trainer.py#L112-L119) registries. The [Fuji experiment file](https://github.com/apple/axlearn/blob/5602a78/axlearn/experiments/text/gpt/fuji.py#L1106) uses `itertools.product` to generate configs across versions, model sizes, and attention types. The [`get_trainer_config_fn`](https://github.com/apple/axlearn/blob/5602a78/axlearn/experiments/text/gpt/common.py#L681) factory builds a full `SpmdTrainer.Config` by composing sub-configs:

```python
def get_trainer_config_fn(*, model_cfg, learner_cfg, max_step, ...) -> TrainerConfigFn:
    def config_fn() -> InstantiableConfig:
        cfg = SpmdTrainer.default_config()
        cfg.model = model_cfg
        cfg.learner = learner_cfg
        cfg.max_step = max_step
        cfg.checkpointer.keep_last_n = 3
        cfg.checkpointer.keep_every_n_steps = keep_every_n_steps
        cfg.mesh_shape = mesh_shape
        return cfg
    return config_fn

# In fuji.py — generate configs for every (version × model_size × flash_attention) combination:
for version, model_size, flash_attention in itertools.product(Version, MODEL_SIZES, [True, False]):
    config_map[config_name] = get_trainer_config_fn(
        model_cfg=get_model_config(model_size, version),
        learner_cfg=adamw_decoupled_learner_config(...),
        ...
    )
```

AXLearn's config system also has [fuzzy-match typo detection](https://github.com/apple/axlearn/blob/5602a78/axlearn/common/config.py) — setting an unknown field raises an error with "did you mean?" suggestions using trigram overlap matching:

```python
def similar_names(name: str, candidates: Iterable[str]) -> list[str]:
    def overlaps(name, key):
        matches = sum(1 for i in range(len(name) - 2) if name[i:i+3] in key)
        return float(matches) / max(len(name) - 2, 1)
    return [key for score, key in sorted(...) if score > 0.5]

# In __setattr__:
if key not in _attr_fields_dict_cache(type(self)):
    raise UnknownFieldError(f"{name} (did you mean: {similar_names(name, self.keys())})")
```

### What config-as-code gets right

- **Forking configs is trivial.** `config = base(); config.lr = 1e-4; return config` — one line gets you a variant. 5 model sizes × 3 data configs × 2 precision modes = 30 one-liner functions, not 30 YAML files.
- **Dependencies between fields are natural.** `cfg.grad_accum = target_tokens // (cfg.batch_size * cfg.seq_len)` — just Python.
- **Type safety at definition time.** IDE autocomplete, import-time validation, type checking for free. `Literal` types catch invalid enum values at parse time.
- **Validation is co-located with definition.** `__post_init__` on each dataclass section means adding a field and its constraints happens in the same place, not in a distant 1,150-line function.
- **Composition compounds.** Adding a new experiment dimension (say, precision mode) means adding one modifier function, not duplicating every existing config.

### Where it has costs

- **Auditability requires serialization.** A YAML file is a static record of what ran. A Python function might call other functions, read environment variables, or branch on conditions. You need a serialization step (`dataclasses.asdict()` → JSON) to get the same guarantee.
- **Deep nesting has real overhead.** AXLearn's 4+ levels of nesting means constructing a config requires understanding the module hierarchy, and debugging "which level set this value?" gets harder. Two levels (torchtitan) is the sweet spot for most projects.
- **Everyone has to learn the patterns.** A YAML file is self-explanatory. Registry functions, `Configurable.build()`, `__init_subclass__` enforcement — these are powerful but add concepts that every contributor has to understand.

---

## Approach 4: Config Libraries

The approaches above are all *patterns within training frameworks* — they build config handling from stdlib or common tools. A different option is to use a purpose-built configuration library. Two from the Google ecosystem are worth understanding.

### ml_collections — ConfigDict

`ConfigDict`, from google's `ml_collections`` repo, is a dict with guardrails. It adds dot access, runtime type safety, key locking, and "did you mean" typo detection on top of a plain dict:

```python
cfg = config_dict.ConfigDict()
cfg.lr = 3e-4
cfg.batch_size = 32
cfg.model = config_dict.ConfigDict()
cfg.model.hidden_dim = 512
cfg.model.num_layers = 6
```

Once a field has a type, it's locked — assigning a `str` to an `int` field raises `TypeError`. Calling `cfg.lock()` prevents adding *new* keys, catching typos:

```python
cfg.lock()
cfg.lr = 1e-4           # OK — existing field, same type
cfg.learing_rate = 1e-4  # KeyError: Did you mean "lr" instead of "learing_rate"?
```

The most distinctive feature is `FieldReference` — reactive values that propagate changes:

```python
lr = config_dict.FieldReference(1e-3)
cfg.encoder_lr = lr
cfg.decoder_lr = lr
lr.set(3e-4)  # both encoder_lr and decoder_lr update
```

Configs are defined as `get_config()` functions in Python files, with CLI overrides via `absl.flags`:

```bash
python train.py --config=configs/experiment.py --config.lr=1e-4 --config.model.hidden_dim=1024
```

`FrozenConfigDict` provides an immutable, hashable snapshot — useful for reproducibility records and cache keys.

**Trade-offs:** ConfigDict is maximally flexible (any key, any value, add fields at runtime) but its schema is implicit — types are inferred from the first value assigned, and there's no static type checking. IDE autocomplete doesn't work because field names aren't known at definition time. It's the dominant config approach in the JAX/Google research ecosystem, but it occupies a middle ground that gets the worst of both worlds: more ceremony than a plain dict, less safety than a dataclass.

### Fiddle — deferred construction

Fiddle's core idea is more radical: **configuration is a deferred function call**. Instead of defining a separate config class, you wrap the thing you're actually constructing:

```python
cfg = fdl.Config(Transformer)
cfg.num_layers = 6
cfg.hidden_dim = 512
cfg.attention = fdl.Config(MultiHeadAttention)
cfg.attention.num_heads = 8

model = fdl.build(cfg)  # calls Transformer(num_layers=6, hidden_dim=512, attention=MultiHeadAttention(num_heads=8))
```

`fdl.build()` traverses the config DAG bottom-up — inner configs are built first, then passed as arguments to outer configs. If the same `Config` object appears in multiple places, it's built once and shared (memoization), which naturally expresses weight sharing.

The schema comes from the function/class signatures — setting an unknown parameter raises `AttributeError` immediately, listing valid parameter names. No separate config class needed.

Composition uses `fiddlers` — functions that transform configs:

```python
def base_config():
    cfg = fdl.Config(Trainer)
    cfg.model = fdl.Config(Transformer, hidden_dim=512)
    return cfg

def make_large(cfg):
    cfg.model.hidden_dim = 2048
    cfg.model.num_layers = 24
```

Tags enable cross-cutting concerns — mark parameters with a semantic label, then sweep all of them at once:

```python
class LearningRate(fdl.Tag):
    "The optimizer learning rate."

cfg.lr = LearningRate.new(default=1e-3)
fdl.set_tagged(cfg, tag=LearningRate, value=3e-4)  # sets ALL LearningRate-tagged params
```

CLI integration uses `absl.flags` with composable directives:
```bash
python train.py --config=config:base_config --config=fiddler:make_large --config=set:model.hidden_dim=1024
```

**Trade-offs:** Fiddle is zero-boilerplate — no config classes to maintain, the schema is the code being configured. The DAG structure with memoization is genuinely powerful for complex model architectures. But it has a steeper learning curve (the "deferred call" mental model is unusual), no runtime type checking (relies on static analysis), and it's tightly coupled to the Google/absl ecosystem. The `auto_config` decorator — which transforms normal Python construction code into config-producing code via AST transformation — is clever but personally it feels like a step too far in magic.

### Where config libraries fit

Neither `ConfigDict` nor `Fiddle` has been widely adopted in the PyTorch training ecosystem. The frameworks we looked at (torchtitan, torchtune, AXLearn, Megatron) all build their own config patterns from stdlib tools (`dataclasses`, `argparse`) or lightweight third-party libraries (OmegaConf, tyro).

The pattern that's converging: **dataclasses as the foundation, registry functions for experiment definition, tyro or similar for CLI auto-generation**. This gives you static types, IDE support, and composition without a heavy dependency. Fiddle's `build()` pattern is elegant and has influenced torchtitan v2's `Configurable.build()`, but the full Fiddle library brings more machinery than most PyTorch projects want.

ConfigDict's `FieldReference` (reactive values) and Fiddle's `Tags` (cross-cutting sweeps) are interesting features that none of the dataclass-based approaches have. If you need them, the libraries earn their keep. For a research project at our scale, they're not necessary.

---

## What Makes Sense for naniGPT

We're adopting the torchtitan v2 pattern wholesale. The project is heading toward multiple parallelism strategies and model architectures, so the `Configurable` + `build()` pattern earns its keep now rather than later — retrofitting it after the fact is always harder than starting with it.

**What we're building:**

1. **`Configurable` base class with `build()`** — Every component (trainer, model, optimizer, data loader, profiler) inherits from `Configurable` and owns a nested `Config` dataclass. `config.build()` constructs the owning component. `__init_subclass__` enforces `@dataclass(kw_only=True)` on all configs.
2. **Nested dataclasses** — `TrainerConfig` contains `ModelConfig`, `DataConfig`, `EvalConfig`, `ProfilingConfig`. Each is a typed dataclass with defaults. Two levels of nesting max to start.
3. **Registry functions** — Named experiments as functions returning `TrainerConfig` instances. Base config + variants that modify it, same pattern as `llama3_debugmodel()` → `llama3_debugmodel_float8()`.
4. **`__post_init__` validation** — Constraints (seq_len vs data size, batch size divisibility) checked at config construction with clear error messages. Co-located with the fields they validate.
5. **Configs near their owners** — `ProfilingConfig` lives in the profiling module, `ModelConfig` lives near the model. Only shared configs (like `TrainingConfig`) live in a common location.
6. **`tyro` for CLI** — Auto-generated CLI from the dataclass tree. `--config` selects a named preset from the registry, remaining flags override individual fields with dot-separated paths (`--optimizer.lr 1e-4`). No manual `add_argument()` calls to maintain.
7. **Serialization** — `dataclasses.asdict()` → JSON, logged to wandb and saved alongside checkpoints.

**What we're not doing:**

- **YAML/TOML config files** — Even torchtitan moved away from this. Our configs are Python functions.
- **Hydra / Fiddle / ConfigDict** — Heavy dependencies for a project with ~10 config knobs. We can always adopt one later if the need arises.
