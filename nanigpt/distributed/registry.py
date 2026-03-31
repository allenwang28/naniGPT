"""Type-based sharding registry: dispatch sharding decisions by module type.

Instead of matching modules by name suffix (e.g. name.endswith("attn.q_proj")),
the registry dispatches on type(module). This is unambiguous when a model has
heterogeneous blocks, injected modules (LoRA), or multiple submodels (vision
encoder + language model).

The registry separates resolution from execution:

    resolve_sharding(model, plan)  → what should be sharded and how
    apply_sharding(...)          → actually slice weights and replace forwards

The resolve step is pure — no side effects, no process groups needed. This
makes it usable for both the real sharding path and a future meta-device
dry run.

Usage:

    from nanigpt.distributed.registry import register_sharding, resolve_sharding

    @register_sharding(MyModule)
    def shard_my_module(module, plan):
        return ResolvedSharding(
            children={"proj": colwise(), "out": rowwise()},
            adjustments={"n_heads": module.n_heads // plan.tp_size},
        )

    # Later:
    resolved = resolve_sharding(model, plan)
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch.nn as nn

from nanigpt.distributed.plan import ParallelPlan
from nanigpt.distributed.sharding import ModuleSharding

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ResolvedSharding:
    """Result of a sharding function: what to do to a module's children.

    children: maps child module name (e.g. "q_proj") to its ModuleSharding.
    adjustments: arbitrary attribute overrides to apply to the parent module
        after sharding (e.g. {"n_heads": 2} to halve the head count).
    """

    children: dict[str, ModuleSharding]
    adjustments: dict[str, Any] = field(default_factory=dict)


ShardingFn = Callable[[nn.Module, ParallelPlan], ResolvedSharding | None]

_REGISTRY: dict[type[nn.Module], ShardingFn] = {}


def register_sharding(
    module_type: type[nn.Module],
) -> Callable[[ShardingFn], ShardingFn]:
    """Decorator to register a sharding function for a module type."""

    def decorator(fn: ShardingFn) -> ShardingFn:
        if module_type in _REGISTRY:
            raise ValueError(f"Sharding function already registered for {module_type.__name__}")
        _REGISTRY[module_type] = fn
        return fn

    return decorator


def get_sharding_fn(module_type: type[nn.Module]) -> ShardingFn | None:
    """Look up the registered sharding function for a module type."""
    return _REGISTRY.get(module_type)


def _load_model_shardings() -> None:
    """Import per-model sharding files to populate the registry.

    Each model has a `*_parallel.py` file alongside its model definition
    (torchtitan-style). Importing it triggers the @register_sharding
    decorators. New models add their own file — the registry stays generic.
    """
    import nanigpt.models.dense_transformer.parallelize  # noqa: F401


_load_model_shardings()


@dataclass(frozen=True, slots=True)
class ShardingEntry:
    """One entry in the resolved sharding plan.

    parent_fqn: fully-qualified name of the parent module (e.g. "blocks.0.attn")
    child_name: name of the child to shard (e.g. "q_proj")
    sharding: the ModuleSharding describing how to shard it
    """

    parent_fqn: str
    child_name: str
    sharding: ModuleSharding


@dataclass(frozen=True, slots=True)
class ShardingPlan:
    """Complete resolved sharding plan for a model.

    entries: the list of (parent, child, sharding) tuples
    adjustments: maps parent_fqn to attribute overrides (e.g. n_heads)
    """

    entries: list[ShardingEntry]
    adjustments: dict[str, dict[str, Any]]

    def format_table(self, model: nn.Module, plan: "ParallelPlan") -> str:
        """Pretty-print the sharding plan as a table.

        Uses the model's current parameter shapes (works on both real and
        meta-device models) and the plan's mesh degrees to compute local
        shapes. This runs at the resolution layer — no process groups needed.
        """
        if not self.entries:
            return "  (no modules sharded)"

        from nanigpt.distributed.sharding import Shard, compute_local_shape

        module_dict = dict(model.named_modules())
        mesh_degrees = {
            dim: plan.degree_for(dim)
            for dim in {
                mesh_dim
                for entry in self.entries
                for mesh_dim in entry.sharding.params.get("weight", {})
            }
        }

        rows: list[tuple[str, str, str, str]] = []
        for entry in self.entries:
            fqn = f"{entry.parent_fqn}.{entry.child_name}" if entry.parent_fqn else entry.child_name
            child = module_dict.get(fqn)
            if child is None or not isinstance(child, nn.Linear):
                continue

            weight_placements = entry.sharding.params.get("weight", {})
            placement_strs = []
            for dim_name, placement in weight_placements.items():
                if isinstance(placement, Shard):
                    placement_strs.append(f"{dim_name}: Shard({placement.dim})")
                else:
                    placement_strs.append(f"{dim_name}: Replicate")
            placement_str = ", ".join(placement_strs) if placement_strs else "Replicate"

            global_shape = (child.out_features, child.in_features)
            local_shape = compute_local_shape(global_shape, weight_placements, mesh_degrees)
            rows.append((fqn, placement_str, str(list(global_shape)), str(list(local_shape))))

        fqn_w = max(len(r[0]) for r in rows)
        place_w = max(len(r[1]) for r in rows)
        global_w = max(len(r[2]) for r in rows)

        header = f"  {'module':<{fqn_w}}  {'placement':<{place_w}}  {'global':<{global_w}}  local"
        sep = f"  {'-' * fqn_w}  {'-' * place_w}  {'-' * global_w}  -----"
        body = "\n".join(
            f"  {fqn:<{fqn_w}}  {place:<{place_w}}  {gl:<{global_w}}  {lo}"
            for fqn, place, gl, lo in rows
        )

        # Adjustments section
        adj_lines = ""
        if self.adjustments:
            adj_parts = []
            for parent_fqn, adjs in self.adjustments.items():
                for attr, value in adjs.items():
                    parent = module_dict.get(parent_fqn)
                    original = getattr(parent, attr, "?") if parent else "?"
                    adj_parts.append(f"  {parent_fqn}.{attr}: {original} → {value}")
            adj_lines = "\n  adjustments:\n" + "\n".join(adj_parts)

        return f"{header}\n{sep}\n{body}{adj_lines}"


def resolve_sharding(model: nn.Module, plan: ParallelPlan) -> ShardingPlan:
    """Walk a model and resolve sharding decisions for all registered module types.

    This is the pure resolution step — no weights are modified, no process
    groups are needed. The result can be consumed by apply_sharding() for
    real application, or by a dry run for shape analysis.

    TODO: resolve_sharding and the plan types (ShardingPlan, ShardingEntry)
    may belong in their own module once we have a second consumer (dry run,
    cost model) that makes the right boundary clearer.

    Returns a ShardingPlan with entries for every child that should be sharded,
    plus any attribute adjustments for parent modules.
    """
    entries: list[ShardingEntry] = []
    adjustments: dict[str, dict[str, Any]] = {}

    for fqn, module in model.named_modules():
        sharding_fn = get_sharding_fn(type(module))
        if sharding_fn is None:
            continue

        resolved = sharding_fn(module, plan)
        if resolved is None:
            continue

        for child_name, child_sharding in resolved.children.items():
            entries.append(ShardingEntry(fqn, child_name, child_sharding))

        if resolved.adjustments:
            adjustments[fqn] = resolved.adjustments

    return ShardingPlan(entries=entries, adjustments=adjustments)
