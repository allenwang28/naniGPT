# naniGPT

## What This Is

An exploration of what it looks like to build a frontier training framework. Not an actual frontier training framework — nobody's production workload depends on this and nobody should pretend otherwise.

The goal is to understand the design decisions, tradeoffs, and systems that go into real training infrastructure by building one from scratch. Every module is an opportunity to study how existing codebases solve a problem, try different approaches, and develop informed opinions.

This project grows with the builder. Modules get added, rewritten, and deepened as understanding develops. Nothing here is designed to be finished.

## Guiding Question

For a given compute budget, what's the best way to spend it — and how does that answer change as you vary model architecture, parallelism strategy, data properties, and hardware constraints?

Start small enough to iterate fast. Scale up when the small-scale answers stop being interesting or when you specifically want to study what breaks at larger scale.

## Core Questions

These are the stable research questions driving the project. They don't change when code gets refactored.

1. **Where does time go in a training step?** Decompose into compute, communication, and memory. Measure each. Compare theoretical vs achieved FLOPs. Build roofline models. Understand the gaps.

2. **How do parallelism strategies actually work and compose?** Not "what is FSDP" but "what are the exact communication primitives, when do they fire, what do they cost?" Covers DDP, FSDP/ZeRO, tensor parallelism, expert parallelism, and their compositions.

3. **When and why do MoEs win?** Dense vs MoE at matched FLOPs and matched parameters. Map the crossover points by varying expert count, capacity factor, GPU count. Make expert specialization visible using synthetic data with known ground truth.

4. **What do scaling laws feel like in practice?** Train models at multiple sizes and token budgets. Fit your own curves. Compare to Chinchilla. Compare dense vs MoE scaling behavior.

5. **How does data structure interact with model architecture?** Use controlled synthetic experiments to isolate architectural effects from data effects. Build diagnostic baselines, then explain deviations on real data.

6. **What does high-performance GPU kernel development look like?** Write Triton kernels for real bottlenecks found through profiling. Measure against roofline and PyTorch defaults.

## Design Principles

- **Measurement is built into everything.** Profiling is not an afterthought. Every module should be instrumented from the start. If you can't measure it, you can't understand it.

- **Synthetic first, then real data.** Synthetic data is not a cheap substitute — it's a better tool for specific questions because you control the ground truth. Establish baselines with controlled experiments, then add complexity and explain the deviation. Same methodology as physics.

- **Each module owns one concern.** Modules compose through simple, explicit interfaces. No dependency injection, no plugin registries, no abstract factories.

- **Clarity over optimization.** Code should be readable and understandable. Optimize only after profiling proves a bottleneck, and keep the readable version around for reference.

- **Over-engineering is encouraged.** This is an exploration of design space, not a production system. Build abstractions to understand their tradeoffs. Try different patterns deliberately.

- **Hardware-independent reasoning.** Profile everything as utilization ratios (% of peak FLOPS, % of peak bandwidth) so insights transfer across GPU generations.

## The Rust Track

A cross-cutting concern across the entire project: quantify where Rust can realistically improve a training codebase.

**Methodology:** Implement in Python first. Profile. Identify overhead. Rewrite in Rust (via PyO3). Measure the delta.

**High-value targets:**
- Data pipeline (tokenization, loading, preprocessing, shuffling, batch assembly)
- Checkpointing (serialization, async I/O, filesystem operations)
- Experiment orchestration (config parsing, run management, metric collection)
- Profiling infrastructure (collection, aggregation, reporting)

**Interesting middle ground:**
- Training loop driver in Rust that calls into Python only for forward/backward/step
- Process group management and communication scheduling
- Custom kernel launch wrappers

**Expected outcome:** Rust makes a significant difference in data loading and checkpointing, modest difference in orchestration, near-zero difference in the GPU-bound training hot path. Proving this with measurements is the point.

## Non-Goals

- Production readiness
- Supporting arbitrary models or workloads
- Matching performance of optimized frameworks
- General-purpose distributed training library
- Pipeline parallelism (least transferable parallelism strategy)

## Project Conventions

- **Use `uv` for everything.** Package management, running scripts (`uv run python -m nanigpt.train`), adding dependencies (`uv add`), etc. No pip, no conda, no venv manually.
- **Lint and format with `ruff`.** Run `uv run ruff check .` and `uv run ruff format .` before committing. Config lives in `pyproject.toml`.
- **Use f-strings for formatting.** Prefer `f"step {step}"` over `"step %d" % step` or `"step {}".format(step)`. This applies everywhere including `logging` calls — use `log.info(f"step {step}")` not `log.info("step %d", step)`.
