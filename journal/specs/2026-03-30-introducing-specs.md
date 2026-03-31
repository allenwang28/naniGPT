# Introducing Specs

*2026-03-30*

## The Gap

The codebase has three layers of documentation, each with a different purpose:

- **Journal entries** — exploratory, timestamped. "I tried X, learned Y, decided Z." Good for design rationale and research findings. Bad as reference docs because they accumulate without consolidation.
- **Module docstrings** — local to one file. Good for data flow diagrams, API details, implementation notes. Bad for cross-file narratives ("how do comm.py, tensor_parallel.py, and spmd.py compose?").
- **CLAUDE.md** — project-wide conventions and philosophy. Good for "how do we work here." Bad for "how does TP work."

The missing layer is: "how does this subsystem work?" — the thing you read *before* opening any source file, so you know what you're looking at when you get there.

## Why Now

As the project moves toward AI-native co-development, the readers maintaining and extending the code are increasingly context-limited. A human skimming before a code review and an agent with a finite context window have the same need: build a working mental model of a subsystem *fast*, without reading every file.

Code is the ground truth, but it's expensive to parse — especially distributed systems code where the "why" is rarely obvious from the "what." An agent asked to "add expert parallelism" needs to understand TP's invariants (enter/exit must be paired, nonlinear(P) is forbidden, bias after reduce) before touching anything. Those invariants are currently scattered across docstrings, inline comments, and the reader's head. A spec collects them in one place.

This isn't about making documentation for documentation's sake. It's about making the codebase's implicit knowledge explicit and scannable — a shared interface between the code and any reader (human or agent) who needs to work in it safely.

## The Format

Each spec lives in `docs/specs/` and follows a fixed structure:

- **Overview** — what this subsystem does, in 2-3 paragraphs. Written for someone who knows the domain (distributed training) but not this codebase.
- **Key Files** — each file with a one-liner on its role. Not what's *in* it (that's the docstring's job), but where it sits in the narrative. This is the "which file do I open first?" guide.
- **Invariants** — rules that must hold for correctness. Things that produce silent bugs if violated. This is the most valuable section — invariants are the hardest thing to extract from code alone and the easiest thing to violate without one.
- **Design Decisions** — non-obvious choices and why. Links to journal entries for the full exploration, but the conclusion lives here. This prevents re-litigating settled decisions.

What *doesn't* go in a spec: data flow diagrams, API usage examples, implementation details. Those belong in module docstrings next to the code they describe. Specs point you to the right file; docstrings explain what's inside.

## Specs vs Journal Entries

Journal entries are append-only explorations. Specs are updated in place. A journal entry might explore three approaches to TP dispatch and conclude with one; the spec just states the conclusion and links back to the journal entry for the full reasoning.

Specs can go stale — but they're easier to keep current than journal entries because they're structured. An invariant that no longer holds is obviously wrong when you read it. A journal entry's conclusion buried in paragraph 7 is easy to miss.

## Granularity

One spec per subsystem that spans multiple files. "Tensor parallelism" is a spec (comm.py + tensor_parallel.py + spmd.py). "Data parallelism" will be another (data_parallel.py + mesh.py). The model itself might be another. The profiling system another.

The test: if removing one file would make the spec incomplete, those files belong in the same spec. If two groups of files can be understood independently, they're separate specs.

## Directory Structure

Specs mirror the code structure. `docs/specs/distributed/tensor-parallelism.md` corresponds to the files under `nanigpt/distributed/`. This makes the spec for any module predictable — you know where to look without searching.

## First Spec

`docs/specs/distributed/tensor-parallelism.md` — covers the TP subsystem. Seven invariants, four design decisions, five key files. Written as the template for future specs.
