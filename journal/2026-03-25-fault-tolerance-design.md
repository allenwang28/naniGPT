# Designing naniGPT's fault tolerance system

The [previous entry](2026-02-21-fault-tolerance.md) surveyed how Meta, ByteDance, and
Google handle fault tolerance at 10K-100K+ GPUs — detection mechanisms, recovery
paradigms, checkpointing systems, convergence analysis, and a goodput model. This entry
designs naniGPT's own system, informed by that survey.

Each design decision follows the same pattern: here's the problem, the simplest solution
(the 101), where it breaks at scale (the cracks), and what the more sophisticated version
looks like (the 201). The design and implementation should be extensible enough that
upgrading from 101 to 201 is a parameter change or a component swap, not a rewrite.


## 1. Scoping: what fault tolerance actually touches

The goal is a fault tolerance architecture that doesn't need to be redesigned as scale
changes. The same components, interfaces, and communication patterns should work at 8
GPUs and 100K — tuning parameters (timeouts, checkpoint intervals, pool sizes) will
change, but the architecture shouldn't.

FT in naniGPT exists to understand the design space, not to run production training. That
means we optimize for observability and clarity over raw performance — every component
should be instrumented, every decision should be measurable, and the system should make
it easy to answer "where did time go during recovery?"

### The surface area

Fault tolerance isn't one component — it's a cross-cutting concern that touches nearly
everything in a training system. Before making any design decisions, it's worth mapping
what it actually touches:

**The training loop.** When a failure is detected, who decides what happens? Does the
training loop itself understand partial failures, or does something external kill and restart it?
Either way, the training loop's contract with the rest of the system has to be defined.

**The communication layer.** NCCL collectives are synchronous — one dead rank blocks
every other rank in the group. Detection, timeout behavior, and process group
reconfiguration all live here. This is the layer where most failures *surface*, even
when the root cause is elsewhere (a GPU error, a NIC failure, a host kernel panic).

**The data pipeline.** When a replica fails and recovers, where does it resume in the
data stream? If we're doing elastic training with fewer replicas, do the survivors absorb
the dead replica's data partition to maintain global batch size? The data pipeline needs
to be failure-aware, or at the very least restartable from a known position.

**The checkpointing system.** Every recovery path ends at a checkpoint. How often do we
save? Where (memory, local disk, remote storage)? How fast can we load? Checkpoint
design directly determines recovery time, which directly determines goodput.

**The process lifecycle.** How do training processes get started, monitored, interrupted, and
restarted? In a Kubernetes world, this is pod management — liveness probes, restart
policies, sidecar containers. The lifecycle layer is what connects detection ("this
process is unhealthy") to recovery ("start a new one").

**The parallelism configuration.** HSDP group sizes, the number of DP replicas, TP
degree — these aren't just performance decisions. They determine the fault domain (what
can you lose and keep training?), the redundancy (how many copies of the model state
exist?), and the blast radius (how much throughput do you lose per failure?). Fault
tolerance requirements should feed back into parallelism choices, not be an afterthought.

The rest of this entry works through design decisions across these surfaces. The
ordering is roughly: choose a recovery strategy (section 2), design the detection
layer (section 3), design the recovery mechanism (section 4), integrate with
checkpointing (section 5), and tie it together (section 6).

### The decision framework

Before designing anything, it's worth stepping back to understand what we're designing
*for*. A pre-training run where fault tolerance matters is typically the "hero" run —
the one you've invested weeks of preparation into. A lot of design decisions go into
this: parallelism strategy and HSDP group sizes for maximizing MFU, scaling laws for
producing the best model (i.e. minimizing loss for a given compute budget), and for
fault tolerance, maximizing goodput. These interact. Your parallelism config determines
your fault domain. Your checkpoint interval affects both goodput and training throughput.
Your batch size determines how much gradient signal you lose when a replica goes down.
Fault tolerance isn't a standalone system — it's one axis of a multi-dimensional
optimization, and it needs to compose cleanly with the others.

With that framing, the recovery strategies form a spectrum of complexity and goodput:

**Checkpoint-restart** is the simplest. Any failure kills the job, you reload from the
last checkpoint, restart NCCL, resume. The training code doesn't know about failures.
The cost is dead time: cold recovery at 98K GPUs takes ~15 minutes, and with an MTBF of
18 minutes, you're training less than half the time.

**Checkpoint-restart with hot swaps** is slightly more complex but has better goodput.
Instead of cold-starting replacement machines through the cluster scheduler, you maintain
a pool of pre-validated standbys. The recovery is still stop-and-restart, but the "stop"
is shorter because machine allocation is instant. ByteRobust reports 10.87x faster recovery 
vs the cold requeue path.

**Elastic HSDP** is the most complex but minimizes downtime. Healthy replicas continue
training while the failed one recovers. The cost is twofold. First, implementation
complexity: quorum protocols, process group reconfiguration, zero-gradient catch-up,
learning rate adjustment. Second, it affects your numerics — running with fewer replicas
changes the effective batch size, which changes the gradient noise, which changes the
loss trajectory. The previous entry's convergence analysis (section 6) showed this
doesn't hurt final model quality in Meta's experiments, but it does mean you need
mechanisms like record/replay to study the numerical behavior and distinguish "loss
spike from elastic event" from "loss spike from bad data." The debugging surface area
grows. Meta reports 80% effective training at 100K GPUs vs 44% for checkpoint-restart,
but that goodput comes with operational complexity that checkpoint-restart avoids
entirely.

In large-scale runs, complexity has a real cost beyond engineering time — it's harder to
debug, harder to reason about failure modes, and harder to have confidence that the
system is doing what you think it's doing. So we want to be systematic and principled
about when to reach for more complexity.

The previous entry sketched a
[goodput model](2026-02-21-fault-tolerance.md#7-modeling-goodput-mathematically) that
provides a rough framework for reasoning about this. For a given GPU count N with
per-server failure rate p, checkpoint interval N_ckpt, and recovery time T_cold or
T_elastic:

    goodput_sync    ≈ 1 - T_s/(N_ckpt·T_iter) - (N_ckpt·T_iter/2 + T_cold)/MTBF
    goodput_elastic ≈ 1 - T_s/(N_ckpt·T_iter) - T_elastic/(R·MTBF)

This model is simplified — it assumes memoryless failures, at most one failure per
checkpoint interval, and constant recovery times, all of which break down at extreme
scale. The previous entry noted that Meta's measured MTBF at 100K GPUs is ~3x worse
than linear extrapolation predicts, which this model doesn't capture well. Still, it's
directionally useful: the crossover where elastic starts to meaningfully outperform
synchronous is *somewhere* in the 10-20K GPU range with Meta's measured failure rates.
Below that, failures are rare enough that either paradigm works fine. Above that, the
gap widens quickly. The exact crossover point depends on numbers we don't have
confident estimates for — your actual failure rate, your actual recovery time, the
correlation structure of your failures. The model gives you the shape of the curve, not
the precise coordinates.

This means we want to support all three approaches — checkpoint-restart,
checkpoint-restart with hot swaps, and elastic — as deployment-time configuration
choices, not architectural commitments. That sounds difficult, but the key insight is
that these aren't three separate systems. They share the same detection layer, the same
checkpointing infrastructure, and the same process lifecycle management. The difference
is what happens *after* a failure is detected: does the coordinator restart everything
from cold, swap in a standby and restart the affected group, or tell the healthy
replicas to keep going? That's a policy decision in a controller, not a structural
difference in the code.

If we get the component contracts right — detection, checkpointing, process lifecycle,
recovery policy — each approach is a different composition of the same building blocks.


## 2. Detection

A GPU throws an ECC error. How long until the system knows? That interval —
detection time — is pure waste. The entire cluster is either stalled (waiting for an
NCCL timeout) or, worse, training on garbage (silent data corruption). The naive
is a dead NIC, a kernel panic, or a flaky GPU. Every failure looks the same — a
baseline is NCCL's default timeout: ~30 minutes, regardless of whether the root cause
collective that never completes. Detection is about surfacing the *actual* failure
before it becomes a generic timeout.

### 2.1 What can go wrong

The previous entry's analysis of failure modes across Meta, ByteDance, and Google
converges on four categories, ordered by increasing difficulty:

**Explicit failures.** A GPU throws a CUDA error, a NIC goes down, a host kernel
panics — somewhere in the stack, an error code exists. The problem is that this error
code lives in the driver or hardware layer, and the vanilla PyTorch training loop never
sees it. Your Python process calls `dist.all_reduce()`, which calls into NCCL, which
talks to the GPU driver, which talks to the hardware. If a NIC dies, the driver knows.
But your training process blocks, waiting for a collective that will never complete,
until the timeout fires. The error is explicit *at the hardware level* but invisible
*at the training level* without an observer to surface it.

**Implicit failures (hangs).** No component is reporting an error. An NVLink connection
degrades, and CUDA kernels stall on load/store operations that will never complete — but
from the GPU's perspective it's just a very slow memory access, not an error. ByteDance
reports that job hangs account for 9.9% of all incidents *after* active inspection has
already caught the failures with identifiable hardware signatures. Without active
inspection, the implicit failure rate would be much higher — many "explicit" failures
only become explicit if you have an observer watching for them.

**Stragglers.** Nothing is broken, but something is slow. One rank is thermally
throttled, or a NIC is congested, or a data loader is falling behind. In synchronous
training, the slowest rank determines the speed of the entire group. Stragglers don't
produce errors or hangs — they silently reduce throughput for thousands of GPUs without
any single component being "broken."

**Silent data corruption (SDC).** The hardest case. The hardware computes a wrong
answer and reports success. No error, no hang, no slowness — just wrong gradients
propagating through your model. At Meta's scale, SDC accounted for 5.5% of all
interruptions, with 37 incidents traced to just 7 hosts. SDC is rare per-step but
compounding — a corrupted gradient affects every subsequent step until detected.

Each category requires a fundamentally different detection approach. Explicit failures
need an observer layer that surfaces hardware errors before they become NCCL timeouts.
Implicit failures need instrumentation that identifies outliers across ranks. Stragglers
need per-phase timing decomposition. SDC needs checksums or replay. The 101/201
progression in each case follows the same pattern: start with a cheap, coarse-grained
signal, and add precision when the coarse signal isn't enough.

### 2.2 The sidecar pattern

The core principle: detection must never block the training loop. If your observability
infrastructure sits on the critical path — if checking GPU health adds latency to every
training step — your system isn't production-ready. Detection runs alongside training,
not inside it.

The universal pattern across every system we surveyed is a **sidecar process** per node
that watches hardware and training state and reports outward. ByteRobust
calls it the Robust Agent (~5,000 lines of Python). axlearn runs an FT Agent as the
pod's main process with the trainer as a subprocess. NCCLX embeds a watchdog thread
inside the communication library itself. The implementations differ, but the topology
is the same: an observer per node, a coordinator above.

    ┌────────────────────────────── K8s Pod ─────────────────────────────────┐
    │                                                                        │
    │  ┌─────────────────────┐         ┌──────────────────────────────────┐  │
    │  │   Sidecar Process   │         │       Training Process           │  │
    │  │                     │  polls  │                                  │  │
    │  │  - GPU/NIC health  ◄├─────────┤  forward → backward → allreduce  │  │
    │  │  - process liveness │         │                                  │  │
    │  │  - phase timers     │         │         GPU 0 .. GPU N           │  │
    │  │                     │         │                                  │  │
    │  └─────────────────────┘         └──────────────────────────────────┘  │
    │                                                                        │
    └────────────────────────────────────────────────────────────────────────┘

The sidecar is a simple polling loop:

```python
def run_sidecar(trainer_process, report_fn, poll_interval=5.0):
    while True:
        status = HealthStatus()

        # check hardware
        for gpu in get_local_gpus():
            status.gpu[gpu.id] = query_cuda_device_status(gpu)
        for nic in get_local_nics():
            status.nic[nic.id] = query_nic_status(nic)

        # check training process
        status.trainer_alive = trainer_process.is_alive()
        status.trainer_step = read_shared_step_counter()

        # report (push to whoever is listening)
        report_fn(status)

        time.sleep(poll_interval)
```

**The interface.** The sidecar needs to expose two things:

1. **Health status**: a structured report of what this node looks like right now. Is the
   GPU responsive? Are the NICs up? Is the training process alive? This is the data
   that recovery decisions are made from.

2. **Diagnostics**: when something is wrong, *what* is wrong. Not just "unhealthy" but
   "GPU 3 has uncorrectable ECC errors" or "NIC eth1 has been flapping for 30 seconds."
   This is what determines whether to evict the node, restart the process, or flag for
   manual investigation.

The sidecar should be queryable (something can pull status on demand or on a heartbeat)
and it should push alerts for urgent events (a GPU falling off the bus shouldn't wait
for the next poll interval). Who receives these reports and what they do with them is a
recovery and coordination question — section 4 and 5.

**101: heartbeat + basic hardware polling.** The simplest useful sidecar polls CUDA
device status, NIC state, and training process liveness on a timer (say, every few
seconds). It reports via a health endpoint. If the sidecar itself stops reporting
(heartbeat timeout), the node is assumed dead. This catches explicit failures within a
poll interval and catches node-level crashes via heartbeat timeout. It doesn't identify
*what* broke — just *that* something broke.

**Where the 101 breaks.** Two gaps. First, the poll interval sets a floor on detection
time — a 5-second poll means up to 5 seconds of delay before you even know something
is wrong. At scale, this adds up across many failure events. Second, basic polling
catches hardware that's clearly broken (CUDA error, NIC down) but misses degraded
hardware — a GPU that's responding but slow, a NIC that's up but dropping packets, an
NVLink that's functional but at reduced bandwidth. These show up as stragglers, not as
failures, and the basic sidecar won't flag them.

**201: active inspection with root-cause classification.** The sidecar monitors
additional channels: CUDA events on every collective (NCCLX's watchdog pattern — poll
`cudaEventQuery` every 100ms), RDMA counters, stdout/stderr of the training process,
and training metrics (loss, gradient norm, MFU). When something looks wrong, the sidecar
classifies the failure — not just "unhealthy" but one of a set of known failure types
(NCCLX's Analyzer uses 24 verdict types, from `STUCK_INSIDE_NCCL` to
`CHECKSUM_MISMATCH` to `FLAKY_OR_SLOW_CONNECTION`). Downstream, this classification
enables informed decisions: a flaky NIC might warrant monitoring, while a GPU with
uncorrectable ECC errors warrants immediate eviction.

The 201 also introduces cross-rank correlation. No single node's sidecar can determine
whether the problem is local or systemic. If 7,999 processes are stuck in
`ncclAllReduce` but 1 is stuck in `cudaMalloc`, that outlier is probably the root cause
(ByteRobust's stack-trace clustering). Someone needs to aggregate per-node reports and
look for patterns — this is where a global view across all sidecars becomes essential.
We'll return to who does this aggregation and how in section 5.

### 2.3 Straggler detection

Stragglers are the subtlest detection problem because nothing is *wrong* — something is
just *slow*. And the symptom is always the same: step time increases. The challenge is
decomposing that symptom into a root cause.

A training step has four phases, and a straggler in any of them produces the same
end-to-end slowdown:

    ┌──────────┐ ┌─────────┐ ┌──────────┐ ┌───────────┐
    │data load │→│ forward │→│ backward │→│ allreduce │
    └──────────┘ └─────────┘ └──────────┘ └───────────┘

    Step is slow + forward starts late           → data loader straggler
    Step is slow + forward/backward are slow     → compute straggler
    Step is slow + fwd/bwd fine + allreduce slow → communication straggler

A data loader straggler is particularly tricky because it's a CPU-side bottleneck that
looks like a GPU problem from the outside. The data loader runs on CPU — tokenization,
preprocessing, batch assembly, host-to-device transfer. If any of these fall behind,
the GPU sits idle waiting for the next batch. In a synchronous setup, one slow data
loader means one rank starts its forward pass late, which means every other rank waits
at the allreduce. The symptom is indistinguishable from a slow GPU unless you have
per-phase timing.

**101: end-to-end step timing.** Time each training step. Maintain a rolling baseline
(median of last N steps). Flag steps that exceed the baseline by some threshold (e.g.,
1.5x). This tells you *that* you have a straggler problem and *when* it started, but
not *which rank* or *why*. At small scale, this is often enough — you have few enough
nodes that manual investigation is tractable.

**Where the 101 breaks.** At scale, "some step was slow" isn't actionable. You have
thousands of ranks and you need to know which one is the bottleneck. And transient
stragglers (thermal throttling that lasts a few seconds, a data loader hiccup from a
slow disk read) might not produce a clear signal in end-to-end timing — the step is a
little slow but within noise.

**201: per-rank per-phase breakdown.** Instrument each phase of the training step with
CUDA events or timers: data loading time, forward time, backward time, allreduce time.
Report per-rank. Correlate across ranks: if rank 47's data loading time is 3x the
median while everyone else is normal, that's a data loader straggler on rank 47. If
rank 47's allreduce time is high but its forward/backward are fine, that's a
communication straggler — likely a NIC or switch issue on that node.

This is NCCLX's CollTrace approach applied more broadly — CollTrace times every
collective per-rank, and the Analyzer correlates across ranks to find outliers. The
extension is to apply the same pattern to non-communication phases (data loading,
compute) so you can distinguish the full range of straggler types.

The action on stragglers is less clear-cut than for failures. A failed GPU gets evicted.
A slow GPU might be worth tolerating if eviction and recovery cost more throughput than
the straggler does. The system needs a policy: at what point does a straggler become
worth evicting? This is a goodput optimization — the cost of tolerating the straggler
(reduced throughput for the group) vs the cost of evicting it (recovery time + potential
over-eviction of the whole parallel group). We won't solve this in the 101, but the
instrumentation needs to be there so the data exists when we need it.

### 2.4 Silent data corruption

SDC is fundamentally different from every other failure type. There's no error, no hang,
no slowness — the hardware computes a wrong answer and reports success. The detection
mechanism can't be "watch for something going wrong" because from every observable
signal, everything looks fine. Instead, SDC detection requires actively *checking*
that results are correct.

The previous entry described an escalation ladder that chains increasingly expensive
checks, with each level only triggered when the cheaper check is inconclusive:

    step N completes
         │
         ▼
    ┌─────────────┐    no spike
    │ 1. DETECT   │──────────────────────► continue training (99.75% of steps)
    │ loss/gnorm  │
    │ vs baseline │
    └─────┬───────┘
          │ spike detected
          ▼
    ┌─────────────┐    replay matches
    │ 2. CONFIRM  │──────────────────────► real spike, not SDC (94% of replays)
    │ replay step │
    └─────┬───────┘
          │ mismatch
          ▼
    ┌─────────────┐    checksums match
    │ 3. LOCALIZE │──────────────────────► transient, schedule monitoring
    │ checksummed │
    │ replay      │
    └─────┬───────┘
          │ checksum mismatch on rank R
          ▼
    ┌─────────────┐
    │ 4. REMOVE   │──────────────────────► evict node, swap standby
    │ evict node  │
    └─────────────┘

The first level — monitoring loss and gradient norm against a rolling baseline — is
cheap, since you're already computing these values. A spike (loss exceeding e.g. 1.4x
baseline, or grad norm exceeding 2.0x) triggers escalation. The baseline calculation
matters: it needs to resist previous spikes inflating it, or one SDC event makes the
detector blind to the next. Combining the window median with a recent-half mean works
well.

The second level — replay — requires deterministic (or near-deterministic) execution.
Replay the step with identical inputs; if the replay produces the same loss, the spike
was real (bad data, training dynamics), not SDC. If it produces different values, you
have a hardware fault or software bug. 

The third level — checksummed replay — attaches checksums to collective communication
inputs and outputs during replay. A checksum mismatch on a specific collective's inputs
identifies the node that produced corrupted data.

The key design insight is that each level is more expensive than the last, so you only
escalate when the cheaper check is inconclusive. Most steps never trigger level 1. Most
spikes are real and stop at level 2. Only genuine SDC reaches levels 3 and 4. For
naniGPT, the escalation ladder is the right pattern — the open question is how far up
the ladder we can practically climb on PyTorch, given the determinism constraints.


## 3. Checkpointing

Every recovery path — checkpoint-restart, hot swap, elastic — ends at a checkpoint.
It's the universal dependency. The speed of your checkpointing system sets a floor on
your recovery time, and the frequency of your checkpoints determines how much work you
lose per failure.

### What's in a checkpoint

A training checkpoint is everything you need to resume from a given step:

- **Model parameters.** The weights themselves.
- **Optimizer state.** For Adam, this is a float32 copy of parameters plus momentum and
  variance — 3x the parameter memory. For a 70B model in bf16, that's ~140GB of
  parameters plus ~420GB of optimizer state.
- **Dataloader position.** Which samples have been consumed. Without this, you either
  repeat data (biasing training) or skip data (wasting tokens).
- **RNG state.** Random number generator state for all devices. Required for
  deterministic replay and for ensuring dropout/augmentation don't repeat patterns
  after recovery.

The total checkpoint size scales linearly with model size and is dominated by optimizer
state. This is the fundamental tension: bigger models need more bytes saved, but the
I/O bandwidth of your storage doesn't grow with model size.

### Storage tiers

Not all checkpoints need the same durability. Production systems use multiple tiers
simultaneously, each trading durability for speed:

    speed       ┌──────────────┐
    ◄───────────┤  in-memory   │  fastest recovery, no disk I/O
                │  (GPU/CPU)   │  lost on node failure
                ├──────────────┤
                │  host memory │  fast recovery, survives GPU failure
                │  (pinned CPU)│  lost on node failure
                ├──────────────┤
                │  local disk  │  survives process crash
                │  (NVMe SSD)  │  lost on node failure
                ├──────────────┤
                │  remote      │  survives node failure
                │  (HDFS/S3)   │  slowest, but durable
    ───────────►└──────────────┘
    durability

The key insight from the previous entry's survey: fast-path recovery (in-memory or host
memory) and durable checkpoints (remote storage) serve different purposes and should
coexist. In-memory snapshots handle the common case — a single replica fails, recovers
from a peer's memory in seconds. Durable checkpoints handle the uncommon case —
correlated failures, cluster restarts, cross-stage transitions — where in-memory state
is gone.

For fault tolerance specifically, the question is: how fast can a recovering replica
get state from a surviving peer? If we're doing elastic recovery, the answer should be
"P2P transfer from a surviving DP replica's memory" — not "read from remote storage."
Remote storage is the backstop, not the fast path.

### The frequency tradeoff

How often should we save durable checkpoints? The tension:

- **Too often:** checkpoint stall overhead eats into training time. Even with async
  saving, there's a minimum blocking time for the D2H copy.
- **Too rarely:** when a failure hits, you lose more training progress. The expected
  lost work per failure is half the checkpoint interval (failure is uniformly
  distributed within the interval).

Young's formula gives the optimal interval: `T_ckpt = sqrt(2 * MTBF * T_save)`, where
`T_save` is the checkpoint save time. The intuition: if saving is fast, checkpoint
often. If saving is slow, space them out and accept more lost work per failure.

This interacts with the recovery strategy. For elastic recovery, durable checkpoints
matter less for single-failure recovery (you recover from a peer's memory). But they
still matter for correlated failures, and they're essential for operational needs
(evaluation, cross-stage transitions, debugging). The frequency decision should be
driven by the correlated failure rate, not the single-failure rate — which is
substantially lower.

### 101: synchronous save to persistent storage

The simplest approach: every N steps, block training, write the full checkpoint to
remote storage, resume. No pipelining, no tiers — just `torch.save()` to a shared
filesystem.

This works at small scale where checkpoints are small (a few GB) and MTBF is long
(days). The save time is a rounding error in the total training time.

**Where it breaks.** Checkpoint size grows with model size. A 70B model's full
checkpoint (params + optimizer) is ~560GB. Writing that synchronously to HDFS takes
minutes. Meanwhile, the entire cluster is idle. At scale where MTBF approaches the
checkpoint interval, you're losing a significant fraction of training time to
checkpointing alone — and you haven't even failed yet.

### 201: async pipeline with tiered storage

The save path becomes a pipeline: snapshot tensors to pinned CPU memory (sub-second
blocking), then serialize and upload to persistent storage in the background while
training continues. The previous entry covered ByteCheckpoint's implementation in
detail — the key result is sub-second stalls (0.34-0.59s) even at 8,960 GPUs, with
end-to-end save times of 20-51 seconds running entirely in the background.

The tiered approach for fault tolerance:

- **Every step (or every few steps):** snapshot to host memory. This is the fast-path
  recovery source. Sub-second overhead. Lost on node failure.
- **Every N steps:** async save to persistent storage. This is the backstop. Seconds of
  background I/O, sub-second blocking. Survives everything.

The frequency of each tier is tuned independently. The in-memory snapshot frequency is
cheap enough to run every step. The persistent save frequency is driven by the ETTR
formula and the correlated failure rate.

For P2P recovery in elastic mode, a surviving replica sends its in-memory snapshot
directly to the recovering replica. No disk I/O on the recovery path at all — the
persistent checkpoint is only needed when in-memory state is unavailable.


## 4. Recovery

Let's trace through the full recovery sequence for each strategy. Along the way, we'll
discover what each component — the training loop, the data loader, the process group —
needs to support.

### 4.1 Checkpoint-restart

The simplest path. A failure is detected; everything stops; we start over from the last
checkpoint.

    failure detected
         │
         ▼
    all ranks stop
         │
         ▼
    allocate replacement machine (or reuse existing)
         │
         ▼
    NCCL re-initialization across all ranks
         │
         ▼
    load checkpoint from persistent storage
         │
         ▼
    data loader seeks to checkpoint position
         │
         ▼
    resume training

Each step surfaces a requirement:

**Machine allocation.** Either cold (request from cluster scheduler — minutes) or warm
(draw from a standby pool — seconds). The hot-swap variant of checkpoint-restart is
just this step being fast. Everything else in the sequence is the same.

**NCCL re-initialization.** Every rank establishes connections with every peer in its
communication groups. This is where scale hurts: NCCL init grows superlinearly with
GPU count (the previous entry documented 17s at 16K GPUs, ~200s at 98K GPUs). The
process group needs to be constructable from scratch — there's no incremental
reconfiguration in the checkpoint-restart path, you tear down everything and rebuild.

**Checkpoint loading.** The replacement needs model state. There are two sources:

- *Persistent storage:* read from HDFS/S3. This is the cold path — bounded by storage
  I/O bandwidth. The previous entry covered ByteCheckpoint's zero-redundancy loading
  where ranks split the reads and broadcast to peers.
- *P2P from a healthy peer's memory:* the surviving replicas still have the checkpoint
  in host memory (section 3's tiered storage). The replacement can fetch state directly
  from a peer over the network — same mechanism as elastic recovery, just with everyone
  stopped while it happens. Much faster than reading from remote storage.

The P2P path means hot-swap and elastic recovery share the same checkpoint transfer
mechanism. The difference isn't where the state comes from — it's whether the healthy
replicas stop while the transfer happens.

**Data loader resumption.** This is easy to overlook but critical. The data loader needs
to resume from exactly where it was at the checkpoint step — same position in the
dataset, same shuffling state. If it can't, you either repeat data (biasing training) or
skip data (wasting tokens). The simplest approach: include the dataloader's RNG state
and sample index in the checkpoint. On recovery, seed the loader from checkpoint state
and it produces the same sequence from that point forward.

**The total cost.** All of these are sequential — you can't load the checkpoint until
NCCL is initialized, you can't initialize NCCL until the replacement machine is
allocated. The previous entry measured the end-to-end timeline at ~10-15 minutes at 98K
GPUs. During all of this, every GPU in the cluster is idle.

### 4.2 Elastic recovery

Same failure, fundamentally different sequence. The healthy replicas keep training
while the failed one recovers in parallel.

    failure detected
         │
         ├──────────────────────────────────────┐
         ▼                                      ▼
    healthy replicas                      failed replica
         │                                      │
    detect missing peer                   allocate replacement
    (NCCL timeout or                            │
     quorum protocol)                     start new training process
         │                                      │
    reconfigure process group             receive checkpoint from
    to exclude failed replica             healthy peer (P2P)
    (1st reconfiguration: R → R-1)              │
         │                                      │
    continue training at                  load state
    (R-1)/R throughput                          │
         │                                      │
         │                                ready to rejoin
         │                                      │
    reconfigure process group ◄─────────────────┘
    to include recovered replica
    (2nd reconfiguration: R-1 → R)
         │
         ▼
    recovered replica contributes
    zero gradient on first step
         │
         ▼
    full throughput resumed

New requirements surface at every step:

**Process group reconfiguration — twice.** There are two reconfigurations in the elastic
path: shrink when the failure is detected (R → R-1), and expand when the replacement is
ready (R-1 → R). The healthy replicas can't tear down NCCL and rebuild for either one —
they're still training. The process group needs to support both removing a dead member
and adding a new one without a full restart. This is what torchft's `ProcessGroupBaby`
solves with subprocess isolation, and what NCCLX/FTAR solve with a fault-tolerant
allreduce that can operate on a changing set of members. The key capability: the
allreduce must tolerate membership changes without blocking or crashing. The second
reconfiguration (adding the recovered replica back) introduces a brief stall — FT-HSDP
measured ~100 seconds for the first-step effect from NCCL initialization when the
recovered replica rejoins.

**P2P checkpoint transfer.** Instead of loading from remote storage, the recovering
replica gets state directly from a surviving peer's memory. This is much faster — no
disk I/O, just a network transfer between two nodes. The surviving replica needs to have
a recent snapshot available in host memory (section 3's tiered storage). The transfer
happens in the background while the healthy replicas continue training.

**The zero-gradient mechanism.** When the recovering replica rejoins the process group,
it may not have finished loading its checkpoint yet — the P2P transfer is still in
flight. The approach from FT-HSDP: the recovering replica participates in the allreduce
but contributes zero gradients. This allows it to "warm up" — it's present in the
process group and participating in collectives while the checkpoint transfer completes
in the background. The allreduce averages across all replicas including the zero,
effectively diluting the gradient by `(R-1)/R` for those steps. Once the checkpoint is
loaded, the recovering replica has the same model state as everyone else and contributes
real gradients going forward. No state divergence, no accumulated error.

**Learning rate scaling during degraded operation.** With fewer replicas, the effective
batch size is smaller, which changes the gradient noise. The previous entry's
convergence analysis found that square root LR scaling (`lr *= sqrt(healthy/total)`)
outperforms both no adjustment and linear scaling — it flattens loss fluctuation during
degraded operation without being as aggressive as linear scaling. This is an adjustment
the training loop needs to accept from the recovery system.

**Data loader behavior.** Two options when a replica goes down:
- **Reduce global batch size.** Survivors keep their existing data partitions. The
  effective batch shrinks by `1/R`. Simpler, and the convergence analysis from the
  previous entry shows this doesn't hurt final model quality.
- **Absorb the dead replica's partition.** Survivors each take a slice of the dead
  replica's data range to maintain global batch size. More complex — each survivor's
  data loader needs to be told "you now also cover samples X through Y." axlearn
  implements this with an elastic input pipeline that pads with `target_labels = -1`
  for zero loss contribution when partitions don't divide evenly.

For the 101, reducing global batch size is the right choice — it requires no data
loader changes, and the convergence impact is negligible for the duration of degraded
operation.

**The stall.** Even in elastic recovery, there's a brief stall. The healthy replicas
need to detect the failure (NCCL timeout or quorum) and reconfigure. FT-HSDP measured
~3 minutes of full-cluster stall per failure event (target ~1.5 minutes with bug fixes),
followed by degraded operation at `(R-1)/R` throughput. Compared to checkpoint-restart's
10-15 minutes of *total* cluster idleness, this is a significant improvement — but
it's not zero.

### 4.3 The contracts

We've traced two recovery paths and discovered what each component needs to support.
To name them explicitly:

**Training loop.** Must support two operations beyond normal training:
1. *Stop and resume from checkpoint* (checkpoint-restart mode).
2. *Continue with adjusted configuration* — fewer replicas, different effective batch
   size, possibly adjusted learning rate (elastic mode).

The training loop should not need to know *why* the configuration changed or manage the
recovery process. It receives a new configuration and continues.

**Data loader.** Must be *resumable from a checkpoint position*. Given the RNG state and
sample index from a checkpoint, it must produce the same sample sequence from that point
forward. This is required by both recovery modes.

**Process group.** Must support two modes:
1. *Full reconstruction* from scratch (checkpoint-restart). Standard NCCL init.
2. *Partial reconfiguration* — remove a dead member, continue with survivors, later add
   the recovered member back (elastic). This is the harder requirement and is what
   distinguishes the 201 from the 101.

**Checkpointing system.** Must support two consumption patterns:
1. *Load from persistent storage* (checkpoint-restart). Standard checkpoint loading.
2. *P2P transfer from a peer's in-memory snapshot* (elastic). A surviving replica
   serves its snapshot directly to the recovering replica over the network.

**Sidecar / detection.** Must report health status that is sufficient to distinguish
"this node is dead" from "this node is slow" from "this node is producing wrong
results." The recovery action is different for each.

These contracts are deliberately minimal. Each component does one thing; the
coordination of *when* and *how* they're invoked is someone else's problem. That
"someone else" is the subject of the next section.


## 5. Where does fault tolerance logic live?

We've designed the pieces: sidecars that detect failures, a tiered checkpointing system,
and two recovery paths with explicit component contracts. The coordination question
remains: who receives sidecar reports, decides which recovery strategy to execute, tells
the process group to reconfigure, directs checkpoint transfers, and signals the training
loop to adjust?

There are two approaches in the literature, and they produce very different codebases.

### The framework approach: FT logic inside the training code

The framework approach makes the training code itself fault-aware. torchft is the
clearest example. Here's what the training loop looks like:

```python
def train_step(state, batch, ft_manager, ft_process_group):
    # FT: check quorum before doing anything
    ft_manager.start_quorum(timeout=60)

    # FT: if we're recovering, we need to load state from a peer
    if ft_manager.is_healing():
        state = load_checkpoint_from_peer(ft_manager.recover_src())

    optimizer.zero_grad()
    loss = model(batch)
    loss.backward()

    # FT: scale gradients based on how many replicas are alive
    if ft_manager.num_participants() < ft_manager.total_replicas():
        scale = math.sqrt(ft_manager.num_participants() / ft_manager.total_replicas())
        for p in model.parameters():
            p.grad *= scale

    # FT: use fault-tolerant allreduce instead of normal one
    ft_process_group.allreduce(gradients)

    # FT: two-phase commit — don't apply if any worker in replica had an error
    if ft_manager.should_commit():
        optimizer.step()
    else:
        # roll back — this step didn't count
        pass
```

The FT-prefixed lines are the problem. The quorum check, the healing path, the gradient
scaling, the fault-tolerant allreduce, the commit protocol — each is a reasonable design
choice, but collectively they mean fault tolerance logic is scattered across the
training loop. And this is just the training loop — the data loader, the checkpointing
code, and the process group initialization all have their own FT branches.

In practice, this produces the "if paft/else" problem — conditionals throughout the
codebase that branch on whether fault tolerance is active. Every new feature interacts
with every FT code path. The training code becomes harder to read, harder to test, and
harder to reason about, because the normal training path and the fault-tolerant training
path are interleaved in every component.

This isn't a criticism of torchft's engineering — it's the pragmatic choice given the
programming model. Most training codebases — Megatron, PyTorch FSDP, DeepSpeed — are
SPMD: every rank runs the same program independently, coordinating through explicit
collectives. There is no single controller (the alternative) to put FT logic in. If 
your codebase is SPMD and you need fault tolerance, the framework is the only place it *can* 
go. The alternative requires a fundamentally different architecture that most existing
codebases don't have.

That said, the complexity is real. If the allreduce is responsible for handling missing
ranks, the allreduce *must* know about replica health. If the training loop is
responsible for quorum, the training loop *must* call quorum. The complexity is inherent
to the approach, not to the implementation.

### The single-controller approach: FT logic above the training code

The alternative is to push all coordination into a single process that sits above the
training loop. The training code doesn't know about fault tolerance. It runs steps. If
something breaks, the coordinator handles recovery and gives the training loop a new
world to run in.

The previous entry showed this in Google's Pathways, where the training loop is a
try/except:

```python
while step < final_step:
    try:
        state = jitted_train_step(state)
        step += 1
    except jax.errors.JaxRuntimeError as error:
        elastic_manager.maybe_reshard_down(error=error, ...)
```

The training loop doesn't know about quorum protocols, process group reconfiguration,
or zero-gradient mechanisms. Recovery is "rebuild the mesh, re-jit, restore from
snapshot" — regular single-process Python. The training code stays clean because
there are no FT code paths in the training code. The coordinator owns all of it.

This maps directly onto the contracts from section 4.3. Each component exposes a
minimal interface (the training loop accepts new configurations, the data loader
resumes from a position, the process group supports reconfiguration, the checkpointing
system supports P2P transfer). The coordinator is the only process that knows *when* and
*why* to invoke those interfaces. The components don't need to know about each other.

### The choice: single controller

The case for single-controller goes beyond fault tolerance. It's the right architecture
for large-scale training systems in general, and FT is one of several reasons why.

Consider RL training loops. RLHF, GRPO, and online RL all require an actor-like
coordination pattern: generate rollouts, score them, update the policy, repeat. The
controller manages the interaction between generation and training — deciding when to
sample, when to train, how to distribute work. This is fundamentally a single-controller
pattern. If you build your trainer as a single-controller system, RL loops compose
naturally. If your trainer is SPMD, bolting on RL coordination requires the same kind of
invasive framework changes that FT does — and you end up with "if rl/else" alongside
"if paft/else."

The broader argument: frontier labs need training systems that are long-term
maintainable. The training loop will grow to support FT, RL, online evaluation,
curriculum scheduling, and features we haven't invented yet. Each of these is a
cross-cutting concern. In an SPMD codebase, each one adds conditionals throughout.
In a single-controller codebase, each one is a policy in the coordinator — cleanly
separated from the training code and from each other.

The FT-specific arguments reinforce this:

**The contracts are already clean.** Section 4.3 defined minimal per-component
contracts. Each component does one thing. The coordination logic — receive sidecar
reports, decide recovery strategy, orchestrate the sequence — is a single
responsibility that belongs in a single place. Spreading it across components would
re-introduce the "if paft/else" problem we're trying to avoid.

**The recovery strategy is a policy, not a structural choice.** Section 1 established
that checkpoint-restart, hot swap, and elastic should be deployment-time configuration
choices. A single coordinator that implements these as different policies is natural. If
the policy logic is scattered across the allreduce, the gradient accumulation, and the
optimizer, switching strategies means changing code in multiple components.

**Observability and record/replay.** A single coordinator produces a single decision
log: "at time T, sidecar on node X reported failure Y, I chose recovery strategy Z,
here's the timeline." This is dramatically easier to debug than reconstructing the
sequence from logs scattered across thousands of training processes. More importantly,
the coordinator's log is a complete record of every configuration change during
training — which replicas were alive at each step, what the effective batch size was,
what LR scaling was applied. Recording this is just serializing coordinator state.
Replaying it is feeding that log back into a fresh coordinator. This is exactly the
record/replay capability we identified in section 1 as necessary for debugging elastic
training's numerical behavior — and it falls out naturally from the single-controller
architecture rather than requiring separate instrumentation.

The coordinator's responsibilities:

1. **Aggregate sidecar reports.** Receive health status from all sidecars. Correlate
   across nodes to identify patterns (section 2.2's cross-rank correlation).

2. **Choose recovery strategy.** Based on the failure type, the current cluster state,
   and the configured policy (checkpoint-restart, hot swap, or elastic), decide what
   to do.

3. **Orchestrate recovery.** Direct the sequence: tell the process group to
   reconfigure, tell a healthy peer to serve its checkpoint, tell the replacement to
   load state, tell the training loop to adjust its configuration.

4. **Track state.** Know which replicas are healthy, which are recovering, what step
   each is on. This is the global view that enables informed decisions.

The coordinator is a single point of failure — if it dies, nobody is making recovery
decisions. But there's no reason it can't be fault-tolerant itself. The coordinator's
state is small: the current quorum membership, which replicas are healthy, what step
each is on, the configured recovery policy. This is kilobytes, not the gigabytes of
model state that make GPU process recovery expensive. The coordinator can checkpoint
its own state to persistent storage (or even just a local file) on every decision, and
a standby coordinator can take over via leader election in seconds. Kubernetes makes
this straightforward — run the coordinator as a replicated deployment with a leader
election sidecar. The training loop continues running during a coordinator failover; it
just won't get recovery orchestration until the new leader is elected, which is fast
because there's no GPU state to restore.



## 6. The full picture

Everything from sections 2-5 comes together here. First, what the deployment looks
like. Then, what the controller actually does — in pseudocode, showing that the training
framework stays clean while the controller owns all FT complexity.

### The deployment

Two nodes shown, but the pattern repeats for every node in the cluster. Each node runs
a training process and a sidecar. The controller sits above, receiving reports and
issuing commands. Each process exposes simple endpoints.

    ┌──────────────────────────────────────────────────────────────────┐
    │                        Controller                                │
    │                                                                  │
    │   • /sidecar/report    ← receives health + phase timing          │
    │   • /trainer/configure ← sends config changes to trainers        │
    │   • /checkpoint/serve  ← directs P2P checkpoint transfers        │
    │   • /record            ← logs every decision for replay          │
    │                                                                  │
    └────────┬───────────────────────────────────────┬─────────────────┘
             │                                       │
             ▼                                       ▼
    ┌─────── Node 0 (DP Replica 0) ────────┐ ┌────── Node 1 (DP Replica 1) ────────┐
    │                                      │ │                                     │
    │  ┌────────────┐  ┌────────────────┐  │ │  ┌────────────┐  ┌───────────────┐  │
    │  │  Sidecar   │  │   Trainer      │  │ │  │  Sidecar   │  │   Trainer     │  │
    │  │            │  │                │  │ │  │            │  │               │  │
    │  │ GET /health│  │ POST /configure│  │ │  │ GET /health│  │POST /configure│  │
    │  │ GET /phase │  │ GET /metrics   │  │ │  │ GET /phase │  │ GET /metrics  │  │
    │  │  _timers   │  │ POST /ckpt     │  │ │  │  _timers   │  │ POST /ckpt    │  │
    │  │            │  │   /serve       │  │ │  │            │  │   /serve      │  │
    │  │            │  │ POST /ckpt     │  │ │  │            │  │ POST /ckpt    │  │
    │  │            │  │   /load        │  │ │  │            │  │   /load       │  │
    │  └────────────┘  └────────────────┘  │ │  └────────────┘  └───────────────┘  │
    │                                      │ │                                     │
    │  GPU 0 .. GPU N                      │ │  GPU 0 .. GPU N                     │
    └──────────────────────────────────────┘ └─────────────────────────────────────┘
The endpoints are the component contracts from section 4.3, made concrete. The sidecar
exposes health and phase timing data. The trainer accepts configuration changes and
exposes metrics. The checkpoint system can serve snapshots to peers or load from a peer.
The controller is the only process that calls these endpoints — the components don't
talk to each other directly.

### The actor: a dumb training worker

Training workers here are modeled as Monarch `@endpoint`-style actors — each operation
is a single endpoint that returns a lightweight future immediately. The specific
framework doesn't matter — this could be Monarch actors, Ray actors, or plain HTTP
servers. The point is that the controller makes remote calls to workers, and each call
returns a handle that can be awaited later. Workers are dumb; they execute what they're
told and maintain only local state. The controller owns global state and coordination.

This actor pattern is the same one that RL training loops use — Tinker models
generators, trainers, and inference servers as actors with endpoints, and a controller
orchestrates the rollout → train → inference sync cycle. We're not designing a fault
tolerance system and an RL system separately. We're designing a single-controller
training architecture that handles both. The actor APIs are the same whether the
controller is driving pre-training steps or RL loops.

```python
class TrainerActor:
    """A training worker. Exposes endpoints, executes what it's told."""

    @endpoint
    async def forward_backward(self, batch) -> Future[Metrics]:
        """Forward, backward, allreduce. Returns metrics."""
        loss = self.model(batch)
        loss.backward()
        dist.all_reduce(self.gradients)  # communication is internal to the actor
        return Metrics(loss=loss.item(), grad_norm=self.grad_norm())

    @endpoint
    async def optim_step(self, lr_scale: float = 1.0) -> Future[None]:
        """Apply gradients to update model parameters."""
        if lr_scale != 1.0:
            for p in self.model.parameters():
                p.grad *= lr_scale
        self.optimizer.step()
        self.optimizer.zero_grad()

    @endpoint
    async def snapshot(self) -> Future[None]:
        self.checkpoint_manager.snapshot_to_host_memory()

    @endpoint
    async def serve_checkpoint(self, dst_addr: str) -> Future[None]:
        self.checkpoint_manager.send_to(dst_addr)

    @endpoint
    async def load_checkpoint(self, src_addr: str) -> Future[None]:
        self.checkpoint_manager.recv_from(src_addr)

    @endpoint
    async def reconfigure(self, config: Config) -> Future[None]:
        self.rebuild_process_group(config.world_size, config.rank)
        self.data_loader.seek(config.data_position)
```

The allreduce is inside `forward_backward` — it's an implementation detail of how
gradients are synchronized, not something the controller needs to orchestrate. The
controller cares about "compute gradients" and "apply gradients," matching Tinker's
`forward_backward` / `optim_step` API. The sidecar can still time the allreduce phase
internally for straggler detection without the controller being involved.

No quorum checks. No `should_commit()`. No zero-gradient branches. Each endpoint is a
single, clear operation. The actor doesn't know about fault tolerance.

### Background watchers: observing cluster state

TODO - below here needs more work

The controller needs to know about the cluster's health, but it shouldn't be polling
sidecars synchronously in the training loop — that would add latency to every step.
Instead, background watchers continuously maintain a `ClusterState` object that the
control loop reads at step boundaries.

This follows the K8s controller pattern: a watch on a resource produces events, and
the reconcile loop reads the current state when it's ready to act. The watchers observe;
the control loop decides.

```python
class ClusterState:
    """Maintained by background watchers. Read by the control loop."""
    def __init__(self):
        self.node_health: dict[NodeId, HealthReport] = {}
        self.failed_nodes: list[NodeId] = []
        self.recovered_nodes: list[NodeId] = []

    def needs_shrink(self) -> bool:
        return len(self.failed_nodes) > 0

    def needs_grow(self) -> bool:
        return len(self.recovered_nodes) > 0

async def watch_sidecars(cluster_state: ClusterState):
    """Background task. Polls sidecars, updates cluster_state continuously."""
    while True:
        for node in cluster_state.all_nodes():
            report = await node.sidecar.get_health()
            cluster_state.node_health[node.id] = report
            if not report.healthy and node.id not in cluster_state.failed_nodes:
                cluster_state.failed_nodes.append(node.id)
        await asyncio.sleep(poll_interval)

async def watch_standby_pool(cluster_state: ClusterState, standby_pool):
    """Background task. Recovers failed nodes, signals when ready."""
    while True:
        if cluster_state.failed_nodes:
            for node_id in list(cluster_state.failed_nodes):
                replacement = await standby_pool.allocate()
                source = pick_healthy(cluster_state)
                await source.serve_checkpoint(dst_addr=replacement.addr)
                await replacement.load_checkpoint(src_addr=source.addr)
                cluster_state.recovered_nodes.append(replacement)
                cluster_state.failed_nodes.remove(node_id)
        await asyncio.sleep(1)
```

The watchers run concurrently with the training loop. They update `ClusterState`; the
control loop reads it. There's no interrupt mechanism — the training step either
completes or throws an exception. The watchers provide context so the control loop
knows *what happened* when it's time to act.

### The recovery policy

The policy defines what to do on state transitions. Different policies implement
different strategies over the same actor endpoints:

```python
class RecoveryPolicy:
    async def on_shrink(self, cluster_state, trainers) -> list[TrainerActor]:
        """Called when cluster_state.needs_shrink(). Returns updated trainer list."""
        raise NotImplementedError

    async def on_grow(self, cluster_state, trainers) -> list[TrainerActor]:
        """Called when cluster_state.needs_grow(). Returns updated trainer list."""
        raise NotImplementedError
```

**Checkpoint-restart with hot swap:** stop everyone, wait for recovery, restart.

```python
class CheckpointRestartPolicy(RecoveryPolicy):

    async def on_shrink(self, cluster_state, trainers):
        record(event="failure_detected", nodes=cluster_state.failed_nodes)
        # nothing to do here — the step already failed.
        # the background watcher is recovering failed nodes.
        # we just wait until recovered nodes are available.
        return [t for t in trainers if t.node_id not in cluster_state.failed_nodes]

    async def on_grow(self, cluster_state, trainers):
        # recovered nodes are ready — add them back and reconfigure everyone
        recovered = cluster_state.pop_recovered()
        all_trainers = trainers + recovered
        config = compute_config(all_trainers)
        await asyncio.gather(*[t.reconfigure(config) for t in all_trainers])
        record(event="recovery_complete")
        return all_trainers
```

**Elastic HSDP:** healthy replicas continue, recovery happens in the background.

```python
class ElasticPolicy(RecoveryPolicy):

    async def on_shrink(self, cluster_state, trainers):
        record(event="failure_detected", nodes=cluster_state.failed_nodes)

        # reconfigure healthy trainers to exclude failed replicas (R → R-1)
        healthy = [t for t in trainers if t.node_id not in cluster_state.failed_nodes]
        lr_scale = math.sqrt(len(healthy) / len(trainers))
        degraded_config = compute_config(healthy, lr_scale=lr_scale)
        await asyncio.gather(*[t.reconfigure(degraded_config) for t in healthy])

        record(event="degraded_operation", replicas=len(healthy))
        return healthy  # continue training with fewer replicas

    async def on_grow(self, cluster_state, trainers):
        # recovered replicas ready — add them back (R-1 → R)
        recovered = cluster_state.pop_recovered()
        all_trainers = trainers + recovered
        full_config = compute_config(all_trainers, recovering=recovered)
        await asyncio.gather(*[t.reconfigure(full_config) for t in all_trainers])

        record(event="recovery_complete")
        return all_trainers
```

### The control loop

Everything converges here. The control loop is the training loop — it drives actors
through Tinker-style endpoint calls and handles state transitions at step boundaries
via the policy. Background watchers maintain `ClusterState`; the loop reads it.

FT decisions naturally synchronize at step boundaries. A training step is
forward → backward → allreduce → optimizer step — these are synchronous from the
controller's perspective. You can't meaningfully interrupt mid-step. So the loop has
two paths: **proactive** (check state at the top of each step) and **reactive** (catch
exceptions from a failed step, consult state to understand why).

```python
async def run(trainers, data_loader, policy, cluster_state):
    # background watchers maintain cluster_state concurrently
    asyncio.create_task(watch_sidecars(cluster_state))
    asyncio.create_task(watch_standby_pool(cluster_state, standby_pool))

    step = 0
    while step < num_steps:

        # --- proactive: check watcher state at step boundary ---
        if cluster_state.needs_shrink():
            trainers = await policy.on_shrink(cluster_state, trainers)
        if cluster_state.needs_grow():
            trainers = await policy.on_grow(cluster_state, trainers)

        # --- training step ---
        try:
            batch = await data_loader.next_batch()
            metrics = await trainers.forward_backward(batch)

            trainers.optim_step()
            trainers.snapshot()

            record(step, metrics)
            step += 1

        except StepFailed as e:
            # reactive: step failed, watcher already knows why
            cluster_state.correlate(e)  # enrich with watcher diagnostics
            record(event="mid_step_failure", error=e, diagnosis=cluster_state)
            continue  # will be caught by needs_shrink() next iteration
```

The training step — `forward_backward → optim_step → snapshot` — is the
same four lines regardless of mode. The policy handles everything else. The watchers
provide continuous observability without blocking the loop. And the `record()` calls
capture every decision for the replay log from section 5.

When a step fails (reactive path), the exception propagates up from the actor futures.
The loop doesn't try to diagnose the failure itself — it asks `cluster_state` to
correlate the exception with the watcher's diagnostics. The watcher has been
maintaining health data continuously, so by the time the exception arrives, the
diagnosis is usually already there: "node X's sidecar reported GPU ECC errors 3
seconds ago." The next loop iteration sees `needs_shrink()` and calls the policy.

### What this buys us

**Same actors, same loop, different policy.** At 128 GPUs, instantiate
`CheckpointRestartPolicy`. At 10K+ GPUs, switch to `ElasticPolicy`. The actor code
is identical. The control loop is identical. The watchers are identical. Only the
policy changes.

**FT and RL share the same architecture.** The actor endpoints, the controller-driven
loop, the background watchers — this is the same pattern an RL loop uses. Swap
"on_shrink / on_grow" for "generate rollouts / update policy / sync weights to
inference" and the shape is the same. Building a single-controller training
architecture means fault tolerance and RL are two policies over the same
infrastructure, not two separate systems.

**Record/replay for free.** The `record()` calls throughout the control loop and
policies capture the complete training history: cluster state at each step, every
failure event, every recovery decision, every configuration change. Replaying a
training run is feeding this log into a fresh controller — the actors don't need to
know they're in a replay.
