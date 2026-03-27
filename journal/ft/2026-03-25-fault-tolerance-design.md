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

Everything from sections 2-5 comes together here. First, the architecture at a glance.
Then, each component in detail — actors, the control loop, and the recovery policies.

The controller has two concurrent async concerns: the **health monitor** (observes the
cluster) and the **control loop** (drives training steps). They communicate through
shared state: an **EventSource** queue of typed events, and a **WorldView** state
machine that tracks cluster topology. The health monitor writes; the control loop
reads and acts.

    ┌──────────────────────────────── Controller ──────────────────────────────────┐
    │                                                                              │
    │  HEALTH MONITOR          SHARED STATE            CONTROL LOOP                │
    │  (async loop)                                     (async loop)               │
    │                                                                              │
    │  ┌────────────────┐      ┌──────────────┐        ┌────────────────────────┐  │
    │  │ poll group     │      │ EventSource  │        │ between steps:         │  │
    │  │  leaders       │ ───► │ (event queue)│ ◄───── │  drain → WorldView     │  │
    │  │ poll K8s pods  │write │              │  drain │  needs_reconfigure()?  │  │
    │  │                │      └──────────────┘        │  → policy.handle()     │  │
    │  │ produces:      │                              │                        │  │
    │  │  NodeFailed    │      ┌──────────────┐        │ during steps:          │  │
    │  │  NodeRecovered │      │  WorldView   │ ◄───── │  drain → WorldView     │  │
    │  │  HealthUpdate  │      │              │  read  │  needs_abort()?        │  │
    │  └────────────────┘      │  HEALTHY     │        │  → ncclCommAbort()     │  │
    │                          │  FAILED      │        │                        │  │
    │                          │  RECOVERING  │        │ training step:         │  │
    │                          │  READY_TO_   │        │  forward_backward()    │  │
    │                          │    JOIN      │        │  optim_step()          │  │
    │                          └──────────────┘        │  snapshot()            │  │
    │                                                  │                        │  │
    │                                                  │  ┌──────────────────┐  │  │
    │                                                  │  │ Policy           │  │  │
    │                                                  │  │ ckpt-restart     │  │  │
    │                                                  │  │ or elastic       │  │  │
    │                                                  │  └──────────────────┘  │  │
    │                                                  └───────────┬────────────┘  │
    │                                                              │               │
    └───▲──────────────────────────────────────────────────────────┼───────────────┘
        │ group leader reports                                     │
        │ + K8s watches                                            │ actor endpoints
        │                                                          ▼
    ┌───┴─────────────────────────────────────────────────────────────────────────┐
    │                      DP Replicas (each = group of nodes)                    │
    │                                                                             │
    │  ┌─────────────────────────────┐    ┌─────────────────────────────┐         │
    │  │ DP Replica 0                │    │ DP Replica 1                │         │
    │  │                             │    │                             │         │
    │  │ ┌─────────────────────────┐ │    │ ┌─────────────────────────┐ │         │
    │  │ │ Group Leader (rank 0)   │ │    │ │ Group Leader (rank 0)   │ │         │
    │  │ │ • aggregates sidecars   │ │    │ │ • aggregates sidecars   │ │         │
    │  │ │ • can abort local group │ │    │ │ • can abort local group │ │  ...    │
    │  │ │ • reports to controller │ │    │ │ • reports to controller │ │         │
    │  │ └─────────────────────────┘ │    │ └─────────────────────────┘ │         │
    │  │                             │    │                             │         │
    │  │ Sidecar+Actor  Sidecar+Actor│    │ Sidecar+Actor  Sidecar+Actor│         │
    │  │ (rank 0)       (rank 1) ... │    │ (rank 0)       (rank 1) ... │         │
    │  └─────────────────────────────┘    └─────────────────────────────┘         │
    └─────────────────────────────────────────────────────────────────────────────┘

The rest of this section drills into each component.

### The actor: a dumb training worker

Training workers here are modeled as Monarch `@endpoint`-style actors — each operation
is a single endpoint that returns a lightweight future immediately. The specific
framework doesn't matter — this could be Monarch actors, Ray actors, or plain HTTP
servers. The point is that the controller makes remote calls to workers, and each call
returns a handle that can be awaited later. Workers are dumb; they execute what they're
told and maintain only local state. The controller owns global state and coordination.

This is worth pausing on, because it's the key architectural bet. Building a
single-controller system is harder than bolting PAFT onto an SPMD codebase. But the
payoff extends far beyond fault tolerance. The same actor + controller pattern is
exactly what RL training loops need — Tinker models generators, trainers, and
inference servers as actors, and a controller orchestrates the rollout → train →
inference sync cycle. It's what curriculum scheduling needs — a controller that decides
what data each replica sees and when to change it. It's what online evaluation needs —
a controller that can snapshot weights to an eval actor mid-training and collect results
without interrupting the training loop. Every one of these is a cross-cutting concern
that, in an SPMD codebase, would require its own set of "if X/else" conditionals
scattered through the training code. In a single-controller architecture, each is a
policy or a new actor type — cleanly separated from the training step and from each
other.

We are paying the single-controller tax once, and getting fault tolerance, RL, curriculum
control, online eval, and whatever comes next as different compositions of the same
building blocks.

```python
class TrainerActor:
    """A training worker. Exposes endpoints, executes what it's told."""

    @endpoint
    async def forward_backward(self, batch) -> Future[Metrics]:
        """Forward, backward, allreduce. Returns metrics.

        For pre-training, the controller sends an index into the dataset rather
        than the full batch — each actor has a local data loader that can seek
        to any position. This keeps the controller lightweight (it sends
        integers, not tensors) and naturally gives it control over global data
        ordering, which is the same surface used for elastic data redistribution
        (section 4.2) and curriculum scheduling. For RL, the controller would
        send actual rollout data generated by a separate actor.

        Each phase (data loading, forward, backward, allreduce) is instrumented
        internally — timestamps written to shared memory where the sidecar can
        read them for straggler detection (section 2.3). Metrics include loss
        and grad norm, which feed the SDC escalation ladder (section 2.4).
        """
        batch = self.data_loader.load(data_index)
        loss = self.model(batch)
        loss.backward()
        dist.all_reduce(self.gradients)  # communication is internal to the actor
        return Metrics(loss=loss.item(), grad_norm=self.grad_norm())

    @endpoint
    async def optim_step(self, lr_scale: float = 1.0) -> Future[None]:
        """Apply gradients. lr_scale adjusts for degraded operation
        (section 4.2 — sqrt scaling during elastic recovery)."""
        if lr_scale != 1.0:
            for p in self.model.parameters():
                p.grad *= lr_scale
        self.optimizer.step()
        self.optimizer.zero_grad()

    @endpoint
    async def snapshot(self) -> Future[None]:
        """Snapshot to host memory — the fast tier from section 3.
        Sub-second overhead, lost on node failure. This is the recovery
        source for P2P checkpoint transfer in both hot-swap and elastic."""
        self.checkpoint_manager.snapshot_to_host_memory()

    @endpoint
    async def serve_checkpoint(self, dst_addr: str) -> Future[None]:
        """Send snapshot to a recovering peer (section 4 — P2P transfer
        from a surviving replica's host memory, no disk I/O)."""
        self.checkpoint_manager.send_to(dst_addr)

    @endpoint
    async def load_checkpoint(self, src_addr: str) -> Future[None]:
        """Receive state from a healthy peer. After loading, the actor
        enters zero-gradient warmup (section 4.2) until fully synced."""
        self.checkpoint_manager.recv_from(src_addr)

    @endpoint
    async def reconfigure(self, config: Config) -> Future[None]:
        """Rebuild process group for new topology (section 4.3 contracts)."""
        self.rebuild_process_group(config.world_size, config.rank)

```


### Controller components

**Group leaders** — one per DP replica. Each replica contains multiple nodes (TP/PP
ranks within the replica). The group leader (rank 0 of each replica) aggregates sidecar
reports from its nodes and reports a per-replica health summary upward. This serves two
purposes. First, it reduces the fan-in to the controller — the health monitor talks to
R group leaders instead of N sidecars (where N >> R). Second, the group leader can
handle local urgency without waiting for the controller: if a node in its group fails
mid-collective, the group leader can trigger `ncclCommAbort` within its own process
group immediately, then report the failure upward. The controller decides what to do
next (shrink, replace, etc.), but the abort happens locally and fast.

This is the same pattern as torchft's Manager (one per replica, runs on rank 0) and
axlearn's `replica_manager` (worker_id == 0 in each TPU slice). The group leader
doesn't make recovery decisions — it observes, aggregates, and can abort its local
group for urgency.

**Health monitor** — an async loop in the controller that polls group leaders (not
individual sidecars) and produces typed events into the EventSource queue. Runs
continuously on its own cadence. This is the only component that knows about K8s — swap
it for Slurm or bare metal and everything else is unchanged.

**EventSource** — a queue of typed events (`NodeFailed`, `NodeRecovered`,
`HealthUpdate`). The health monitor writes; the control loop drains. A pod crashing
and a sidecar detecting GPU ECC errors both become the same typed event in the same
queue.

**WorldView** — the state machine. Consumes typed events and maintains the controller's
model of the cluster: which replicas are HEALTHY, FAILED, RECOVERING, or READY_TO_JOIN.
Exposes two queries for the control loop's two modes:

- `needs_reconfigure()` — has the topology changed? Check at step boundaries for
  proactive handling (a recovered node is ready to rejoin, a straggler was flagged).
- `needs_abort()` — is something urgent enough to abort in-flight work? Check during
  the step to avoid waiting for NCCL timeouts.

**Why the health monitor and control loop are separate async concerns.** They operate on
different timescales. The health monitor polls continuously. The control loop runs at
step granularity — one step might take 8 seconds. If the health monitor detects a GPU
failure at t=1s into a step, the control loop needs to know *during* the step, not at
the next step boundary. Separating them lets the controller react to urgent failures
mid-step.

### The controllers

Checkpoint-restart and elastic recovery don't share the same loop shape.
Checkpoint-restart is a simple loop with synchronous recovery. Elastic
is a stateful loop with background recovery, warm init notifications, and multi-phase
reconfiguration. Trying to force both into one loop with a policy handler makes the
simple case carry the complex case's structure.

Instead, each recovery strategy is its own **controller** — a complete `run()`
implementation that owns its loop shape, state management, and async coordination.
The controllers share the same building blocks (actors, EventSource, WorldView,
health monitor), but they're different programs. Switching strategies is choosing
which controller to run, not configuring a generic one.

Both controllers share the same concurrent health monitoring pattern for mid-step
abort:

```python
async def do_step(trainers, data_scheduler, step):
    """The training step. Same in all controllers."""
    data_indices = data_scheduler.next_indices(trainers)
    metrics = await trainers.forward_backward(data_indices)
    trainers.optim_step()
    trainers.snapshot()
    return metrics

async def step_with_monitoring(trainers, data_scheduler, step, event_source, world):
    """Run a training step while concurrently monitoring for urgent failures."""
    step_task = asyncio.create_task(do_step(trainers, data_scheduler, step))

    while not step_task.done():
        world.ingest(event_source.drain())
        if world.needs_abort():
            await abort_collectives(trainers)
            break
        await asyncio.sleep(0.1)

    return await step_task  # raises StepFailed if aborted
```

**CheckpointRestartController** — the simple case. On failure, stop everything, get a
replacement, load checkpoint, reconfigure, resume.

```python
class CheckpointRestartController:
    async def run(self, trainers, event_source, world, data_scheduler):
        step = 0
        while step < num_steps:
            # ingest events, check for failures
            world.ingest(event_source.drain())

            if world.has_failed():
                failed = world.failed_nodes()
                healthy = world.healthy_trainers()
                record(event="failure_detected", nodes=failed)

                # get replacement from standby pool
                for node_id in failed:
                    standby = world.get_standby()
                    # P2P checkpoint from healthy peer's memory
                    source = healthy[0]
                    await source.serve_checkpoint(dst_addr=standby.addr)
                    await standby.load_checkpoint(src_addr=source.addr)
                    world.mark_active(standby)

                # reconfigure all trainers with new topology
                trainers = world.active_trainers()
                config = compute_config(trainers)
                await asyncio.gather(*[t.reconfigure(config) for t in trainers])
                record(event="recovery_complete")

            # training step
            try:
                metrics = await step_with_monitoring(
                    trainers, data_scheduler, step, event_source, world
                )
                record(step, metrics)
                step += 1
            except StepFailed as e:
                record(event="step_failed", error=e)
                continue
```

**ElasticController** — the stateful case. On failure, shrink and keep training.
Recovery happens in the background. When the replacement is warm-initialized, grow
back.

```python
class ElasticController:
    async def run(self, trainers, event_source, world, data_scheduler):
        step = 0
        pending_warmup = None  # background recovery task

        while step < num_steps:
            world.ingest(event_source.drain())

            # --- shrink: a node failed, reconfigure with fewer replicas ---
            if world.has_failed():
                failed = world.failed_nodes()
                record(event="failure_detected", nodes=failed)

                # reconfigure healthy trainers to exclude failed (R → R-1)
                trainers = world.healthy_trainers()
                lr_scale = math.sqrt(len(trainers) / (len(trainers) + len(failed)))
                config = compute_config(trainers, lr_scale=lr_scale)
                await asyncio.gather(*[t.reconfigure(config) for t in trainers])
                record(event="degraded_operation", replicas=len(trainers))

                # mark failed nodes for cloud eviction if hardware-related
                for node_id in failed:
                    if world.failure_reason(node_id).is_hardware:
                        mark_for_cloud_eviction(node_id)

                # kick off background recovery: get standby, warm init
                pending_warmup = asyncio.create_task(
                    self._warm_init(world, trainers)
                )

            # --- grow: warm init completed, add recovered replica back ---
            if pending_warmup and pending_warmup.done():
                warmed_node = pending_warmup.result()
                pending_warmup = None

                # reconfigure all trainers to include recovered (R-1 → R)
                trainers.append(warmed_node)
                config = compute_config(trainers, recovering=[warmed_node])
                await asyncio.gather(*[t.reconfigure(config) for t in trainers])
                record(event="recovery_complete")

            # training step
            try:
                metrics = await step_with_monitoring(
                    trainers, data_scheduler, step, event_source, world
                )
                record(step, metrics)
                step += 1
            except StepFailed as e:
                record(event="step_failed", error=e)
                continue

    async def _warm_init(self, world, healthy_trainers):
        """Background task: get standby, load checkpoint, return warmed node."""
        standby = world.get_standby()
        source = healthy_trainers[0]
        await source.serve_checkpoint(dst_addr=standby.addr)
        await standby.load_checkpoint(src_addr=source.addr)
        return standby
```

The training step is the same three lines in both controllers —
`forward_backward → optim_step → snapshot`. The difference is entirely in the control
flow *around* it. The `CheckpointRestartController` handles recovery synchronously
before the step. The `ElasticController` handles shrink synchronously, kicks off
recovery in the background, and picks up the result when it's ready.

Notice what WorldView does here: the controllers call `world.has_failed()`,
`world.healthy_trainers()`, `world.get_standby()`, `world.mark_active()`. WorldView
tracks all node state — active, standby, failed, recovering — and the controllers
read and write through it. There's no separate NodeManager; WorldView *is* the
complete model of the world.

Let's trace a concrete scenario through the `ElasticController`:

**GPU ECC failure on node-3, mid-step:**
```
t=0.0s  step begins — do_step() fires forward_backward on all replicas
t=1.2s  sidecar on node-3 detects ECC errors, writes to K8s
t=1.4s  health monitor picks it up, puts NodeFailed in EventSource
t=1.5s  concurrent monitor drains it, WorldView marks node-3 FAILED
t=1.5s  needs_abort() returns true — controller triggers ncclCommAbort
t=1.6s  in-flight allreduce on nodes 0-2 aborts immediately
t=1.6s  step_task raises StepFailed, control loop catches, continues
t=1.6s  next iteration: world.has_failed() returns true
t=1.6s  ElasticController shrinks to 3 replicas, starts background warm init
t=1.6s  training continues at 3/4 throughput with sqrt LR scaling
  ...   (background: standby allocated, P2P checkpoint transferred)
t=12s   pending_warmup.done() — warmed node ready
t=12s   ElasticController grows back to 4 replicas, reconfigures all
t=12s   full throughput resumed
```

Detection-to-shrink: ~400ms. Training never stopped on healthy replicas.

### What this buys us

**Same actors, different controllers.** At 128 GPUs, run `CheckpointRestartController`.
At 10K+ GPUs, run `ElasticController`. The actor code is identical. The health monitor
is identical. The WorldView is identical. The controllers use the same building blocks
but have different loop shapes.

**FT and RL share the same architecture.** The actor endpoints, the controller-driven
loop, the async event monitoring — this is the same pattern an RL loop uses. An RL
controller would be a third implementation with its own loop shape (generate rollouts →
score → train → sync weights to inference), using the same actors and event
infrastructure.

**Record/replay for free.** The `record()` calls capture every step, every failure,
every configuration change. Replay is a third controller implementation —
`ReplayController` — that reads from the recorded log instead of live events. It
drives the same actors through the same training steps with the same data indices and
the same configuration changes. The actors don't know they're in a replay.

This is useful for debugging elastic training's numerical behavior (section 1) —
replay a run with recorded failure events and study the loss trajectory, gradient
noise, and LR scaling effects in isolation. It's also useful for validating the FT
system itself — replay a recorded failure sequence through a new controller
implementation and verify it produces the right recovery decisions.


## 7. What's still incomplete

This entry explored the design space and landed on an architecture — single-controller,
actor-based workers, async health monitoring, strategy-specific controllers over shared
components. The structure feels right, but the design isn't bulletproof yet. Before
writing a full design doc or implementation plan, several things need more work:

**Adversarial scenario analysis.** We traced one failure scenario (GPU ECC mid-step)
through the elastic controller. A real design needs to survive harder cases: multiple
simultaneous failures, failure during reconfiguration (some trainers have the new
config, some have the old), controller failure mid-recovery, cascading failures that
exhaust the standby pool, false positives from sidecars, and SDC during elastic
operation (where loss spikes are ambiguous — is it corruption or expected noise from
fewer replicas?). Each of these could crack an abstraction that looks clean in the
single-failure case.

**WorldView state machine formalization.** We sketched the states (ACTIVE, STANDBY,
FAILED, RECOVERING) but haven't formalized the valid transitions, or what happens
when events arrive in unexpected order. A proper state machine with defined transitions
and error states would catch design bugs before implementation.

**Group leader protocol.** We introduced group leaders as per-replica aggregators that
can abort their local group for urgency. The exact protocol — what the group leader
reports, how often, how the controller discovers group leaders, what happens when a
group leader itself fails — needs to be worked out.

**Standby pool sizing and lifecycle.** How many standbys, how to provision them, how
to health-check before accepting into the pool, how to replenish after use. The
previous entry covered ByteRobust's binomial model for pool sizing, but the operational
details (where do standbys come from, how long do they stay idle, what's the cost
tradeoff) are unresolved.

**Integration with existing training code.** The actor definition assumes clean
endpoints like `forward_backward` and `reconfigure`. Wrapping an existing SPMD training
step (Megatron, FSDP) behind these endpoints is mechanical but non-trivial — especially
process group reconstruction on `reconfigure()`. The gap between the pseudocode and a
working implementation needs to be mapped.

**Testing strategy.** The architecture is testable at every layer in principle
(mock EventSource, mock WorldView, mock actors). But the integration tests — does
the elastic controller actually recover correctly when a real GPU fails on a real
cluster? — require infrastructure we don't have yet.

The goal of this entry was to build enough clarity on the structure that a full design
can proceed with confidence. The next step is picking the hardest adversarial scenarios
above and tracing them through until either the design holds or we learn what needs to
change.
