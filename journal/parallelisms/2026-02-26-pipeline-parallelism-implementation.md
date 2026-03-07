# Implementing Pipeline Parallelism for naniGPT

*2026-02-26*

The [parallelism strategies entry](2026-02-24-parallelism-strategies.md) covers the theory — schedules, bubble fractions, cost formulas. This entry is about what it actually looks like to implement PP for naniGPT. The goal: someone reading this should be able to write GPipe and 1F1B from scratch, knowing where the gotchas are before they hit them.

---

## Why PP first

PP is a good first parallelism to implement because it requires the least framework machinery:

- **The model already has the right structure.** `DenseTransformer.blocks` is a `nn.ModuleList` of identical `TransformerBlock`s. Each block takes `(B, S, d_model)` and returns `(B, S, d_model)` — uniform activation shape between every layer. Splitting is just slicing this list.

From [`nanigpt/models/dense_transformer.py`](../nanigpt/models/dense_transformer.py):
```python
class DenseTransformer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight  # weight tying
```

- **It forces you to learn the foundational distributed primitives.** `dist.send`/`dist.recv`, process groups, `dist.init_process_group` — these are the same primitives that every other parallelism builds on, but PP uses them in their rawest form (point-to-point, not collectives). No DTensor, no compiler integration, no NCCL magic beyond basic send/recv.

- **The training loop changes are localized.** The model code, optimizer, and data loading are mostly unchanged. The big change is the training step itself — it becomes a schedule of microbatch forwards and backwards with P2P communication between them.

- **The scheduling problem is interesting in its own right.** Going from GPipe → 1F1B → interleaved is a progression in algorithm design, not just systems engineering. Each schedule uses the same building blocks but arranges them differently.

---

## The three hard problems

The theory in the strategies entry makes PP sound clean: split the model, send activations forward, send gradients backward. The implementation is trickier than the formulas suggest, and the difficulty comes from three specific problems.

### Problem 1: Autograd across process boundaries

This is the fundamental problem that makes PP implementation non-trivial. Consider two stages on two ranks:

```
  Rank 0 (stage 0)              Rank 1 (stage 1)
  ┌───────────────┐             ┌────────────────┐
  │ x = embed(ids)│             │                │
  │ y = blocks(x) │──send(y)───▶│ y_recv = ???   │
  │               │             │ z = blocks(y)  │
  │               │             │ loss = CE(z)   │
  └───────────────┘             └────────────────┘
```

When rank 0 calls `dist.send(y, dst=1)` and rank 1 calls `dist.recv(y_recv, src=0)`, the received tensor `y_recv` is a **new tensor** with no connection to rank 0's autograd graph. It's just a buffer of floats that happened to arrive over the network.

When rank 1 runs `loss.backward()`, autograd traces back through `z = blocks(y_recv)` and computes `y_recv.grad`. But that gradient has nowhere to go — there's no autograd edge connecting `y_recv` back to rank 0's `y`. The gradient dies at the process boundary.

The fix has three parts:

```python
# Rank 1 (receiving side):
y_recv = torch.empty(shape, dtype=dtype, device=device)
dist.recv(y_recv, src=0)
y_recv.requires_grad_(True)  # (1) Make it a leaf that autograd tracks

z = stage_1_forward(y_recv)
loss = F.cross_entropy(z.view(-1, V), targets.view(-1)) / num_microbatches
loss.backward()
# Now y_recv.grad exists — it's dL/dy

dist.send(y_recv.grad, dst=0)  # (2) Send gradient back to rank 0

# Rank 0 (after receiving the gradient):
grad_from_downstream = torch.empty(shape, dtype=dtype, device=device)
dist.recv(grad_from_downstream, src=1)
y.backward(grad_from_downstream)  # (3) Continue autograd on rank 0
```

Step (1) is the key insight: `requires_grad_(True)` on a tensor that wasn't produced by an autograd op makes it a **leaf tensor** — autograd will accumulate gradients into `.grad` for it during `backward()`. Without this, `y_recv.grad` would be `None` after backward.

Step (3) is the manual autograd continuation. `y.backward(grad_from_downstream)` is equivalent to saying "pretend the loss was some downstream function whose gradient w.r.t. `y` is `grad_from_downstream`, and continue backpropagating from there." This computes gradients for all parameters in stage 0's blocks and for the embedding.

**The pattern:** PP forward sends activations downstream, PP backward sends gradients upstream. The P2P communication in backward *mirrors* forward — same ranks, opposite direction.

### Problem 2: Gradient accumulation across microbatches

With `M` microbatches per training step, each microbatch runs an independent forward-backward through the pipeline. The gradients from all microbatches must be accumulated before the optimizer steps.

PyTorch makes this easy if you know the rule: **don't call `optimizer.zero_grad()` between microbatches.** By default, `loss.backward()` *accumulates* into `.grad` — it adds to whatever's already there. So if you run `M` microbatches worth of forward-backward without zeroing, you get the sum of all `M` gradients.

But this sum should be the *mean*, not the sum. There are two equivalent ways to handle this:

**Option A: Scale the loss.** Divide each microbatch's loss by `M`:
```python
loss = F.cross_entropy(logits, targets) / M  # scale before backward
loss.backward()
```

**Option B: Scale the gradients at the end.** After all microbatches, divide every `.grad` by `M`:
```python
for p in model.parameters():
    if p.grad is not None:
        p.grad /= M
```

Option A is simpler and is what most implementations use (including torchtitan). It's also numerically slightly better — you avoid the large intermediate sum.

The gotcha: the loss you *log* should be the unscaled loss, not the `loss / M` value. Otherwise your loss curves look artificially low.

### Problem 3: Deadlock avoidance

`dist.send` and `dist.recv` are **blocking** by default — `send` blocks until the matching `recv` is posted, and vice versa. If rank 0 tries to send while rank 1 is doing something else, both ranks hang forever.

For GPipe this is almost trivial: the schedule is lock-step. All ranks run forwards in order, then all run backwards in order. The send/recv pairs naturally match because the schedule is fully determined.

For 1F1B it's trickier. During steady state, a rank alternates forward and backward. Its forward sends an activation downstream, and its backward expects a gradient from downstream. The downstream rank is doing the opposite — receiving an activation from upstream and sending a gradient to upstream. If both try to send at the same time, they deadlock.

The standard fix is the **even/odd rank trick**: for any P2P exchange between adjacent ranks, even-numbered ranks send first (then receive), and odd-numbered ranks receive first (then send). This breaks the circular dependency:

```python
def send_recv_pair(send_tensor, recv_tensor, peer_rank, my_rank, group):
    """Exchange tensors with a peer, avoiding deadlock."""
    if my_rank % 2 == 0:
        dist.send(send_tensor, dst=peer_rank, group=group)
        dist.recv(recv_tensor, src=peer_rank, group=group)
    else:
        dist.recv(recv_tensor, src=peer_rank, group=group)
        dist.send(send_tensor, dst=peer_rank, group=group)
```

Alternatively, use `dist.isend`/`dist.irecv` (non-blocking versions) and wait on both:

```python
send_op = dist.isend(send_tensor, dst=peer_rank, group=group)
recv_op = dist.irecv(recv_tensor, src=peer_rank, group=group)
send_op.wait()
recv_op.wait()
```

The non-blocking approach is simpler (no even/odd logic) and lets NCCL overlap the transfers. For a minimal implementation, either works; for performance, prefer `isend`/`irecv`.

---

## Building blocks

Four pieces of code to write, in order.

### 1. Model splitting

```python
def build_stage(
    config: TransformerConfig,
    stage_id: int,
    num_stages: int,
    device: torch.device,
) -> nn.Module:
    """Build only the layers for this pipeline stage."""
```

The full `DenseTransformer` has: `token_emb → drop → blocks[0:N] → ln_f → lm_head`.

Split the `N` blocks into `num_stages` contiguous chunks. Stage 0 gets `token_emb + drop + blocks[0:k]`. Middle stages get `blocks[i*k:(i+1)*k]`. The last stage gets `blocks[(num_stages-1)*k:] + ln_f + lm_head`.

```python
layers_per_stage = config.n_layers // num_stages
assert config.n_layers % num_stages == 0, (
    f"n_layers ({config.n_layers}) must be divisible by num_stages ({num_stages})"
)

start = stage_id * layers_per_stage
end = start + layers_per_stage

blocks = nn.ModuleList([TransformerBlock(config) for _ in range(start, end)])
```

**Design question: build only the local chunk, or build the full model and prune?**

Torchtitan builds the full model (on meta device to avoid memory), then extracts the relevant submodule. This is simpler for complex models with irregular structure. For our uniform transformer, building only the local chunk is more honest about what each rank actually holds and avoids the meta-device dance:

```python
class PipelineStage(nn.Module):
    """One stage of a pipeline-parallel model."""

    def __init__(
        self,
        config: TransformerConfig,
        stage_id: int,
        num_stages: int,
    ):
        super().__init__()
        self.stage_id = stage_id
        self.num_stages = num_stages
        self.is_first = stage_id == 0
        self.is_last = stage_id == num_stages - 1

        layers_per_stage = config.n_layers // num_stages
        start = stage_id * layers_per_stage
        end = start + layers_per_stage

        # First stage: embedding + first chunk of blocks
        if self.is_first:
            self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
            self.drop = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(layers_per_stage)]
        )

        # Last stage: final layernorm + output head
        if self.is_last:
            self.ln_f = nn.LayerNorm(config.d_model)
            self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
            # Weight tying: need the embedding from stage 0.
            # In PP, stage 0 and last stage are different ranks,
            # so we can't tie. Create a separate embedding for the head.
            # This adds ~vocab_size * d_model params but is the simplest approach.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_first:
            x = self.drop(self.token_emb(x))  # (B, S) -> (B, S, d_model)

        for block in self.blocks:
            x = block(x)

        if self.is_last:
            x = self.ln_f(x)
            x = self.lm_head(x)  # (B, S, d_model) -> (B, S, vocab_size)

        return x
```

**Weight tying caveat.** The current `DenseTransformer` ties `lm_head.weight = token_emb.weight`. With PP, these live on different ranks (stage 0 and last stage). Options:
1. **Don't tie.** The last stage gets its own `lm_head` with independent weights. Simplest. Adds `vocab_size × d_model` parameters (~13M for GPT-2 vocab + d_model=256). For our learning purposes, this is fine.
2. **Communicate the embedding.** All-gather the embedding weight from stage 0 to the last stage before the head projection. Correct but adds complexity for a marginal memory save.
3. **Put embedding on all stages.** Wastes memory. Not worth it.

For a minimal implementation, option 1.

**Input/output shape contract.** Stages communicate tensors of shape `(B_micro, S, d_model)` — the hidden activation after blocks. Exception: the first stage's input is `(B_micro, S)` integer token IDs (from the dataloader), and the last stage's output is `(B_micro, S, vocab_size)` logits (consumed by the loss function). Only the inter-stage tensors need to be communicated, and they're always `(B_micro, S, d_model)`.

### 2. P2P communication

```python
def send_activation(tensor: torch.Tensor, dst: int) -> None:
    """Send an activation tensor to the next pipeline stage."""
    dist.send(tensor.contiguous(), dst=dst)

def recv_activation(
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
    src: int,
) -> torch.Tensor:
    """Receive an activation tensor from the previous pipeline stage."""
    tensor = torch.empty(shape, dtype=dtype, device=device)
    dist.recv(tensor, src=src)
    tensor.requires_grad_(True)  # Critical: enable autograd tracking
    return tensor
```

These are deliberately thin. The `requires_grad_(True)` on recv is the most important line — it's what makes Problem 1's solution work. The shape, dtype, and device must be known ahead of time, which is fine because the inter-stage activation shape `(B_micro, S, d_model)` is fixed for a given config.

For gradients, the same functions work — a gradient is just a tensor of the same shape flowing in the opposite direction. No `requires_grad_` needed on gradient tensors (they're never part of an autograd graph themselves).

### 3. Microbatching

```python
# In the training step, before the schedule runs:
input_ids = tokens[:, :-1]   # (B, S)
targets = tokens[:, 1:]      # (B, S)

# Split along batch dimension
input_microbatches = input_ids.chunk(num_microbatches, dim=0)   # M tensors of (B/M, S)
target_microbatches = targets.chunk(num_microbatches, dim=0)     # M tensors of (B/M, S)
```

`torch.chunk` handles the case where `B` doesn't divide evenly by `M` — the last chunk is smaller. But for simplicity, require `B % M == 0` (assert in config validation).

The microbatch size `B_micro = B / M` should be small enough that `M ≥ pp_size` to keep the pipeline reasonably full. The strategies entry shows the bubble fraction is `(pp-1) / M`, so `M = 4 * pp_size` gives a ~25% bubble — a reasonable starting point.

### 4. Schedule

This is where the three problems come together. Start with GPipe (simpler), then upgrade to 1F1B.

---

## GPipe: full implementation walkthrough

The complete GPipe training step, annotated with where each of the three problems shows up.

```python
def gpipe_step(
    stage: PipelineStage,
    microbatches: list[torch.Tensor],     # input_ids chunks, only used by first stage
    target_mbs: list[torch.Tensor],       # target chunks, only used by last stage
    optimizer: torch.optim.Optimizer,
    num_microbatches: int,
    rank: int,
    world_size: int,
    dtype: torch.dtype,
    device: torch.device,
    d_model: int,
    seq_len: int,
    vocab_size: int,
) -> float:
    """One training step using GPipe schedule."""

    prev_rank = rank - 1 if rank > 0 else None
    next_rank = rank + 1 if rank < world_size - 1 else None
    is_first = rank == 0
    is_last = rank == world_size - 1

    # Cache for backward pass
    # Each entry: (input_to_stage, output_of_stage)
    # input_to_stage is what we received (need its .grad for sending upstream)
    # output_of_stage is what we computed (need it to call .backward on)
    saved = []
    losses = []
    total_loss = 0.0

    # Determine shapes for P2P
    mb_size = microbatches[0].shape[0] if is_first else None
    if not is_first:
        # All ranks need to know microbatch size for recv shape.
        # In practice, this comes from config. Here we hardcode the shape.
        mb_size = microbatches[0].shape[0] if is_first else None
        # We'll receive (B_micro, S, d_model) from prev_rank
    activation_shape = (microbatches[0].shape[0], seq_len, d_model)

    # =====================
    #   FORWARD FILL
    # =====================
    for mb_idx in range(num_microbatches):
        if is_first:
            # First stage: run embedding + blocks
            x = microbatches[mb_idx]  # (B_micro, S) token IDs
            with torch.amp.autocast(device_type="cuda", dtype=dtype):
                output = stage(x)
            # Save for backward. For first stage, no received tensor to track.
            saved.append((None, output))
        else:
            # Receive activation from upstream          ← Problem 1
            x_recv = recv_activation(
                activation_shape, dtype, device, src=prev_rank,
            )
            with torch.amp.autocast(device_type="cuda", dtype=dtype):
                output = stage(x_recv)
            saved.append((x_recv, output))

        if is_last:
            # Compute loss on last stage
            with torch.amp.autocast(device_type="cuda", dtype=dtype):
                loss = F.cross_entropy(
                    output.view(-1, vocab_size),
                    target_mbs[mb_idx].view(-1),
                )
                loss = loss / num_microbatches          # ← Problem 2: scale for accumulation
            losses.append(loss)
            total_loss += loss.item() * num_microbatches  # unscaled for logging
        else:
            # Send activation downstream
            send_activation(output.detach(), dst=next_rank)
            # .detach() because we don't need autograd to track across the send;
            # we handle it manually via saved tensors and .backward(grad)

    # =====================
    #   BACKWARD DRAIN
    # =====================
    for mb_idx in range(num_microbatches):
        x_recv, output = saved[mb_idx]

        if is_last:
            # Last stage: backward from loss
            losses[mb_idx].backward()                    # ← Computes output.grad, weight grads
        else:
            # Receive gradient from downstream           ← Problem 1 (gradient direction)
            grad = torch.empty_like(output)
            dist.recv(grad, src=next_rank)
            output.backward(grad)                        # ← Problem 1: continue autograd chain

        if not is_first:
            # Send gradient upstream                     ← Problem 1
            dist.send(x_recv.grad, dst=prev_rank)

    # =====================
    #   OPTIMIZER STEP
    # =====================
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    return total_loss / num_microbatches
```

**Walk through the data flow for 2 stages, 2 microbatches:**

```
  Time    Rank 0 (first stage)              Rank 1 (last stage)
  ────    ──────────────────────            ──────────────────────
  t=1     F0: output₀ = stage(mb₀)         (waiting for recv)
          send(output₀) ──────────────────▶ recv → x_recv₀
  t=2     F1: output₁ = stage(mb₁)         F0: output₀ = stage(x_recv₀)
          send(output₁) ──────────────────▶ loss₀ = CE(output₀, targets₀) / M
  t=3     (waiting for recv)                F1: output₁ = stage(x_recv₁)
                                            loss₁ = CE(output₁, targets₁) / M
  t=4     (waiting for recv)                B0: loss₀.backward()
                                            send(x_recv₀.grad) ──────────▶
  t=5     recv → grad₀                     B1: loss₁.backward()
          B0: output₀.backward(grad₀)      send(x_recv₁.grad) ──────────▶
  t=6     recv → grad₁
          B1: output₁.backward(grad₁)
  t=7     optimizer.step()                  optimizer.step()
```

Note: this is slightly idealized — in reality, the blocking sends/recvs serialize things more. But this shows the logical data flow. The idle time at t=3 and t=4 on rank 0 is the pipeline bubble.

**Deadlock risk in GPipe:** Because all forwards happen before all backwards, and within each phase the ranks execute in pipeline order, the send/recv pairs naturally match. Rank 0 sends, rank 1 receives — they're always in sync. Deadlock is only a risk when a rank could try to send and receive to/from the same peer simultaneously, which doesn't happen in GPipe's lock-step schedule.

---

## 1F1B: a scheduling upgrade

Same data flow primitives, different schedule. The three hard problems don't change — the solution to autograd across boundaries, gradient accumulation, and deadlock avoidance is identical. The only difference is the *order* in which forwards and backwards execute.

### The state machine

Each rank's schedule has three phases:

**Warmup:** Run `num_warmup = num_stages - rank - 1` forward-only microbatches. This fills the pipeline — earlier ranks need more warmup because their outputs have to propagate further before the last stage can start backward.

**Steady state:** Alternate 1 backward + 1 forward, for `num_microbatches - num_warmup` iterations. Each backward frees one microbatch's cached activations before the next forward allocates new ones, keeping peak memory flat.

**Cooldown:** Run the remaining `num_warmup` backwards. These drain the pipeline of the warmup microbatches.

```python
def one_f_one_b_step(
    stage: PipelineStage,
    microbatches: list[torch.Tensor],
    target_mbs: list[torch.Tensor],
    optimizer: torch.optim.Optimizer,
    num_microbatches: int,
    rank: int,
    num_stages: int,
    dtype: torch.dtype,
    device: torch.device,
    d_model: int,
    seq_len: int,
    vocab_size: int,
) -> float:
    prev_rank = rank - 1 if rank > 0 else None
    next_rank = rank + 1 if rank < num_stages - 1 else None
    is_first = rank == 0
    is_last = rank == num_stages - 1

    num_warmup = num_stages - rank - 1
    num_steady = num_microbatches - num_warmup
    num_cooldown = num_warmup

    saved = {}  # mb_idx -> (x_recv, output)
    losses = {}
    total_loss = 0.0
    activation_shape = (microbatches[0].shape[0], seq_len, d_model)

    fwd_idx = 0  # next microbatch to forward
    bwd_idx = 0  # next microbatch to backward

    def run_forward(mb_idx):
        nonlocal total_loss
        if is_first:
            with torch.amp.autocast(device_type="cuda", dtype=dtype):
                output = stage(microbatches[mb_idx])
            saved[mb_idx] = (None, output)
        else:
            x_recv = recv_activation(activation_shape, dtype, device, src=prev_rank)
            with torch.amp.autocast(device_type="cuda", dtype=dtype):
                output = stage(x_recv)
            saved[mb_idx] = (x_recv, output)

        if is_last:
            with torch.amp.autocast(device_type="cuda", dtype=dtype):
                loss = F.cross_entropy(
                    output.view(-1, vocab_size),
                    target_mbs[mb_idx].view(-1),
                ) / num_microbatches
            losses[mb_idx] = loss
            total_loss += loss.item() * num_microbatches
        else:
            send_activation(output.detach(), dst=next_rank)

    def run_backward(mb_idx):
        x_recv, output = saved.pop(mb_idx)  # pop to free memory

        if is_last:
            losses[mb_idx].backward()
        else:
            grad = torch.empty_like(output)
            dist.recv(grad, src=next_rank)
            output.backward(grad)

        if not is_first:
            dist.send(x_recv.grad, dst=prev_rank)

    # ---- Warmup: forward only ----
    for _ in range(num_warmup):
        run_forward(fwd_idx)
        fwd_idx += 1

    # ---- Steady state: 1B then 1F ----
    for _ in range(num_steady):
        # Backward first (frees memory), then forward
        if bwd_idx < fwd_idx:  # have something to backward
            run_backward(bwd_idx)
            bwd_idx += 1
        run_forward(fwd_idx)
        fwd_idx += 1

    # ---- Cooldown: backward only ----
    while bwd_idx < num_microbatches:
        run_backward(bwd_idx)
        bwd_idx += 1

    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return total_loss / num_microbatches
```

### Memory advantage

This is the real win over GPipe. The `saved` dict tells the story:

- **GPipe:** `saved` grows to `M` entries during the forward fill, then drains during backward. Peak = `M`.
- **1F1B:** `saved` grows to `num_warmup` entries during warmup. During steady state, each `run_backward` pops one entry and `run_forward` adds one — the size stays flat. Peak = `num_warmup = num_stages - rank - 1 ≤ num_stages - 1`.

For `num_stages=4, M=64`: GPipe caches 64 microbatches of activations, 1F1B caches at most 3. Each activation is `B_micro × S × d_model × Φ` bytes — for `B_micro=8, S=512, d_model=768, Φ=2`: that's 6 MB per microbatch. GPipe: 384 MB. 1F1B: 18 MB.

The bubble fraction is the same: `(pp-1) / M`. 1F1B's advantage is purely memory.

### Deadlock in 1F1B

During steady state, consider two adjacent ranks in the middle of the pipeline:

```
  Rank k:   ... backward(mb_i) → forward(mb_j) ...
  Rank k+1: ... backward(mb_i) → forward(mb_j) ...
```

Rank k's backward sends a gradient to rank k-1 (upstream) and rank k's forward sends an activation to rank k+1 (downstream). These are different peers, so no conflict.

But rank k's forward *sends* to rank k+1, and simultaneously rank k+1's backward *sends* to rank k. Both are sending to each other at the same time. With blocking `dist.send`, both block waiting for the other's `dist.recv` — deadlock.

This is where `isend`/`irecv` or the even/odd trick (from Problem 3 above) is essential. The 1F1B schedule is more latency-sensitive to this than GPipe because sends and recvs from different microbatches can interleave in time.

---

## Process group setup

Before any P2P communication, initialize the distributed backend:

```python
import torch.distributed as dist
import os

def init_distributed():
    """Initialize process group for pipeline parallelism."""
    dist.init_process_group(backend="nccl")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    return rank, world_size, device
```

Launch with `torchrun`:
```bash
# 2-stage pipeline on 2 GPUs:
torchrun --nproc_per_node=2 -m nanigpt.train_pp

# 4-stage pipeline on 4 GPUs:
torchrun --nproc_per_node=4 -m nanigpt.train_pp
```

For PP, the process group is simple — all ranks are in the same group, and P2P send/recv uses `dist.send(tensor, dst=rank)` directly. No sub-groups needed (unlike TP, where you'd create groups per node). If we later compose PP with DP, we'd create separate process groups for each dimension.

---

## How to test it

### Correctness: loss curve comparison

The most important test: a single-GPU run and a 2-stage PP run should produce the **same loss curve** (within floating point tolerance) when:
- Same model config, same data, same seed
- PP uses `num_microbatches=1` (no gradient accumulation effects)
- Weight tying is disabled on both (to match PP's untied last stage)

```bash
# Single GPU baseline
uv run python -m nanigpt.train --config small-synthetic --training.num-steps 50

# 2-stage PP
torchrun --nproc_per_node=2 uv run python -m nanigpt.train_pp \
    --config small-synthetic --training.num-steps 50 --pp-size 2 --num-microbatches 1
```

If the losses diverge, the gradient flow across the process boundary is wrong — check Problem 1's implementation first.

### Correctness: gradient accumulation

Run with `num_microbatches > 1` and compare against a single-GPU run with the same effective batch size. The losses should match (modulo floating point from different accumulation order).

### Performance: measuring the bubble

Instrument the schedule with `torch.cuda.Event` timers:

```python
fwd_start = torch.cuda.Event(enable_timing=True)
fwd_end = torch.cuda.Event(enable_timing=True)

fwd_start.record()
run_forward(mb_idx)
fwd_end.record()

torch.cuda.synchronize()
fwd_ms = fwd_start.elapsed_time(fwd_end)
```

Track total forward time, total backward time, and total idle time per rank. The bubble is:
```
measured_bubble = idle_time / (fwd_time + bwd_time + idle_time)
theoretical_bubble = (pp_size - 1) / num_microbatches
```

These should be close. If measured > theoretical, there's serialization overhead (blocking send/recv, CUDA synchronization points) eating into utilization.

---

## What comes after

The GPipe → 1F1B progression covers the core abstractions. From here, the path forward:

- **Interleaved 1F1B.** Each rank holds `v` non-contiguous chunks of layers (round-robin assignment). Reduces the bubble by `v×` at the cost of `v×` P2P communication. The model splitting becomes more complex (each rank holds `v` disjoint layer ranges), and the schedule needs to track which virtual stage each forward/backward operates on.

- **Zero-bubble schedules (ZB1P / ZBV).** Split backward into B_input (compute `dL/dX`, send upstream immediately) and B_weight (compute `dL/dW`, defer to fill idle slots). The strategies entry covers the theory. Implementation-wise, this means splitting PyTorch's `backward()` into two explicit `torch.autograd.backward()` calls with different `inputs=` arguments — one for the previous layer's activation, one for the weights.

- **Using PyTorch's `PipelineStage` + schedule IR.** `torch.distributed.pipelining` provides `PipelineStage` (wraps a model chunk and handles P2P), schedule classes (`ScheduleGPipe`, `Schedule1F1B`, `ScheduleInterleaved1F1B`, etc.), and a schedule IR that lets you compose schedules with FSDP/TP. This trades raw control for composability with other parallelisms — the right move once PP works standalone and we want to combine it with FSDP/TP.

---

## Integration with the training loop

The current training loop in [`nanigpt/train.py`](../nanigpt/train.py) is single-GPU. The PP version needs a separate entry point (`train_pp.py`) that:

1. Calls `init_distributed()` to set up the process group
2. Builds only the local stage via `PipelineStage(config, rank, world_size)`
3. Creates the optimizer over only the local stage's parameters
4. Replaces the forward-backward-step block with `gpipe_step()` or `one_f_one_b_step()`
5. Only computes and logs the loss on the last stage (other stages don't have it)
6. Uses `torchrun` for launch

The data loading stays the same on all ranks — each rank loads the same batch and splits it into microbatches. Only the first and last ranks actually consume the input IDs and targets, respectively. Middle ranks ignore the data and only process activations received via P2P.

This is slightly wasteful (middle ranks load data they don't use), but it's the simplest approach and avoids the complexity of rank-dependent data loading. Optimize later if profiling shows data loading is a bottleneck (it won't be — P2P communication and the bubble dominate).
