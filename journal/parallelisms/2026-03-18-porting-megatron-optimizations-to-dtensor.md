# Porting Megatron Optimizations to a DTensor-Based Training Stack

*2026-03-18*

The [parallelism strategies entry](2026-02-24-parallelism-strategies.md) established two contrasting approaches: Megatron encodes parallelism *into* the model via explicit `autograd.Function` subclasses, while torchtitan applies parallelism *onto* the model via DTensor placements. The conclusion was to follow torchtitan's style for composability and study Megatron for understanding.

This entry is about the next question: **once you've committed to DTensor as the default, how do you surgically bring in Megatron's performance optimizations?** DTensor gives you composability and clean model code. Megatron gives you hand-tuned communication-computation overlap that DTensor's automatic redistribution doesn't replicate. These aren't mutually exclusive — DTensor has well-defined escape hatches at every level of the stack.

Repos studied:
- **torchtitan** @ [`c7378f6`](https://github.com/pytorch/torchtitan/tree/c7378f666c6ca0465287505e786346c87a16d996)
- **Megatron-LM** @ [`32efeffd`](https://github.com/NVIDIA/Megatron-LM/tree/32efeffd2d3cca48f1bfa575e1f2d0f0feac3d0f)

---

## Part 1: What Megatron Has That DTensor Doesn't

Not everything in Megatron is a "Megatron optimization." Many of its patterns — column/row parallel, the f/g conjugate pair, sequence parallelism — are already implemented in DTensor via `ColwiseParallel`, `RowwiseParallel`, and `SequenceParallel`. The collectives are identical. The difference is whether you write `ColumnParallelLinear` (Megatron) or `parallelize_module(model, mesh, {"attn.wq": ColwiseParallel()})` (torchtitan).

The genuine Megatron advantages — things DTensor doesn't give you automatically — fall into five categories:

### 1. Async backward overlap (`LinearWithGradAccumulationAndAsyncCommunication`)

This is the crown jewel. A single `autograd.Function` that fuses the linear layer's forward and backward with overlapped communication:

[`megatron/core/tensor_parallel/layers.py#L495-L628`](https://github.com/NVIDIA/Megatron-LM/blob/32efeffd2d3cca48f1bfa575e1f2d0f0feac3d0f/megatron/core/tensor_parallel/layers.py#L495-L628):

```python
@staticmethod
def backward(ctx, grad_output):
    input, weight = ctx.saved_tensors

    if ctx.sequence_parallel:
        # 1. Launch async all-gather of input (needed for wgrad, not dgrad)
        handle = dist_all_gather_func(all_gather_buffer, input, group=tp_group, async_op=True)

    # 2. Compute dgrad while all-gather flies — dgrad only needs weight, not full input
    grad_input = grad_output.matmul(weight)

    if ctx.sequence_parallel:
        handle.wait()  # all-gather done

    if ctx.allreduce_dgrad:
        # 3. Launch async all-reduce of dgrad
        handle = torch.distributed.all_reduce(grad_input, group=tp_group, async_op=True)
    elif ctx.sequence_parallel:
        # 3. Launch async reduce-scatter of dgrad
        handle = dist_reduce_scatter_func(sub_grad_input, grad_input, group=tp_group, async_op=True)

    # 4. Compute wgrad while dgrad reduction flies — wgrad needs full input but not reduced dgrad
    grad_weight = grad_output.t().matmul(total_input)

    handle.wait()  # reduction done
    return grad_input, grad_weight, ...
```

The key insight: **dgrad and wgrad have different data dependencies**. dgrad needs `weight` (already local). wgrad needs `total_input` (needs all-gather). dgrad reduction is needed by the *upstream* layer, not by wgrad. So you can overlap: all-gather → dgrad → async reduce of dgrad → wgrad (while reduce flies).

This only works because `CUDA_DEVICE_MAX_CONNECTIONS=1` forces the GPU to execute kernels in launch order. Without it, CUDA can reorder, breaking the overlap.

[`megatron/core/tensor_parallel/layers.py#L729-L745`](https://github.com/NVIDIA/Megatron-LM/blob/32efeffd2d3cca48f1bfa575e1f2d0f0feac3d0f/megatron/core/tensor_parallel/layers.py#L729-L745):

```python
if os.environ.get("CUDA_DEVICE_MAX_CONNECTIONS") != "1":
    if sequence_parallel:
        warnings.warn(
            "When using sequence parallelism it is recommended to set the "
            "environment variable CUDA_DEVICE_MAX_CONNECTIONS to 1 for "
            "maximum speedup"
        )
```

DTensor's automatic redistribution doesn't do this — each op triggers synchronous redistribution when placement mismatches are detected. There's no mechanism to say "start this reduce-scatter, but don't wait — I'll compute wgrad in the meantime."

### 2. Vocab-parallel cross-entropy

Megatron computes cross-entropy loss without ever materializing the full `[batch, seq, vocab]` logits tensor across TP ranks. Each rank holds `vocab_size / tp` columns of logits, and the loss is computed with three targeted all-reduces on scalars:

[`megatron/core/tensor_parallel/cross_entropy.py#L123-L189`](https://github.com/NVIDIA/Megatron-LM/blob/32efeffd2d3cca48f1bfa575e1f2d0f0feac3d0f/megatron/core/tensor_parallel/cross_entropy.py#L123-L189):

```python
@staticmethod
def forward(ctx, vocab_parallel_logits, target, label_smoothing=0.0):
    # 1. Local max per rank, then all-reduce MAX for numerical stability
    logits_max = torch.max(vocab_parallel_logits, dim=-1)[0]
    torch.distributed.all_reduce(logits_max, op=ReduceOp.MAX, group=tp_group)

    # 2. Subtract max, compute local exp and sum_exp
    vocab_parallel_logits = vocab_parallel_logits - logits_max.unsqueeze(-1)
    exp_logits = vocab_parallel_logits.exp()
    sum_exp_logits = exp_logits.sum(dim=-1)

    # 3. Extract predicted logit (only the rank that owns the target class contributes)
    target_mask = (target < vocab_start_index) | (target >= vocab_end_index)
    masked_target = target.clone() - vocab_start_index
    masked_target[target_mask] = 0
    predicted_logits = torch.gather(vocab_parallel_logits, -1, masked_target.unsqueeze(-1)).squeeze(-1)
    predicted_logits[target_mask] = 0.0

    # 4. All-reduce SUM both scalars
    torch.distributed.all_reduce(predicted_logits, op=ReduceOp.SUM, group=tp_group)
    torch.distributed.all_reduce(sum_exp_logits, op=ReduceOp.SUM, group=tp_group)

    # 5. Loss = log(sum_exp) - predicted_logit
    loss = torch.log(sum_exp_logits) - predicted_logits
    return loss
```

The fused variant goes further — it concatenates `predicted_logits` and `sum_exp_logits` before the all-reduce, cutting 2 all-reduces to 1:

[`megatron/core/fusions/fused_cross_entropy.py#L42-L44`](https://github.com/NVIDIA/Megatron-LM/blob/32efeffd2d3cca48f1bfa575e1f2d0f0feac3d0f/megatron/core/fusions/fused_cross_entropy.py#L42-L44):

```python
predicted_logits_sum_exp_logits = torch.cat((predicted_logits, sum_exp_logits))
```

[`megatron/core/fusions/fused_cross_entropy.py#L109-L114`](https://github.com/NVIDIA/Megatron-LM/blob/32efeffd2d3cca48f1bfa575e1f2d0f0feac3d0f/megatron/core/fusions/fused_cross_entropy.py#L109-L114):

```python
torch.distributed.all_reduce(
    predicted_logits_sum_exp_logits, op=torch.distributed.ReduceOp.SUM, group=tp_group
)
```

With DTensor, if you have logits with placement `Shard(-1)` and call `F.cross_entropy(logits, target)`, DTensor would need to all-gather the full logits before computing the loss — `O(batch * seq * vocab)` bytes communicated. Megatron communicates `O(batch * seq)` bytes (the scalar predicted_logit and sum_exp per position). For a 128K vocabulary, that's a ~128K× reduction in communication volume.

### 3. Fused gradient accumulation

Megatron uses a custom CUDA kernel to fuse the weight gradient matmul with accumulation into an fp32 buffer, eliminating a separate memory-bandwidth-bound `grad += new_grad` kernel:

[`megatron/core/tensor_parallel/layers.py#L563-L611`](https://github.com/NVIDIA/Megatron-LM/blob/32efeffd2d3cca48f1bfa575e1f2d0f0feac3d0f/megatron/core/tensor_parallel/layers.py#L563-L611):

```python
if ctx.gradient_accumulation_fusion:
    if wgrad_compute:
        if ctx.sequence_parallel:
            fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp32(
                total_input, grad_output, weight.main_grad
            )
        else:
            fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp32(
                input, grad_output, weight.main_grad
            )
        grad_weight = None  # wgrad accumulated directly into main_grad
```

Standard PyTorch: `grad_weight = grad_output.T @ input` → bf16 result → `main_grad += grad_weight.float()` — two kernels, two memory round-trips. Megatron: one fused kernel that accumulates in fp32 directly.

### 4. Bucketed DDP with async overlap

Megatron's DDP overlaps gradient reduction with backward computation using bucketed backward hooks:

[`megatron/core/distributed/distributed_data_parallel.py#L420-L448`](https://github.com/NVIDIA/Megatron-LM/blob/32efeffd2d3cca48f1bfa575e1f2d0f0feac3d0f/megatron/core/distributed/distributed_data_parallel.py#L420-L448):

```python
def _make_backward_post_hook(self, param):
    def hook(*unused):
        if param.grad is not None and not param.grad_added_to_main_grad:
            param.main_grad.add_(param.grad.data)
        param.grad = None
        if self.ddp_config.overlap_grad_reduce:
            self.param_to_bucket_group[param].register_grad_ready(param, self.force_all_reduce)
    return hook
```

And overlaps parameter all-gather with forward computation:

[`megatron/core/distributed/distributed_data_parallel.py#L384-L418`](https://github.com/NVIDIA/Megatron-LM/blob/32efeffd2d3cca48f1bfa575e1f2d0f0feac3d0f/megatron/core/distributed/distributed_data_parallel.py#L384-L418):

```python
def _make_forward_pre_hook(self):
    def hook(module, *unused):
        for param in module.parameters(recurse=False):
            if param in self.param_to_bucket_group:
                self.param_to_bucket_group[param].finish_param_sync(
                    skip_next_bucket_dispatch=skip_next_bucket_dispatch
                )
    return hook
```

Note: PyTorch's native DDP and FSDP2 (`fully_shard()`) already do bucketed gradient reduction with backward overlap. This is less of a "Megatron advantage" and more of a "both frameworks implement this." The Megatron version has finer control over bucket sizing and the parameter all-gather overlap during forward, but for most cases `fully_shard()` is at parity.

### 5. Pipeline activation deallocation

After sending an activation tensor to the next PP stage, Megatron replaces its `.data` with a scalar to free memory while keeping the autograd graph intact:

[`megatron/core/pipeline_parallel/schedules.py#L154-L166`](https://github.com/NVIDIA/Megatron-LM/blob/32efeffd2d3cca48f1bfa575e1f2d0f0feac3d0f/megatron/core/pipeline_parallel/schedules.py#L154-L166):

```python
def deallocate_output_tensor(out, deallocate_pipeline_outputs=False):
    if (out is None) or (not deallocate_pipeline_outputs):
        return
    assert isinstance(out, torch.Tensor), ...
    assert out._base is None, "output should not be a view"
    out.data = torch.empty((1,), device=out.device, dtype=out.dtype)
```

This works because autograd only needs the `.grad_fn` chain, not the tensor data, until backward actually runs on that op. When backward arrives, the activation is re-received from the upstream rank.

---

## Part 2: The Override Hierarchy

DTensor has five levels of override, from lightest (declarative placement control) to heaviest (bypass DTensor entirely). Each level gives you more control at the cost of more manual work. Torchtitan uses all five in production.

### Level 1: Placement control at module boundaries

**What it does:** Controls how DTensor redistributes tensors at the boundaries between modules — what placement the input should have, what placement the output should have.

**Mechanism:** `PrepareModuleInput`, `PrepareModuleOutput`, `PrepareModuleInputOutput` — these are forward hooks that call `redistribute()` on DTensors flowing between modules.

**When to use:** When DTensor's per-op sharding strategy would choose the wrong redistribution. For example, you want a reduce-scatter (producing `Shard(1)`) instead of an all-reduce (producing `Replicate()`).

[`torchtitan/models/llama4/parallelize.py#L544-L551`](https://github.com/pytorch/torchtitan/blob/c7378f666c6ca0465287505e786346c87a16d996/torchtitan/models/llama4/parallelize.py#L544-L551):

```python
"moe": PrepareModuleInputOutput(
    input_layouts=(Shard(1),),
    desired_input_layouts=(Replicate(),),       # all-gather before MoE
    use_local_input=False,
    output_layouts=(Partial(),),
    desired_output_layouts=(Shard(1),),          # reduce-scatter after MoE
),
```

This single declaration says: "the MoE module receives sequence-sharded input — all-gather it to replicated before entering. The MoE produces partial sums — reduce-scatter them back to sequence-sharded after exiting." The collectives are chosen by placement, not by explicit collective calls.

**What you can't do at this level:** Control *how* the collective runs (async vs sync), fuse multiple collectives, or change what happens inside the module's forward/backward.

### Level 2: Exit and re-enter DTensor land

**What it does:** Converts DTensors to plain `torch.Tensor`s at a boundary, does whatever you want with raw tensors, then converts back.

**Mechanism:** `DTensor.to_local(grad_placements=...)` to exit, `DTensor.from_local(tensor, mesh, placements)` to re-enter. The `grad_placements` parameter is critical — it tells DTensor what placement the gradient will have when backward flows back through this boundary, so DTensor can insert the right collective in the backward pass.

**When to use:** When the operation you need can't be expressed in DTensor's placement algebra. Expert dispatch (all-to-all with dynamic shapes per expert) is the canonical example.

[`torchtitan/models/common/moe/moe.py#L473-L480`](https://github.com/pytorch/torchtitan/blob/c7378f666c6ca0465287505e786346c87a16d996/torchtitan/models/common/moe/moe.py#L473-L480):

```python
if isinstance(x, DTensor):
    x = x.to_local(grad_placements=(Partial(),))
```

The `grad_placements=(Partial(),)` means: "in backward, the gradient arriving here will be a partial sum (each rank holds part of the gradient). DTensor should insert a reduction (reduce-scatter to `Shard(1)`) to get the full gradient." This one line controls the entire backward communication pattern at the MoE boundary.

Similarly, expert weights are extracted as plain tensors for `grouped_mm`:

[`torchtitan/models/common/moe/moe.py#L99-L106`](https://github.com/pytorch/torchtitan/blob/c7378f666c6ca0465287505e786346c87a16d996/torchtitan/models/common/moe/moe.py#L99-L106):

```python
if isinstance(self.w1, DTensor):
    w1 = self.w1.to_local()
    w2 = self.w2.to_local()
    w3 = self.w3.to_local()
```

And raw functional collectives are used for the expert dispatch:

[`torchtitan/distributed/expert_parallel.py#L123-L128`](https://github.com/pytorch/torchtitan/blob/c7378f666c6ca0465287505e786346c87a16d996/torchtitan/distributed/expert_parallel.py#L123-L128):

```python
routed_input = all_to_all_single_autograd(
    routed_input,
    self.output_splits,
    self.input_splits,
    device_mesh.get_group(),
)
```

**What you can't do at this level:** Register new ops that DTensor's sharding propagator can reason about, or get compiler optimizations on the raw-tensor section.

### Level 3: Custom `ParallelStyle` with `distribute_module()`

**What it does:** Defines a new parallelism pattern as a reusable class with custom partition, input, and output functions. This is how you create a new "style" of parallelism (like `ColwiseParallel`, but for your specific use case).

**Mechanism:** Subclass `ParallelStyle`, implement `_apply()`, call `distribute_module()` with custom `partition_fn` (how to shard weights), `input_fn` (forward pre-hook), and `output_fn` (forward post-hook).

**When to use:** When you have a recurring pattern that doesn't fit the standard `ColwiseParallel`/`RowwiseParallel` taxonomy, especially when you need to control backward gradient placement.

[`torchtitan/distributed/tensor_parallel.py#L109-L188`](https://github.com/pytorch/torchtitan/blob/c7378f666c6ca0465287505e786346c87a16d996/torchtitan/distributed/tensor_parallel.py#L109-L188):

```python
class ColwiseParallelWithGradPlacement(ColwiseParallel):
    """ColwiseParallel but with explicit control over input gradient placement in backward."""

    def __init__(self, *, local_input_grad_placements: tuple[Placement, ...] | None = None, **kwargs):
        super().__init__(**kwargs)
        self.local_input_grad_placements = local_input_grad_placements

    @staticmethod
    def _prepare_input_fn(input_layout, desired_input_layout, grad_placements, mod, inputs, device_mesh):
        # Same redistribution as ColwiseParallel, but passes grad_placements
        # to DTensor.from_local() so backward uses the specified placement
        # instead of DTensor's default (which would be an all-reduce)
        ...
        input_tensor = DTensor.from_local(
            input_tensor, device_mesh, (input_layout,), run_check=False,
            grad_placements=(grad_placements,) if grad_placements else None,
        )
```

The `local_input_grad_placements=(Partial(),)` usage defers the gradient reduction from inside the linear layer to the MoE output boundary — avoiding a redundant all-reduce.

Torchtitan also defines `NoParallel` (replicates params as DTensors without sharding) for modules like the router gate that need to live on the same mesh as TP-sharded params:

[`torchtitan/distributed/tensor_parallel.py#L24-L106`](https://github.com/pytorch/torchtitan/blob/c7378f666c6ca0465287505e786346c87a16d996/torchtitan/distributed/tensor_parallel.py#L24-L106):

```python
class NoParallel(ParallelStyle):
    def _apply(self, module, device_mesh):
        return distribute_module(
            module, device_mesh,
            None,  # no custom partitioning — just replicate
            partial(self._prepare_input_fn, self.input_layout, self.desired_input_layout),
            partial(self._prepare_output_fn, self.output_layout, self.local_output_grad_placements),
        )
```

**This is the right level for most Megatron optimizations that involve controlling *what* communication happens.** Writing a `FusedTPLinear` parallel style that applies Megatron's communication pattern but through DTensor's module-level hooks is the cleanest integration path.

### Level 4: Custom ops via `torch.library`

**What it does:** Registers a new operator in PyTorch's op registry with custom forward, backward, and (optionally) DTensor sharding rules. The op is visible to `torch.compile`, Selective Activation Checkpointing, and DTensor's sharding propagator.

**Mechanism:** `torch.library.Library` to define the op schema, `@torch.library.impl` for the implementation, `torch.library.register_autograd` for the backward, and optionally `@register_sharding` to tell DTensor how to shard it.

**When to use:** When you have a fused operation that should be a first-class citizen in the computation graph — visible to the compiler, composable with activation checkpointing, and potentially shardable by DTensor.

[`torchtitan/distributed/deepep/deepep.py#L60-L71`](https://github.com/pytorch/torchtitan/blob/c7378f666c6ca0465287505e786346c87a16d996/torchtitan/distributed/deepep/deepep.py#L60-L71):

```python
_lib = torch.library.Library("deepep", "DEF")
_lib.define("dispatch(Tensor x, ...) -> (Tensor, Tensor, Tensor, Tensor, Tensor)")
_lib.define("combine(Tensor x, Tensor handle_id) -> Tensor")
```

[`torchtitan/distributed/deepep/deepep.py#L237-L242`](https://github.com/pytorch/torchtitan/blob/c7378f666c6ca0465287505e786346c87a16d996/torchtitan/distributed/deepep/deepep.py#L237-L242):

```python
torch.library.register_autograd(
    "deepep::dispatch", _dispatch_backward, setup_context=_dispatch_setup_context
)
torch.library.register_autograd(
    "deepep::combine", _combine_backward, setup_context=_combine_setup_context
)
```

The registered ops appear as `torch.ops.deepep.dispatch.default` and can be added to SAC save lists:

```python
_op_sac_save_list = [
    torch.ops.deepep.dispatch.default,
    torch.ops.deepep.combine.default,
]
```

**This is the right level for Megatron's vocab-parallel cross-entropy.** Define `nanigpt::vocab_parallel_cross_entropy` as a custom op with the targeted-all-reduce forward and the local-softmax-gradient backward. Register a sharding strategy so DTensor knows how to handle it when logits are `Shard(-1)`.

To tell DTensor how to shard a custom op, use the public API:

```python
from torch.distributed.tensor.experimental import register_sharding

@register_sharding(torch.ops.nanigpt.vocab_parallel_cross_entropy.default)
def vpc_sharding(vocab_parallel_logits, target):
    return [
        # output is Replicate (loss is a scalar per position, same on all ranks)
        # input logits are Shard(-1), target is Replicate
        ([Replicate()], [Shard(-1), Replicate()]),
    ]
```

### Level 5: Custom `autograd.Function` bypassing DTensor

**What it does:** Completely bypasses DTensor's dispatch for a specific layer. You write raw `torch.autograd.Function` with explicit collective calls, operating on plain tensors.

**Mechanism:** Write an `autograd.Function` with `forward` and `backward` that call `dist.all_gather`, `dist.reduce_scatter`, etc. directly. Plug it into a DTensor-parallelized model by either (a) replacing a module's forward, or (b) using it inside a custom `ParallelStyle`.

**When to use:** When you need precise control over the ordering of async communication and computation in backward — the `CUDA_DEVICE_MAX_CONNECTIONS=1` overlap pattern that is Megatron's core performance advantage.

This is what it looks like to port Megatron's async backward overlap into a DTensor-based model:

```python
class AsyncTPLinear(torch.autograd.Function):
    """Megatron-style async comm-compute overlap for a TP linear layer.

    Plug this into a DTensor-parallelized model at specific hot-path layers
    where profiling shows DTensor's synchronous redistribution is a bottleneck.
    """
    @staticmethod
    def forward(ctx, input_local, weight_local, tp_group, sequence_parallel):
        ctx.save_for_backward(input_local, weight_local)
        ctx.tp_group = tp_group
        ctx.sequence_parallel = sequence_parallel

        if sequence_parallel:
            # All-gather the sequence-sharded input
            world_size = dist.get_world_size(tp_group)
            gathered = torch.empty(
                [input_local.shape[0] * world_size, *input_local.shape[1:]],
                dtype=input_local.dtype, device=input_local.device,
            )
            dist.all_gather_into_tensor(gathered, input_local, group=tp_group)
            total_input = gathered
        else:
            total_input = input_local

        return torch.matmul(total_input, weight_local.t())

    @staticmethod
    def backward(ctx, grad_output):
        input_local, weight_local = ctx.saved_tensors
        tp_group = ctx.tp_group

        if ctx.sequence_parallel:
            # Step 1: Async all-gather of input (needed for wgrad)
            world_size = dist.get_world_size(tp_group)
            gathered = torch.empty(
                [input_local.shape[0] * world_size, *input_local.shape[1:]],
                dtype=input_local.dtype, device=input_local.device,
            )
            ag_handle = dist.all_gather_into_tensor(gathered, input_local, group=tp_group, async_op=True)

        # Step 2: Compute dgrad (only needs weight, not gathered input)
        grad_input = grad_output.matmul(weight_local)

        if ctx.sequence_parallel:
            ag_handle.wait()
            total_input = gathered

            # Step 3: Async reduce-scatter of dgrad
            rs_output = torch.empty_like(input_local)
            rs_handle = dist.reduce_scatter_tensor(rs_output, grad_input, group=tp_group, async_op=True)
        else:
            total_input = input_local

        # Step 4: Compute wgrad while reduce-scatter flies
        grad_weight = grad_output.reshape(-1, grad_output.shape[-1]).t().matmul(
            total_input.reshape(-1, total_input.shape[-1])
        )

        if ctx.sequence_parallel:
            rs_handle.wait()
            grad_input = rs_output

        return grad_input, grad_weight, None, None
```

To plug this into a DTensor-parallelized model:

```python
# Option A: Replace at module level after parallelize_module()
parallelize_module(model, tp_mesh, plan)

for block in model.blocks:
    original_w1 = block.ffn.w1  # DTensor-parallelized nn.Linear
    # Extract local shards and process group
    weight_local = original_w1.weight.to_local()
    tp_group = tp_mesh.get_group()

    # Replace forward to use async overlap
    def make_fused_forward(w_local, group):
        def fused_forward(x):
            x_local = x.to_local() if isinstance(x, DTensor) else x
            out = AsyncTPLinear.apply(x_local, w_local, group, True)
            return DTensor.from_local(out, tp_mesh, (Shard(-1),))
        return fused_forward

    block.ffn.w1.forward = make_fused_forward(weight_local, tp_group)
```

```python
# Option B: Custom ParallelStyle (cleaner, reusable)
class AsyncTPColwiseParallel(ParallelStyle):
    def _apply(self, module, device_mesh):
        # Shard weight column-wise (same as ColwiseParallel)
        distribute_tensor(module.weight, device_mesh, [Shard(0)])
        if module.bias is not None:
            distribute_tensor(module.bias, device_mesh, [Shard(0)])

        # Install hooks that use AsyncTPLinear instead of DTensor dispatch
        def input_fn(mod, inputs, mesh):
            x = inputs[0]
            x_local = x.to_local() if isinstance(x, DTensor) else x
            return (x_local,)

        def output_fn(mod, inputs, output, mesh):
            return DTensor.from_local(output, mesh, (Shard(-1),))

        distribute_module(module, device_mesh, input_fn=input_fn, output_fn=output_fn)
```

Option B is better because it composes with the rest of `parallelize_module()`:

```python
layer_plan = {
    "attention.wq": ColwiseParallel(),        # standard DTensor
    "attention.wk": ColwiseParallel(),        # standard DTensor
    "attention.wv": ColwiseParallel(),        # standard DTensor
    "attention.wo": RowwiseParallel(),        # standard DTensor
    "ffn.w1": AsyncTPColwiseParallel(),       # Megatron-style async overlap
    "ffn.w2": RowwiseParallel(),              # standard DTensor
}
```

You get DTensor composability everywhere except the one layer where you've profiled a bottleneck and surgically applied the Megatron optimization.

---

## Part 3: What `torch.compile` Already Handles

Before manually porting Megatron optimizations, check whether `torch.compile` already finds the overlap. Torchtitan enables async TP via a single config flag:

[`torchtitan/distributed/tensor_parallel.py#L191-L205`](https://github.com/pytorch/torchtitan/blob/c7378f666c6ca0465287505e786346c87a16d996/torchtitan/distributed/tensor_parallel.py#L191-L205):

```python
def maybe_enable_async_tp(parallelism, compile_config, tp_mesh):
    if not parallelism.enable_async_tensor_parallel:
        return
    if not (compile_config.enable and "model" in compile_config.components):
        raise RuntimeError("Async TP requires 'model' in --compile.components...")
    torch._inductor.config._micro_pipeline_tp = True
```

This tells the Inductor compiler to decompose TP collectives (all-gather, reduce-scatter) into smaller chunks and overlap them with compute. The compiler discovers overlap opportunities that are equivalent to — and sometimes better than — Megatron's manual approach.

Torchtitan benchmarks show 9-16% speedup from async TP (see Part 4 of the [parallelism strategies entry](2026-02-24-parallelism-strategies.md)). This is in the same range as Megatron's manual overlap.

**The compiler can find overlaps humans miss** because it operates on the entire fused computation graph, not on individual layers. But it also has limitations:

- It requires `torch.compile` (not always available during debugging)
- It can't discover algorithmic changes (vocab-parallel cross-entropy)
- It can't do cross-layer optimization (e.g., starting layer N's all-gather while layer N-1's wgrad is still running)
- `_micro_pipeline_tp` is an experimental flag that may change

The practical workflow: **compile first, profile, and only write custom `autograd.Function`s for gaps the compiler misses.**

---

## Part 4: Decision Framework

Given the five override levels and `torch.compile`'s capabilities, here's a decision tree for each Megatron optimization:

```
For each Megatron optimization you want to port:

1. Does torch.compile already handle it?
   └─ Yes → Use torch.compile. Done.
   └─ No → Continue.

2. Is it an algorithmic change (different math, not just different scheduling)?
   └─ Yes → Level 4: Register as a custom op via torch.library.
            Example: vocab-parallel cross-entropy.
   └─ No → Continue.

3. Can it be expressed as a placement/redistribution choice?
   └─ Yes → Level 1: Use PrepareModuleInput/Output.
            Example: reduce-scatter instead of all-reduce at MoE boundary.
   └─ No → Continue.

4. Does it need raw tensors but only at module boundaries?
   └─ Yes → Level 2: to_local() / from_local() with grad_placements.
            Example: expert dispatch with dynamic shapes.
   └─ No → Continue.

5. Is it a reusable pattern across multiple layers?
   └─ Yes → Level 3: Custom ParallelStyle.
            Example: ColwiseParallelWithGradPlacement for deferred reduction.
   └─ No → Level 5: Custom autograd.Function for the specific hot layer.
            Example: Async backward overlap on FFN w1.
```

Applied to the five Megatron optimizations from Part 1:

| Optimization | Level | Rationale |
|---|---|---|
| Async backward overlap | Try `torch.compile` first, Level 5 for gaps | Compiler handles most cases; custom autograd for cross-layer overlap |
| Vocab-parallel cross-entropy | Level 4 (custom op) | Algorithmic — fundamentally different math, compiler can't discover it |
| Fused gradient accumulation | Level 5 (custom autograd) | Requires custom CUDA kernel, below the op level |
| Bucketed DDP overlap | Already in `fully_shard()` | Not needed — FSDP2 does this |
| Pipeline activation deallocation | Level 5 | PP-specific, applies to the schedule implementation, not DTensor |

---

## Part 5: Implications for naniGPT

### Build order

1. **Start with pure DTensor.** `parallelize_module()` + `fully_shard()` + `DeviceMesh`. Get multi-GPU training running. Measure baseline MFU.

2. **Add `torch.compile`.** Enable `_micro_pipeline_tp` for async TP. Measure the delta. This is free performance.

3. **Profile the gaps.** Compare measured MFU against the cost model's projections. The gap tells you where communication is not overlapped.

4. **Port vocab-parallel cross-entropy first** (Level 4). This is the highest-value optimization because it's algorithmic — the compiler can never discover it, and the communication reduction is dramatic.

5. **Profile again.** If backward overlap is the remaining bottleneck, write a custom `AsyncTPLinear` (Level 5) for the FFN layers only.

### What not to port

- Megatron's `DistributedOptimizer` — `fully_shard()` already does this.
- Megatron's `RankGenerator` — `DeviceMesh` is cleaner.
- Megatron's `ColumnParallelLinear` / `RowParallelLinear` as model layers — DTensor's `ColwiseParallel` / `RowwiseParallel` does the same thing declaratively.
- Megatron's explicit f/g `autograd.Function` pairs for TP — DTensor handles this via placement-based redistribution.

### The research question

The interesting output is the measured delta at each step: pure DTensor → +compile → +async TP → +vocab-parallel CE → +custom backward overlap. Each step has diminishing returns. Quantifying *where* the returns flatten out — and how that varies with model size, TP degree, and hardware — is a finding that doesn't exist in the literature.
