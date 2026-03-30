"""Training loop with optional distributed support (DDP / FSDP via Ray).

Run:
    # Single GPU:
    uv run python -m nanigpt.train --config small-synthetic

    # DDP on 2 GPUs:
    uv run python -m nanigpt.train --config small-synthetic-ddp

    # FSDP on 2 GPUs:
    uv run python -m nanigpt.train --config small-synthetic-fsdp
"""

import dataclasses
import json
import logging
import math
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler

import wandb
from nanigpt.config import TrainConfig, parse_config
from nanigpt.data.synthetic import SyntheticData
from nanigpt.data.tokenized import TokenizedData
from nanigpt.distributed import (
    apply_parallelism,
    cleanup_distributed,
    init_distributed,
)
from nanigpt.models.dense_transformer import DenseTransformer
from nanigpt.profiling import flop_counter
from nanigpt.profiling.context import init_context, register_step_end, step_context
from nanigpt.profiling.event_types import EventType
from nanigpt.profiling.timer import get_global_metrics, measure


@torch.no_grad()
def evaluate(model, val_loader, model_config, device, dtype, val_steps):
    """Run validation and return metrics."""
    model.eval()
    total_loss = 0.0
    count = 0
    val_iter = iter(val_loader)

    with torch.no_grad():
        for _ in range(val_steps):
            try:
                batch = next(val_iter)
            except StopIteration:
                val_iter = iter(val_loader)
                batch = next(val_iter)

            tokens = batch.to(device, non_blocking=True)
            input_ids = tokens[:, :-1]
            targets = tokens[:, 1:]

            with torch.amp.autocast(device_type="cuda", dtype=dtype):
                output = model(input_ids)
                loss = F.cross_entropy(
                    output.logits.view(-1, model_config.vocab_size),
                    targets.reshape(-1),
                )

            total_loss += loss.item()
            count += 1

    model.train()
    avg_loss = total_loss / count
    return {"val_loss": avg_loss, "val_perplexity": math.exp(avg_loss)}


def train_worker(rank: int, world_size: int, config: TrainConfig) -> None:
    """Run the training loop for a single rank.

    When world_size == 1, this is a plain single-GPU loop (no process group).
    When world_size > 1, initializes distributed, wraps the model, and uses
    DistributedSampler for data sharding.
    """
    distributed = world_size > 1
    is_main = rank == 0

    logging.basicConfig(
        level=logging.INFO,
        format=f"\033[2m%(asctime)s [rank {rank}] %(levelname)s\033[0m %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger("train")

    if distributed:
        init_distributed(rank, world_size, backend=config.parallel.backend)

    torch.manual_seed(config.training.seed)
    init_context()

    # Device is always cuda:0 — Ray sets CUDA_VISIBLE_DEVICES per worker
    device = "cuda:0" if distributed else config.training.device

    gpu_name = flop_counter.detect_gpu()

    if is_main:
        config_dict = dataclasses.asdict(config)
        log.info(f"Config:\n{json.dumps(config_dict, indent=2, default=str)}")
        wandb.init(
            project=config.logging.wandb_project,
            config=config_dict,
        )

    model = DenseTransformer(config.model).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    non_emb_params = model.num_non_embedding_params()
    if is_main:
        log.info(f"Model: {num_params:,} params ({non_emb_params:,} non-embedding)")

    # ---- Apply distributed wrapping ----
    if distributed:
        model = apply_parallelism(model, world_size, config.parallel)

    # ---- Dataset construction ----
    num_train_samples = config.training.num_steps * config.training.batch_size
    val_loader = None

    match config.data:
        case TokenizedData.Config() as data_cfg:
            if is_main:
                log.info(f"Using real data from {data_cfg.data_dir}")
            train_data = data_cfg.build(split="train", num_samples=num_train_samples)
            val_data = data_cfg.build(
                split="val", num_samples=config.eval.val_steps * config.training.batch_size
            )
            val_loader = DataLoader(
                val_data.dataset,
                batch_size=config.training.batch_size,
                shuffle=False,
                num_workers=val_data.num_workers,
                pin_memory=True,
                drop_last=True,
            )
        case SyntheticData.Config() as data_cfg:
            if is_main:
                log.info("Using synthetic data (RandomTokenDataset)")
            train_data = data_cfg.build(
                vocab_size=config.model.vocab_size, num_samples=num_train_samples
            )

    # ---- DataLoader with optional DistributedSampler ----
    train_sampler = None
    if distributed:
        train_sampler = DistributedSampler(
            train_data.dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=config.training.seed,
        )

    loader = DataLoader(
        train_data.dataset,
        batch_size=config.training.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=train_data.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.lr)

    seq_len = config.data.seq_len
    tokens_per_step = config.training.batch_size * seq_len
    step_flops = flop_counter.total_flops(model, config.training.batch_size, seq_len)

    if is_main:
        peak = flop_counter.GPU_PEAK_TFLOPS.get(gpu_name, "?")
        log.info(f"GPU: {gpu_name} | Peak: {peak} TFLOPS (bf16)")
        log.info(
            f"Batch: {config.training.batch_size} x {seq_len} tokens"
            f" | Steps: {config.training.num_steps}"
            f" | Workers: {world_size}"
        )
        log.info(f"Theoretical FLOPs/step: {step_flops / 1e9:.2f} GFLOPs")

    step_width = len(str(config.training.num_steps))

    model.train()
    data_iter = iter(loader)

    profiler = None
    if is_main:
        profiler = config.profiler.build()
        register_step_end(profiler)

    t0 = time.perf_counter()
    dtype = config.training.torch_dtype

    for step in range(1, config.training.num_steps + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(step)

        with step_context(step):
            with measure(EventType.DATA):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(loader)
                    batch = next(data_iter)
                tokens = batch.to(device, non_blocking=True)
                input_ids = tokens[:, :-1]
                targets = tokens[:, 1:]

            with measure(EventType.FORWARD):
                with torch.amp.autocast(device_type="cuda", dtype=dtype):
                    output = model(input_ids)
                    loss = F.cross_entropy(
                        output.logits.view(-1, config.model.vocab_size),
                        targets.reshape(-1),
                    )

            with measure(EventType.BACKWARD):
                optimizer.zero_grad(set_to_none=True)
                loss.backward()

            with measure(EventType.OPTIMIZER):
                optimizer.step()

        timings = get_global_metrics().last_step_ms()
        train_timings = {k: v for k, v in timings.items() if k != EventType.EVAL.value}
        total_ms = sum(train_timings.values())
        tflops = flop_counter.achieved_tflops(step_flops, total_ms)
        utilization = flop_counter.mfu(tflops, gpu_name)
        tok_per_sec = tokens_per_step / (total_ms / 1000.0)

        train_loss = loss.item()
        train_ppl = math.exp(train_loss)

        log_dict = {
            "loss": train_loss,
            "perplexity": train_ppl,
            "ms_per_step": total_ms,
            "tokens_per_sec": tok_per_sec,
            "tflops": tflops,
            "mfu": utilization,
            **{f"time/{k}": v for k, v in train_timings.items()},
        }

        # ---- Validation ----
        if val_loader is not None and step % config.eval.val_interval == 0:
            with measure(EventType.EVAL):
                val_metrics = evaluate(
                    model,
                    val_loader,
                    config.model,
                    device,
                    dtype,
                    config.eval.val_steps,
                )
            log_dict.update(val_metrics)
            if is_main:
                log.info(
                    f"  [eval] val_loss: {val_metrics['val_loss']:.4f}"
                    f" | val_ppl: {val_metrics['val_perplexity']:.4g}"
                )

        if is_main:
            wandb.log(log_dict, step=step)

        if is_main and (step % config.logging.log_interval == 0 or step == 1):
            log.info(
                f"step: {step:>{step_width}}/{config.training.num_steps}"
                f" | loss: {train_loss:.2f}"
                f" | ppl: {train_ppl:.4g} | ms/step: {total_ms:.1f}"
                f" | tok/s: {tok_per_sec:,.0f}"
                f" | TFLOPS: {tflops:.1f} | MFU: {utilization * 100:.1f}%"
            )

    # ---- Final validation ----
    if val_loader is not None:
        if is_main:
            log.info("Running final validation...")
        with measure(EventType.EVAL):
            final_val = evaluate(
                model,
                val_loader,
                config.model,
                device,
                dtype,
                config.eval.val_steps,
            )
        if is_main:
            log.info(
                f"  [final eval] val_loss: {final_val['val_loss']:.4f}"
                f" | val_ppl: {final_val['val_perplexity']:.4g}"
            )

    # ---- End of training summary ----
    if is_main:
        wall_time = time.perf_counter() - t0
        peak_mem = torch.cuda.max_memory_allocated() / 1e9
        all_mean_timings = get_global_metrics().mean_ms()
        mean_timings = {k: v for k, v in all_mean_timings.items() if k != EventType.EVAL}
        mean_total_ms = sum(mean_timings.values())
        mean_tflops = flop_counter.achieved_tflops(step_flops, mean_total_ms)
        mean_mfu = flop_counter.mfu(mean_tflops, gpu_name)
        mean_tok_per_sec = tokens_per_step / (mean_total_ms / 1000.0) if mean_total_ms > 0 else 0.0

        log.info("--- Training complete ---")
        log.info(f"Wall time: {wall_time:.1f}s ({config.training.num_steps} steps)")
        log.info(f"Mean step: {mean_total_ms:.1f} ms ({tokens_per_step:,} tok/step)")
        for name, ms in mean_timings.items():
            pct = 100.0 * ms / mean_total_ms if mean_total_ms > 0 else 0.0
            log.info(f"  {name:>12s}: {ms:7.2f} ms ({pct:5.1f}%)")
        log.info(f"Mean tok/s: {mean_tok_per_sec:,.0f}")
        log.info(f"Mean TFLOPS: {mean_tflops:.1f}")
        log.info(f"Mean MFU: {mean_mfu * 100:.1f}%")
        log.info(f"Peak memory: {peak_mem:.2f} GB")
        if profiler is not None:
            profiler.print_summary()
            profiler.serve_perfetto()

        summary = {
            "wall_time_s": wall_time,
            "mean_ms_per_step": mean_total_ms,
            "mean_tokens_per_sec": mean_tok_per_sec,
            "mean_tflops": mean_tflops,
            "mean_mfu": mean_mfu,
            "peak_memory_gb": peak_mem,
            "num_params": num_params,
            "non_emb_params": non_emb_params,
        }
        if val_loader is not None:
            summary.update(final_val)

        wandb.summary.update(summary)
        wandb.finish()

    if distributed:
        cleanup_distributed()


def main():
    config = parse_config()
    config.validate()

    if config.parallel.num_workers == 1:
        # Single-GPU: run directly
        train_worker(0, 1, config)
    else:
        # Multi-GPU: launch via Ray
        from nanigpt.launcher import launch

        launch(config, config.parallel.num_workers)


if __name__ == "__main__":
    main()
