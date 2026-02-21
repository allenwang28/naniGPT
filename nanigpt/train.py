"""Single-GPU training loop with profiling.

Run: uv run python -m nanigpt.train
"""

import logging
import math
import time
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import wandb
from nanigpt.data.synthetic import RandomTokenDataset
from nanigpt.data.tokenized import TokenizedDataset
from nanigpt.models.dense_transformer import PRESET_CONFIGS, DenseTransformer
from nanigpt.profiling import flop_counter
from nanigpt.profiling.context import init_context, register_step_end, step_context
from nanigpt.profiling.event_types import EventType
from nanigpt.profiling.timer import get_global_metrics, measure
from nanigpt.profiling.torch_profiler import ProfilerConfig, TorchProfiler

# ---- Config ----
MODEL_PRESET = "small"
BATCH_SIZE = 32
SEQ_LEN = 256
LR = 3e-4
NUM_STEPS = 200
LOG_INTERVAL = 10
DEVICE = "cuda"
DTYPE = torch.bfloat16
NUM_WORKERS = 2
DATA_DIR = Path("data") / "fineweb-1M"  # set to "" for synthetic data
VAL_INTERVAL = 50  # evaluate every N steps
VAL_STEPS = 20  # number of batches to average for val loss
PROFILER_CONFIG = ProfilerConfig(
    enabled=True,
    start_step=10,
    end_step=12,
    warmup_steps=1,
    top_n=15,
    export_trace=True,
)
# -----------------------------------


@torch.no_grad()
def evaluate(model, val_loader, config, device, dtype, val_steps):
    """Run validation and return metrics."""
    model.eval()
    total_loss = 0.0
    count = 0
    val_iter = iter(val_loader)

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
                output.logits.view(-1, config.vocab_size),
                targets.reshape(-1),
            )

        total_loss += loss.item()
        count += 1

    model.train()
    avg_loss = total_loss / count
    return {"val_loss": avg_loss, "val_perplexity": math.exp(avg_loss)}


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="\033[2m%(asctime)s %(levelname)s\033[0m %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger("train")

    torch.manual_seed(42)
    init_context()

    config = PRESET_CONFIGS[MODEL_PRESET]
    config.max_seq_len = SEQ_LEN

    gpu_name = flop_counter.detect_gpu()

    wandb.init(
        project="nanigpt",
        config={
            "model_preset": MODEL_PRESET,
            "model": asdict(config),
            "batch_size": BATCH_SIZE,
            "seq_len": SEQ_LEN,
            "lr": LR,
            "num_steps": NUM_STEPS,
            "dtype": str(DTYPE),
            "gpu": gpu_name,
            "data_dir": str(DATA_DIR),
            "val_interval": VAL_INTERVAL,
            "val_steps": VAL_STEPS,
            "profiler": TorchProfiler(PROFILER_CONFIG).config_dict(),
        },
    )

    model = DenseTransformer(config).to(DEVICE)
    num_params = sum(p.numel() for p in model.parameters())
    non_emb_params = model.num_non_embedding_params()
    log.info(f"Model: {MODEL_PRESET} | {num_params:,} params ({non_emb_params:,} non-embedding)")

    # ---- Dataset construction ----
    val_loader = None
    if DATA_DIR:
        log.info(f"Using real data from {DATA_DIR}")
        dataset = TokenizedDataset(str(DATA_DIR), "train", SEQ_LEN, NUM_STEPS * BATCH_SIZE)
        val_dataset = TokenizedDataset(str(DATA_DIR), "val", SEQ_LEN, VAL_STEPS * BATCH_SIZE)
        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            drop_last=True,
        )
    else:
        log.info("Using synthetic data (RandomTokenDataset)")
        dataset = RandomTokenDataset(
            vocab_size=config.vocab_size,
            seq_len=SEQ_LEN,
            num_samples=NUM_STEPS * BATCH_SIZE,
            distribution="zipf",
        )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    step_flops = flop_counter.total_flops(model, BATCH_SIZE, SEQ_LEN)

    peak = flop_counter.GPU_PEAK_TFLOPS.get(gpu_name, "?")
    log.info(f"GPU: {gpu_name} | Peak: {peak} TFLOPS (bf16)")
    log.info(f"Batch: {BATCH_SIZE} x {SEQ_LEN} tokens | Steps: {NUM_STEPS}")
    log.info(f"Theoretical FLOPs/step: {step_flops / 1e9:.2f} GFLOPs")

    step_width = len(str(NUM_STEPS))

    model.train()
    data_iter = iter(loader)
    profiler = TorchProfiler(PROFILER_CONFIG)
    register_step_end(profiler)
    t0 = time.perf_counter()

    for step in range(1, NUM_STEPS + 1):
        with step_context(step):
            with measure(EventType.DATA):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(loader)
                    batch = next(data_iter)
                tokens = batch.to(DEVICE, non_blocking=True)
                input_ids = tokens[:, :-1]
                targets = tokens[:, 1:]

            with measure(EventType.FORWARD):
                with torch.amp.autocast(device_type="cuda", dtype=DTYPE):
                    output = model(input_ids)
                    loss = F.cross_entropy(
                        output.logits.view(-1, config.vocab_size),
                        targets.reshape(-1),
                    )

            with measure(EventType.BACKWARD):
                optimizer.zero_grad(set_to_none=True)
                loss.backward()

            with measure(EventType.OPTIMIZER):
                optimizer.step()

        timings = get_global_metrics().last_step_ms()
        total_ms = sum(timings.values())
        tflops = flop_counter.achieved_tflops(step_flops, total_ms)
        utilization = flop_counter.mfu(tflops, gpu_name)

        train_loss = loss.item()
        train_ppl = math.exp(train_loss)

        log_dict = {
            "loss": train_loss,
            "perplexity": train_ppl,
            "ms_per_step": total_ms,
            "tflops": tflops,
            "mfu": utilization,
            **{f"time/{k}": v for k, v in timings.items()},
        }

        # ---- Validation ----
        if val_loader is not None and step % VAL_INTERVAL == 0:
            with measure(EventType.EVAL):
                val_metrics = evaluate(model, val_loader, config, DEVICE, DTYPE, VAL_STEPS)
            log_dict.update(val_metrics)
            log.info(
                f"  [eval] val_loss: {val_metrics['val_loss']:.4f}"
                f" | val_ppl: {val_metrics['val_perplexity']:.2f}"
            )

        wandb.log(log_dict, step=step)

        if step % LOG_INTERVAL == 0 or step == 1:
            log.info(
                f"step: {step:>{step_width}}/{NUM_STEPS} | loss: {train_loss:.2f}"
                f" | ppl: {train_ppl:.2f} | ms/step: {total_ms:.1f}"
                f" | TFLOPS: {tflops:.1f} | MFU: {utilization * 100:.1f}%"
            )

    # ---- Final validation ----
    if val_loader is not None:
        log.info("Running final validation...")
        with measure(EventType.EVAL):
            final_val = evaluate(model, val_loader, config, DEVICE, DTYPE, VAL_STEPS)
        log.info(
            f"  [final eval] val_loss: {final_val['val_loss']:.4f}"
            f" | val_ppl: {final_val['val_perplexity']:.2f}"
        )

    # ---- End of training summary ----
    wall_time = time.perf_counter() - t0
    peak_mem = torch.cuda.max_memory_allocated() / 1e9
    mean_timings = get_global_metrics().mean_ms()
    mean_total_ms = sum(mean_timings.values())
    mean_tflops = flop_counter.achieved_tflops(step_flops, mean_total_ms)
    mean_mfu = flop_counter.mfu(mean_tflops, gpu_name)

    log.info("--- Training complete ---")
    log.info(f"Wall time: {wall_time:.1f}s ({NUM_STEPS} steps)")
    log.info(f"Mean step: {mean_total_ms:.1f} ms")
    for name, ms in mean_timings.items():
        pct = 100.0 * ms / mean_total_ms if mean_total_ms > 0 else 0.0
        log.info(f"  {name:>12s}: {ms:7.2f} ms ({pct:5.1f}%)")
    log.info(f"Mean TFLOPS: {mean_tflops:.1f}")
    log.info(f"Mean MFU: {mean_mfu * 100:.1f}%")
    log.info(f"Peak memory: {peak_mem:.2f} GB")
    profiler.print_summary()

    summary = {
        "wall_time_s": wall_time,
        "mean_ms_per_step": mean_total_ms,
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


if __name__ == "__main__":
    main()
