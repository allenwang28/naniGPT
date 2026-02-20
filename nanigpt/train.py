"""Single-GPU training loop with profiling.

Run: uv run python -m nanigpt.train
"""

import logging
import time
from dataclasses import asdict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import wandb
from nanigpt.data.synthetic import RandomTokenDataset
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
PROFILER_CONFIG = ProfilerConfig(
    enabled=True,
    start_step=10,
    end_step=12,
    warmup_steps=1,
    top_n=15,
    export_trace=True,
)
# -----------------------------------


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
            "profiler": TorchProfiler(PROFILER_CONFIG).config_dict(),
        },
    )

    model = DenseTransformer(config).to(DEVICE)
    num_params = sum(p.numel() for p in model.parameters())
    non_emb_params = model.num_non_embedding_params()
    log.info(f"Model: {MODEL_PRESET} | {num_params:,} params ({non_emb_params:,} non-embedding)")

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
    header = (
        f"{'step':>{step_width * 2 + 1}}    {'loss':>7}  {'ms/step':>7}  {'TFLOPS':>6}   {'MFU':>5}"
    )
    log.info(header)

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

        wandb.log(
            {
                "loss": loss.item(),
                "ms_per_step": total_ms,
                "tflops": tflops,
                "mfu": utilization,
                **{f"time/{k}": v for k, v in timings.items()},
            },
            step=step,
        )

        if step % LOG_INTERVAL == 0 or step == 1:
            log.info(
                f"{step:>{step_width}}/{NUM_STEPS}  {loss.item():7.2f}  {total_ms:7.1f}"
                f"  {tflops:6.1f}  {utilization * 100:5.1f}%"
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

    wandb.summary.update(
        {
            "wall_time_s": wall_time,
            "mean_ms_per_step": mean_total_ms,
            "mean_tflops": mean_tflops,
            "mean_mfu": mean_mfu,
            "peak_memory_gb": peak_mem,
            "num_params": num_params,
            "non_emb_params": non_emb_params,
        }
    )
    wandb.finish()


if __name__ == "__main__":
    main()
