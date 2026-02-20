"""Single-GPU training loop with profiling.

Run: uv run python -m nanigpt.train
"""

import logging
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from nanigpt.data.synthetic import RandomTokenDataset
from nanigpt.models.dense_transformer import PRESET_CONFIGS, DenseTransformer
from nanigpt.profiling import flop_counter
from nanigpt.profiling.context import init_context, step_context
from nanigpt.profiling.event_types import EventType
from nanigpt.profiling.timer import get_global_metrics, measure

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
# -----------------------------------


def main():
    # logging with dimmed timestamps and lgoging level
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

    model = DenseTransformer(config).to(DEVICE)
    num_params = sum(p.numel() for p in model.parameters())
    non_emb_params = model.num_non_embedding_params()
    log.info("Model: %s | %s params (%s non-embedding)", MODEL_PRESET,
             f"{num_params:,}", f"{non_emb_params:,}")

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
    gpu_name = flop_counter.detect_gpu()
    step_flops = flop_counter.total_flops(model, BATCH_SIZE, SEQ_LEN)

    log.info("GPU: %s | Peak: %s TFLOPS (bf16)",
             gpu_name, flop_counter.GPU_PEAK_TFLOPS.get(gpu_name, "?"))
    log.info("Batch: %d x %d tokens | Steps: %d", BATCH_SIZE, SEQ_LEN, NUM_STEPS)
    log.info("Theoretical FLOPs/step: %.2f GFLOPs", step_flops / 1e9)

    step_width = len(str(NUM_STEPS))
    header = (f"{'step':>{step_width * 2 + 1}}    {'loss':>7}  {'ms/step':>7}  {'TFLOPS':>6}   "
              f"{'MFU':>5}")
    log.info(header)

    model.train()
    data_iter = iter(loader)
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

        if step % LOG_INTERVAL == 0 or step == 1:
            timings = get_global_metrics().last_step_ms()
            total_ms = sum(timings.values())
            tflops = flop_counter.achieved_tflops(step_flops, total_ms)
            utilization = flop_counter.mfu(tflops, gpu_name)
            log.info(
                "%*d/%d  %7.2f  %7.1f  %6.1f  %5.1f%%",
                step_width, step, NUM_STEPS,
                loss.item(), total_ms, tflops, utilization * 100,
            )

    # ---- End of training summary ----
    wall_time = time.perf_counter() - t0
    peak_mem = torch.cuda.max_memory_allocated() / 1e9
    mean_timings = get_global_metrics().mean_ms()
    mean_total_ms = sum(mean_timings.values())
    mean_tflops = flop_counter.achieved_tflops(step_flops, mean_total_ms)
    mean_mfu = flop_counter.mfu(mean_tflops, gpu_name)

    log.info("--- Training complete ---")
    log.info("Wall time: %.1fs (%d steps)", wall_time, NUM_STEPS)
    log.info("Mean step: %.1f ms", mean_total_ms)
    for name, ms in mean_timings.items():
        pct = 100.0 * ms / mean_total_ms if mean_total_ms > 0 else 0.0
        log.info("  %12s: %7.2f ms (%5.1f%%)", name, ms, pct)
    log.info("Mean TFLOPS: %.1f", mean_tflops)
    log.info("Mean MFU: %.1f%%", mean_mfu * 100)
    log.info("Peak memory: %.2f GB", peak_mem)


if __name__ == "__main__":
    main()
