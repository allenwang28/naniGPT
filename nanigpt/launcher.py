"""Ray-based multi-GPU launcher.

Spawns one TrainingWorker actor per GPU. Each actor initializes its own
process group rank and runs the training loop. Ray manages CUDA_VISIBLE_DEVICES
so each actor sees device 0 locally.
"""

import logging

import ray

from nanigpt.env import MASTER_ADDR, MASTER_PORT

log = logging.getLogger("launcher")


@ray.remote(num_gpus=1)
class TrainingWorker:
    """A Ray actor that runs one rank of distributed training."""

    def __init__(self, rank: int, world_size: int, master_addr: str, master_port: str):
        self.rank = rank
        self.world_size = world_size
        MASTER_ADDR.set_value(master_addr)
        MASTER_PORT.set_value(master_port)

    def run(self, config) -> None:
        """Run the training loop for this rank with the given config."""
        from nanigpt.train import train_worker

        train_worker(self.rank, self.world_size, config)


def launch(config, num_workers: int) -> None:
    """Spawn Ray actors and run distributed training.

    Args:
        config: Validated TrainConfig to pass to each worker.
        num_workers: Number of GPU workers to spawn.
    """
    # .rayignore at project root excludes .venv from packaging.
    ray.init(ignore_reinit_error=True)

    master_addr = MASTER_ADDR.get_value()
    master_port = MASTER_PORT.get_value()

    log.info(f"Launching {num_workers} workers via Ray ({master_addr}:{master_port})")

    workers = [
        TrainingWorker.remote(rank, num_workers, master_addr, master_port)
        for rank in range(num_workers)
    ]

    futures = [w.run.remote(config) for w in workers]
    ray.get(futures)
    ray.shutdown()
    log.info("All workers finished")
