"""Dataset reading pre-tokenized memmap files produced by prepare.py."""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from nanigpt.configurable import Configurable

log = logging.getLogger("tokenized")


class TokenizedData(Configurable):
    """Pre-tokenized memmap data produced by prepare.py."""

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        data_dir: str = "data/fineweb-1M"
        """Path to directory containing {split}.bin and meta.json."""

        seq_len: int = 256
        """Sequence length (tokens per sample)."""

        num_workers: int = 2
        """DataLoader worker processes."""

        def __post_init__(self):
            if self.seq_len <= 0:
                raise ValueError(f"seq_len must be positive, got {self.seq_len}")
            if not self.data_dir:
                raise ValueError("data_dir must be non-empty for TokenizedData.Config")
            if self.num_workers < 0:
                raise ValueError(f"num_workers must be non-negative, got {self.num_workers}")

    def __init__(self, config: Config, *, split: str, num_samples: int):
        self.dataset = TokenizedDataset(
            config.data_dir,
            split,
            config.seq_len,
            num_samples,
        )
        self.num_workers = config.num_workers


class TokenizedDataset(Dataset):
    """Reads contiguous token chunks from a memory-mapped binary file.

    Backs the file with np.memmap so the OS pages in only the chunks that
    are actually accessed — a 1GB token file doesn't require 1GB of RSS.
    Each __getitem__ samples a random offset and returns (seq_len + 1,)
    contiguous tokens as int64, ready to split into input_ids[:, :-1] and
    targets[:, 1:]. The idx argument is ignored; offsets are uniformly
    random over the file, so every epoch sees a different set of windows.

    num_samples controls the nominal epoch length (typically NUM_STEPS *
    BATCH_SIZE) since the underlying data is a flat array with no natural
    document boundaries.
    """

    def __init__(self, data_dir: str, split: str, seq_len: int, num_samples: int):
        self.seq_len = seq_len
        self.num_samples = num_samples

        data_path = Path(data_dir)
        bin_path = data_path / f"{split}.bin"
        if not bin_path.exists():
            raise FileNotFoundError(f"Data file not found: {bin_path}")

        self.data = np.memmap(bin_path, dtype=np.uint16, mode="r")

        if len(self.data) < seq_len + 1:
            raise ValueError(
                f"Data file {bin_path} has {len(self.data)} tokens, "
                f"need at least {seq_len + 1} (seq_len + 1)"
            )

        self.max_offset = len(self.data) - seq_len - 1

        # Load and log metadata if available
        meta_path = data_path / "meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            log.info(
                f"Loaded {split} split: {len(self.data):,} tokens "
                f"(dataset={meta.get('dataset', '?')}, encoding={meta.get('encoding', '?')})"
            )
        else:
            log.info(f"Loaded {split} split: {len(self.data):,} tokens")

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> torch.Tensor:
        offset = torch.randint(0, self.max_offset + 1, (1,)).item()
        chunk = self.data[offset : offset + self.seq_len + 1]
        return torch.from_numpy(chunk.astype(np.int64))
