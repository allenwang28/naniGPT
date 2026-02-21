"""Dataset reading pre-tokenized memmap files produced by prepare.py."""

import json
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

log = logging.getLogger("tokenized")


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
