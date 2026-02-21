"""Tests for TokenizedDataset."""

import json

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from nanigpt.data.tokenized import TokenizedDataset

VOCAB_SIZE = 50257
SEQ_LEN = 64
NUM_SAMPLES = 16


def _write_fake_data(tmp_path, split, num_tokens, vocab_size=VOCAB_SIZE):
    """Write a fake memmap file with random tokens."""
    rng = np.random.default_rng(42)
    tokens = rng.integers(0, vocab_size, size=num_tokens, dtype=np.uint16)
    mm = np.memmap(tmp_path / f"{split}.bin", dtype=np.uint16, mode="w+", shape=(num_tokens,))
    mm[:] = tokens
    mm.flush()
    del mm

    meta = {
        "dataset": "test",
        "encoding": "gpt2",
        "vocab_size": vocab_size,
        "num_tokens": num_tokens,
        "train_tokens": num_tokens,
        "val_tokens": 0,
    }
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    return tokens


@pytest.fixture()
def data_dir(tmp_path):
    _write_fake_data(tmp_path, "train", 1000)
    _write_fake_data(tmp_path, "val", 200)
    return tmp_path


def test_dataset_length(data_dir):
    ds = TokenizedDataset(str(data_dir), "train", SEQ_LEN, NUM_SAMPLES)
    assert len(ds) == NUM_SAMPLES


def test_item_shape(data_dir):
    ds = TokenizedDataset(str(data_dir), "train", SEQ_LEN, NUM_SAMPLES)
    item = ds[0]
    assert item.shape == (SEQ_LEN + 1,)
    assert item.dtype == torch.long


def test_item_values_in_range(data_dir):
    ds = TokenizedDataset(str(data_dir), "train", SEQ_LEN, NUM_SAMPLES)
    for i in range(min(10, NUM_SAMPLES)):
        item = ds[i]
        assert item.min() >= 0
        assert item.max() < VOCAB_SIZE


def test_val_split(data_dir):
    ds = TokenizedDataset(str(data_dir), "val", SEQ_LEN, NUM_SAMPLES)
    assert len(ds) == NUM_SAMPLES
    item = ds[0]
    assert item.shape == (SEQ_LEN + 1,)


def test_too_short_raises(tmp_path):
    _write_fake_data(tmp_path, "train", SEQ_LEN)  # exactly seq_len, need seq_len + 1
    with pytest.raises(ValueError, match="need at least"):
        TokenizedDataset(str(tmp_path), "train", SEQ_LEN, NUM_SAMPLES)


def test_dataloader_compatible(data_dir):
    ds = TokenizedDataset(str(data_dir), "train", SEQ_LEN, NUM_SAMPLES)
    loader = DataLoader(ds, batch_size=4, num_workers=0)
    batch = next(iter(loader))
    assert batch.shape == (4, SEQ_LEN + 1)
    assert batch.dtype == torch.long
