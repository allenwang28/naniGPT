"""Tests for data preparation utilities."""

from unittest.mock import MagicMock

import numpy as np

from nanigpt.data.prepare import tokenize_stream, write_memmap


def test_write_memmap(tmp_path):
    """Roundtrip write/read through memmap."""
    data = np.array([100, 200, 300, 50000], dtype=np.uint16)
    path = tmp_path / "test.bin"
    write_memmap(path, data)

    loaded = np.memmap(path, dtype=np.uint16, mode="r")
    np.testing.assert_array_equal(loaded, data)


def test_tokenize_stream_respects_limit():
    """tokenize_stream stops after reaching num_tokens."""
    # Create a mock encoder
    encoder = MagicMock()
    encoder.encode_ordinary = MagicMock(side_effect=lambda text: list(range(100)))

    # Create a fake infinite document stream
    docs = [{"text": f"document {i}"} for i in range(1000)]

    result = tokenize_stream(iter(docs), encoder, num_tokens=250)
    assert len(result) == 250
    assert result.dtype == np.uint16


def test_uint16_sufficient_for_gpt2_vocab():
    """GPT-2 vocab size (50257) fits in uint16 (max 65535)."""
    gpt2_vocab_size = 50257
    assert gpt2_vocab_size < np.iinfo(np.uint16).max
