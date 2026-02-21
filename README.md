# naniGPT

An exploration of what it looks like to build a frontier training framework — by actually building one.

This is not a production system or a library. A from-scratch study of the design decisions, tradeoffs, and systems behind real training infrastructure. Every module is an opportunity to try different approaches and develop informed opinions.

Measurement is built into everything. Synthetic data before real data. Rust where it helps, Python where it doesn't, and numbers to tell the difference.

## Quick Start

```bash
uv sync
uv run python -m nanigpt.train
```

## Data Preparation

By default, training uses synthetic random tokens (Zipf distribution). To train on real text data, prepare a tokenized dataset first:

```bash
# Small dataset for debugging
uv run python -m nanigpt.data.prepare --num-tokens 1_000_000 --output-dir data/fineweb-1M

# Larger dataset for real runs
uv run python -m nanigpt.data.prepare --num-tokens 100_000_000 --output-dir data/fineweb-100M
```

This streams from [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu), tokenizes with tiktoken's GPT-2 encoding, and writes `train.bin`, `val.bin` (uint16 memmap), and `meta.json` to the output directory.

Options: `--dataset`, `--subset`, `--encoding`, `--num-tokens`, `--val-fraction` (default 1%), `--seed`.

Then set `DATA_DIR` in `nanigpt/train.py` to the output path to enable real data training with validation perplexity tracking.
