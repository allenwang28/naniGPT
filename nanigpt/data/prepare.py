"""Offline data preparation: download, tokenize, and write memmap files.

Run:
    uv run python -m nanigpt.data.prepare --num-tokens 10_000_000 --output-dir data/fineweb-10M
"""

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np

log = logging.getLogger("prepare")


def tokenize_stream(dataset_iter, encoder, num_tokens: int) -> np.ndarray:
    """Tokenize documents from an iterable until we reach num_tokens."""
    tokens: list[int] = []
    docs = 0
    t0 = time.perf_counter()
    last_log = 0

    for doc in dataset_iter:
        encoded = encoder.encode_ordinary(doc["text"])
        tokens.extend(encoded)
        docs += 1

        # Progress logging every 1M tokens
        if len(tokens) - last_log >= 1_000_000:
            elapsed = time.perf_counter() - t0
            tps = len(tokens) / elapsed
            log.info(f"  {len(tokens):,} tokens ({docs:,} docs) — {tps:,.0f} tok/s")
            last_log = len(tokens)

        if len(tokens) >= num_tokens:
            break

    return np.array(tokens[:num_tokens], dtype=np.uint16)


def write_memmap(path: Path, data: np.ndarray) -> None:
    """Write a numpy array to a memory-mapped file."""
    mm = np.memmap(path, dtype=data.dtype, mode="w+", shape=data.shape)
    mm[:] = data[:]
    mm.flush()
    del mm


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="\033[2m%(asctime)s %(levelname)s\033[0m %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Prepare tokenized dataset")
    parser.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    parser.add_argument("--subset", default="sample-10BT")
    parser.add_argument("--encoding", default="gpt2")
    parser.add_argument("--num-tokens", type=int, default=10_000_000)
    parser.add_argument("--val-fraction", type=float, default=0.01)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import datasets
    import tiktoken

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Loading dataset: {args.dataset} ({args.subset})")
    ds = datasets.load_dataset(args.dataset, args.subset, split="train", streaming=True)
    ds = ds.shuffle(seed=args.seed, buffer_size=10_000)

    log.info(f"Tokenizer: {args.encoding}")
    enc = tiktoken.get_encoding(args.encoding)

    log.info(f"Tokenizing up to {args.num_tokens:,} tokens...")
    t0 = time.perf_counter()
    all_tokens = tokenize_stream(iter(ds), enc, args.num_tokens)
    elapsed = time.perf_counter() - t0
    log.info(f"Tokenized {len(all_tokens):,} tokens in {elapsed:.1f}s")

    # Validate uint16 is sufficient
    max_token = int(all_tokens.max())
    assert max_token < 65535, f"Token {max_token} exceeds uint16 range"

    # Split into train/val
    val_size = int(len(all_tokens) * args.val_fraction)
    train_tokens = all_tokens[: len(all_tokens) - val_size]
    val_tokens = all_tokens[len(all_tokens) - val_size :]

    log.info(f"Train: {len(train_tokens):,} tokens")
    log.info(f"Val:   {len(val_tokens):,} tokens")

    write_memmap(output_dir / "train.bin", train_tokens)
    write_memmap(output_dir / "val.bin", val_tokens)

    meta = {
        "dataset": args.dataset,
        "subset": args.subset,
        "encoding": args.encoding,
        "vocab_size": enc.n_vocab,
        "num_tokens": len(all_tokens),
        "train_tokens": len(train_tokens),
        "val_tokens": len(val_tokens),
        "val_fraction": args.val_fraction,
        "seed": args.seed,
        "dtype": "uint16",
    }
    meta_path = output_dir / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    log.info(f"Wrote {meta_path}")


if __name__ == "__main__":
    main()
