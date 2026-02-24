from dataclasses import dataclass
from typing import Literal

import torch
from torch.utils.data import Dataset

from nanigpt.configurable import Configurable


class SyntheticData(Configurable):
    """Synthetic random token data for controlled experiments."""

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        seq_len: int = 256
        """Sequence length (tokens per sample)."""

        distribution: Literal["uniform", "zipf"] = "zipf"
        """Token distribution."""

        zipf_exponent: float = 1.0
        """Exponent for Zipfian distribution. Only used when distribution='zipf'."""

        num_workers: int = 2
        """DataLoader worker processes."""

        def __post_init__(self):
            if self.seq_len <= 0:
                raise ValueError(f"seq_len must be positive, got {self.seq_len}")
            if self.zipf_exponent <= 0:
                raise ValueError(f"zipf_exponent must be positive, got {self.zipf_exponent}")
            if self.num_workers < 0:
                raise ValueError(f"num_workers must be non-negative, got {self.num_workers}")

    def __init__(self, config: Config, *, vocab_size: int, num_samples: int):
        self.dataset = RandomTokenDataset(
            vocab_size=vocab_size,
            seq_len=config.seq_len,
            num_samples=num_samples,
            distribution=config.distribution,
            zipf_exponent=config.zipf_exponent,
        )
        self.num_workers = config.num_workers


class RandomTokenDataset(Dataset):
    """Generates random token sequences for training loop testing.

    Supports uniform and Zipfian (power law) token distributions.
    Tokens are generated on the fly — no storage needed.
    """

    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        num_samples: int,
        distribution: str = "uniform",
        zipf_exponent: float = 1.0,
    ):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_samples = num_samples
        self.distribution = distribution

        if distribution == "zipf":
            # Precompute Zipfian probabilities: P(rank k) ∝ 1/k^s
            ranks = torch.arange(1, vocab_size + 1, dtype=torch.float64)
            weights = 1.0 / ranks.pow(zipf_exponent)
            self.probs = (weights / weights.sum()).float()
        elif distribution != "uniform":
            raise ValueError(f"Unknown distribution: {distribution}. Use 'uniform' or 'zipf'.")

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> torch.Tensor:
        if self.distribution == "uniform":
            # +1 to seq_len so we can split into input and target in the training loop
            return torch.randint(0, self.vocab_size, (self.seq_len + 1,))
        else:
            # Zipfian: sample from precomputed distribution
            return torch.multinomial(self.probs, self.seq_len + 1, replacement=True)
