import torch
from torch.utils.data import Dataset


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
