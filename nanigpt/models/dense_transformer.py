"""Dense transformer for next-token prediction.

Standard GPT-style architecture: token embeddings, N transformer blocks
(pre-norm, multi-head self-attention with RoPE, feedforward with GELU),
and a tied output projection. Uses PyTorch's scaled_dot_product_attention
which dispatches to Flash Attention when available.

Implements the FlopCountable protocol for analytical FLOP counting:
linear layers contribute 2 * params * tokens, attention contributes
4 * B * n_heads * S^2 * d_head * n_layers.
"""

from dataclasses import dataclass
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TransformerConfig:
    vocab_size: int = 50257
    d_model: int = 768
    n_heads: int = 12
    n_layers: int = 12
    d_ff: int = 3072
    max_seq_len: int = 1024
    dropout: float = 0.0


# Vocab size 50257 matches GPT-2's BPE tokenizer (50k merges + 256 bytes + 1 EOT).
# Model dimensions are scaled-down variants of GPT-2.
PRESET_CONFIGS = {
    "small": TransformerConfig(
        vocab_size=50257,
        d_model=256,
        n_heads=8,
        n_layers=8,
        d_ff=1024,
        max_seq_len=512,
    ),
    "medium": TransformerConfig(
        vocab_size=50257,
        d_model=512,
        n_heads=8,
        n_layers=12,
        d_ff=2048,
        max_seq_len=1024,
    ),
    "large": TransformerConfig(
        vocab_size=50257,
        d_model=768,
        n_heads=12,
        n_layers=12,
        d_ff=3072,
        max_seq_len=1024,
    ),
}


class ModelOutput(NamedTuple):
    logits: torch.Tensor
    aux_loss: torch.Tensor


def precompute_rope_freqs(d_head: int, max_seq_len: int, theta: float = 10000.0) -> torch.Tensor:
    """Precompute the complex exponentials for RoPE.

    Returns shape (max_seq_len, d_head // 2) complex64 tensor.
    """
    freqs = 1.0 / (theta ** (torch.arange(0, d_head, 2).float() / d_head))
    positions = torch.arange(max_seq_len).float()
    angles = torch.outer(positions, freqs)
    return torch.polar(torch.ones_like(angles), angles)


def apply_rope(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Apply rotary positional embeddings to input tensor.

    x: (batch, n_heads, seq_len, d_head)
    freqs: (seq_len, d_head // 2) complex
    """
    # Reshape x into pairs of consecutive dimensions and view as complex
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    # freqs shape: (1, 1, seq_len, d_head // 2)
    freqs = freqs.unsqueeze(0).unsqueeze(0)
    x_rotated = x_complex * freqs[:, :, :x_complex.shape[2], :]
    return torch.view_as_real(x_rotated).reshape(*x.shape).type_as(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.d_head = config.d_model // config.n_heads

        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.register_buffer(
            "rope_freqs",
            precompute_rope_freqs(self.d_head, config.max_seq_len),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape

        q = self.q_proj(x).view(B, S, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.n_heads, self.d_head).transpose(1, 2)

        q = apply_rope(q, self.rope_freqs)
        k = apply_rope(k, self.rope_freqs)

        # F.scaled_dot_product_attention dispatches to Flash Attention when available
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=True,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
        )

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, D)
        return self.resid_dropout(self.out_proj(attn_out))


class FeedForward(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.up = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.down = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down(F.gelu(self.up(x))))


class TransformerBlock(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = MultiHeadAttention(config)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.ffn = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class DenseTransformer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config

        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying between token embedding and lm_head
        self.lm_head.weight = self.token_emb.weight

    def forward(self, input_ids: torch.Tensor) -> ModelOutput:
        x = self.drop(self.token_emb(input_ids))

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        return ModelOutput(logits=logits, aux_loss=torch.tensor(0.0, device=input_ids.device))

    def num_non_embedding_params(self) -> int:
        """Count parameters excluding token embeddings (which are tied to lm_head)."""
        total = sum(p.numel() for p in self.parameters())
        embedding_params = self.token_emb.weight.numel()
        return total - embedding_params

    def flop_count(self, batch_size: int, seq_len: int) -> int:
        """Analytical forward-pass FLOP count.

        Standard approximation: ~6 * N * B * S for forward pass
        where N = non-embedding params, B = batch_size, S = seq_len.

        The factor of 6 comes from: 2 (multiply-add per param) * 3 (fwd + bwd ≈ 3x fwd).
        For forward only, use 2 * N * B * S.

        We also add the attention logits computation: 2 * B * n_heads * S^2 * d_head * n_layers
        """
        N = self.num_non_embedding_params()
        # Linear layers: 2 FLOPs per parameter per token (multiply-add)
        linear_flops = 2 * N * batch_size * seq_len

        # Attention QK^T and attn @ V: 2 * 2 * B * n_heads * S^2 * d_head * n_layers
        d_head = self.config.d_model // self.config.n_heads
        attn_flops = (
            4 * batch_size * self.config.n_heads * seq_len * seq_len * d_head * self.config.n_layers
        )

        return linear_flops + attn_flops
