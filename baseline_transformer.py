"""Non-recurrent baseline Transformer for ArcNeuron ablations (TEMPORARY).

This file is NOT part of the permanent ArcNeuron repository.  It defines a
plain decoder-only Transformer that reuses the exact same block components as
ArcNeuron (RMSNorm, RoPE GQA attention, SwiGLU) but stacks N distinct blocks
instead of reusing a shared recurrent core.  It exists only to answer the
mandatory baseline question: does recurrent depth beat a conventional
parameter-matched Transformer at the same total block count?

The class is drop-in compatible with train.py / eval_overnight.py: it exposes
``.config``, ``forward(tokens, depth=...)`` (depth is accepted and ignored so
the same eval depth-sweep code runs unchanged), and standard nn.Module params.
"""

from dataclasses import dataclass

import torch
from torch import Tensor, nn

# Reuse ArcNeuron's components so the comparison is about recurrence, not about
# different attention or MLP implementations.
import sys
import pathlib
# Support both the Modal container layout (/root/arcneuron) and local runs.
_repo = str(pathlib.Path(__file__).resolve().parent)
if _repo not in sys.path:
    sys.path.insert(0, _repo)
from arcneuron import RMSNorm, CausalSelfAttention, SwiGLU, TransformerBlock


@dataclass(frozen=True)
class BaselineConfig:
    """Mirror of ArcNeuronConfig but with one stacked layer count."""

    vocab_size: int
    dim: int = 256
    n_heads: int = 8
    n_kv_heads: int = 2
    ffn_dim: int = 704
    max_seq_len: int = 256
    n_layers: int = 4  # Total stacked blocks; matched to prelude+core+coda of ArcNeuron.
    rope_theta: float = 10_000.0

    def __post_init__(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if self.dim % self.n_heads != 0:
            raise ValueError("dim must be divisible by n_heads")
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        if (self.dim // self.n_heads) % 2 != 0:
            raise ValueError("attention head dimension must be even for RoPE")
        if self.n_layers <= 0:
            raise ValueError("n_layers must be positive")


class BaselineTransformer(nn.Module):
    """Plain stacked Transformer with the same block primitives as ArcNeuron."""

    def __init__(self, config: BaselineConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.dim)
        self.blocks = nn.ModuleList(
            TransformerBlock(_to_arc_config(config), zero_residual_outputs=False)
            for _ in range(config.n_layers)
        )
        self.final_norm = RMSNorm(config.dim)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight  # Weight tying matches ArcNeuron exactly.
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens: Tensor, depth: int = 1) -> Tensor:
        # depth is accepted for compatibility with the eval depth-sweep but
        # ignored: a non-recurrent model always runs its fixed stack once.
        if tokens.ndim != 2:
            raise ValueError("tokens must have shape [batch, sequence]")
        if tokens.size(1) > self.config.max_seq_len:
            raise ValueError("token sequence exceeds BaselineConfig.max_seq_len")
        x = self.embedding(tokens)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        return self.lm_head(x)


def _to_arc_config(cfg: BaselineConfig):
    # Build a minimal ArcNeuronConfig-compatible object for the shared block
    # components, which only read vocab_size/dim/heads/kv/ffn/max_seq_len/rope.
    from arcneuron import ArcNeuronConfig

    return ArcNeuronConfig(
        vocab_size=cfg.vocab_size,
        dim=cfg.dim,
        n_heads=cfg.n_heads,
        n_kv_heads=cfg.n_kv_heads,
        ffn_dim=cfg.ffn_dim,
        max_seq_len=cfg.max_seq_len,
        prelude_layers=0,
        core_layers=1,
        coda_layers=0,
    )
