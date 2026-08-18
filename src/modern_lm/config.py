"""Configuration for the modern dense baseline.

The default preset is sized to match the DeepSeek-V4 reference's *stored*
parameter count (144,669,412) so the two models hold equal capacity. See
`docs/comparison-protocol.md` for why stored rather than active was chosen.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace


@dataclass(frozen=True)
class ModernConfig:
    # Vocabulary and context are fixed by the shared FineMath artifacts; changing
    # either would break loss comparability with the DeepSeek baseline.
    vocab_size: int = 16384
    max_seq_len: int = 512

    dim: int = 768
    n_layers: int = 15
    n_heads: int = 12
    # Grouped-query attention. Equal to n_heads means full MHA, which is the
    # default for the comparison: GQA's win is inference KV memory, not training
    # throughput, so enabling it here would change capacity without a matching
    # benefit to measure.
    n_kv_heads: int = 12
    ffn_dim: int = 2432

    norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    dropout: float = 0.0

    # Modern-stack switches. All default to the settings actually used in the
    # headline comparison run.
    use_qk_norm: bool = True
    tie_embeddings: bool = False

    # Fuse Q/K/V into one projection and SwiGLU gate/up into another. Purely a
    # systems change in exact arithmetic: the parameter count and function are
    # unchanged. GEMM reduction order moves fp32 gradients by up to 4e-7, while
    # block-aware Muon preserves the original sub-matrix update for identical
    # gradients (see modern_lm.muon and docs/decisions.md#d028). It changes the
    # state dict, so existing checkpoints convert through modern_lm.fusion.
    #
    # Off by default: compiled GB10 tests at 50M and 300M found no throughput or
    # memory win (D031). Retained for other kernels, precisions, and hardware.
    fuse_projections: bool = False

    # Local two-stream Siamese/HybridNorm variant inspired by arXiv 2602.08064.
    # It is not faithful to the paper's current reference implementation; see
    # docs/decisions.md#d018. Off by default because the accepted architecture is
    # single-stream Pre-LN and this branch changes the state dict.
    use_siamese_norm: bool = False

    # Staged levers, deliberately off for the dense baseline so the first
    # comparison isolates the dense modern stack (see docs/comparison-protocol.md).
    use_mtp: bool = False
    mtp_weight: float = 0.1
    use_moe: bool = False
    n_routed_experts: int = 16
    n_shared_experts: int = 1
    experts_per_token: int = 2
    moe_aux_weight: float = 0.01
    moe_every: int = 1  # apply MoE on every Nth block when use_moe is set

    def __post_init__(self):
        if self.dim % self.n_heads:
            raise ValueError("dim must be divisible by n_heads")
        if self.n_heads % self.n_kv_heads:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        head_dim = self.dim // self.n_heads
        if head_dim % 2:
            raise ValueError("head_dim must be even for RoPE")
        if self.use_moe and self.experts_per_token > self.n_routed_experts:
            raise ValueError("experts_per_token exceeds n_routed_experts")

    @property
    def head_dim(self) -> int:
        return self.dim // self.n_heads

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def dense_145m(cls, **overrides) -> "ModernConfig":
        """Phase I capacity-matched dense baseline: 144,630,912 parameters."""
        return replace(cls(), **overrides)

    @classmethod
    def tiny(cls, **overrides) -> "ModernConfig":
        """Small config for tests."""
        base = cls(vocab_size=256, max_seq_len=64, dim=64, n_layers=2, n_heads=4,
                   n_kv_heads=4, ffn_dim=192, n_routed_experts=4,
                   experts_per_token=2)
        return replace(base, **overrides)
