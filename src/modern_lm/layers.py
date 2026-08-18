"""Modern Transformer components: RMSNorm, RoPE, SwiGLU, GQA attention, MoE.

Each piece replaces a 2019-era equivalent in the GPTMicro-style stack:
LayerNorm -> RMSNorm, learned position embeddings -> RoPE, GELU MLP -> SwiGLU,
MHA -> GQA (configurable, defaults to full MHA).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModernConfig


class RMSNorm(nn.Module):
    """Root-mean-square layer norm: no mean subtraction, no bias."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize in fp32 for stability, then return to the input dtype so the
        # bf16 autocast path does not silently upcast the rest of the block.
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dtype)


def build_rope_cache(seq_len: int, head_dim: int, theta: float,
                     device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute cos/sin tables of shape [seq_len, head_dim/2]."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    positions = torch.arange(seq_len, device=device).float()
    angles = torch.outer(positions, inv_freq)
    return angles.cos(), angles.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
               offset: int = 0) -> torch.Tensor:
    """Rotate query/key pairs. `x` is [B, H, T, head_dim].

    `offset` is the absolute position of the first row. Incremental decoding
    feeds one token at a time, and it must be rotated at its real position in
    the sequence -- rotating it at position 0 would silently produce a different
    model than the uncached path.
    """
    t = x.shape[2]
    cos = cos[offset:offset + t].to(x.dtype)[None, None, :, :]
    sin = sin[offset:offset + t].to(x.dtype)[None, None, :, :]
    x1, x2 = x.float().chunk(2, dim=-1)
    cos, sin = cos.float(), sin.float()
    rotated = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return rotated.to(x.dtype)


class Attention(nn.Module):
    """Grouped-query causal attention with optional QK-norm.

    Uses F.scaled_dot_product_attention so the [T, T] score matrix is never
    materialized (the same reason GPTMicro uses SDPA).
    """

    def __init__(self, config: ModernConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.n_rep = config.n_heads // config.n_kv_heads
        self.dropout = config.dropout

        q_rows = config.n_heads * self.head_dim
        kv_rows = config.n_kv_heads * self.head_dim
        self.qkv_splits = (q_rows, kv_rows, kv_rows)
        self.fused = config.fuse_projections
        if self.fused:
            # One [q+k+v, dim] matmul instead of three. Same arithmetic, one
            # read of `x` instead of three, one kernel launch instead of three
            # (D028). `muon_row_blocks` tells the optimizer where the original
            # matrices are inside this one, so Muon still orthogonalizes q, k
            # and v separately -- concatenating first would change the update.
            self.qkv_proj = nn.Linear(config.dim, sum(self.qkv_splits), bias=False)
            self.qkv_proj.muon_row_blocks = self.qkv_splits
        else:
            self.q_proj = nn.Linear(config.dim, q_rows, bias=False)
            self.k_proj = nn.Linear(config.dim, kv_rows, bias=False)
            self.v_proj = nn.Linear(config.dim, kv_rows, bias=False)
        self.o_proj = nn.Linear(config.n_heads * self.head_dim, config.dim, bias=False)

        # QK-norm materially reduces attention-logit blowup at the sustained
        # high learning rates this project's schedules use.
        self.q_norm = RMSNorm(self.head_dim, config.norm_eps) if config.use_qk_norm else None
        self.k_norm = RMSNorm(self.head_dim, config.norm_eps) if config.use_qk_norm else None

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                cache: dict | None = None) -> torch.Tensor:
        b, t, _ = x.shape
        if self.fused:
            q, k, v = self.qkv_proj(x).split(self.qkv_splits, dim=-1)
        else:
            q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        q = q.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)

        if self.q_norm is not None:
            q, k = self.q_norm(q), self.k_norm(k)

        # Past length sets the rotation offset, so a token decoded incrementally
        # is rotated at the same absolute position the uncached path would use.
        past = cache["k"].shape[2] if cache is not None and "k" in cache else 0
        q, k = apply_rope(q, cos, sin, past), apply_rope(k, cos, sin, past)

        if cache is not None:
            if "k" in cache:
                k = torch.cat([cache["k"], k], dim=2)
                v = torch.cat([cache["v"], v], dim=2)
            # Store pre-repeat, so GQA keeps its memory advantage in the cache.
            cache["k"], cache["v"] = k, v

        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        # A single query row attends to the whole cached prefix, so the causal
        # mask must be off there -- is_causal with t=1 would mask everything but
        # the newest key and silently corrupt decoding.
        causal = t > 1
        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=causal)
        out = out.transpose(1, 2).contiguous().view(b, t, -1)
        return self.o_proj(out)


class SwiGLU(nn.Module):
    """Gated feed-forward: (silu(gate(x)) * up(x)) -> down."""

    def __init__(self, dim: int, ffn_dim: int, fuse: bool = False):
        super().__init__()
        self.fused = fuse
        self.ffn_dim = ffn_dim
        if fuse:
            # gate and up read the same `x` and have the same shape, so they are
            # the clearest fusion in the block: one [2*ffn, dim] matmul (D028).
            self.gate_up_proj = nn.Linear(dim, 2 * ffn_dim, bias=False)
            self.gate_up_proj.muon_row_blocks = (ffn_dim, ffn_dim)
        else:
            self.gate_proj = nn.Linear(dim, ffn_dim, bias=False)
            self.up_proj = nn.Linear(dim, ffn_dim, bias=False)
        self.down_proj = nn.Linear(ffn_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.fused:
            gate, up = self.gate_up_proj(x).split((self.ffn_dim, self.ffn_dim), dim=-1)
        else:
            gate, up = self.gate_proj(x), self.up_proj(x)
        return self.down_proj(F.silu(gate) * up)


class MoE(nn.Module):
    """Top-k routed experts plus always-on shared experts.

    Off by default. Present so the staged MoE arm needs no refactor; it is not
    exercised by the dense baseline run.
    """

    def __init__(self, config: ModernConfig):
        super().__init__()
        self.config = config
        self.experts_per_token = config.experts_per_token
        self.router = nn.Linear(config.dim, config.n_routed_experts, bias=False)
        self.routed = nn.ModuleList(
            [SwiGLU(config.dim, config.ffn_dim, config.fuse_projections)
             for _ in range(config.n_routed_experts)])
        self.shared = nn.ModuleList(
            [SwiGLU(config.dim, config.ffn_dim, config.fuse_projections)
             for _ in range(config.n_shared_experts)])
        self.last_aux_loss: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        flat = x.reshape(-1, d)
        logits = self.router(flat.float())
        probs = torch.softmax(logits, dim=-1)
        weights, indices = torch.topk(probs, self.experts_per_token, dim=-1)
        weights = (weights / weights.sum(dim=-1, keepdim=True)).to(x.dtype)

        # Sequence-level load-balance loss (Switch-Transformer form).
        mask = torch.zeros_like(probs).scatter_(1, indices, 1.0)
        self.last_aux_loss = (
            self.config.n_routed_experts * (mask.mean(0) * probs.mean(0)).sum())

        out = torch.zeros_like(flat)
        for expert_id, expert in enumerate(self.routed):
            token_idx, slot_idx = (indices == expert_id).nonzero(as_tuple=True)
            if token_idx.numel() == 0:
                continue
            contribution = expert(flat[token_idx]) * weights[token_idx, slot_idx, None]
            out.index_add_(0, token_idx, contribution.to(out.dtype))
        for expert in self.shared:
            out = out + expert(flat)
        return out.view(b, t, d)


class Block(nn.Module):
    """Decoder block. Pre-norm by default; SiameseNorm two-stream when enabled.

    Pre-norm (`use_siamese_norm=False`) is the historical path and is byte-for-byte
    unchanged: RMSNorm -> attention -> RMSNorm -> feed-forward, one residual
    stream, `forward` takes and returns a tensor.

    The local Siamese/HybridNorm experiment instead carries two streams. Y
    accumulates unnormalized (the Pre-LN identity gradient path); X is
    re-normalized after each combined block update (the Post-LN bounded path).
    One shared block body reads their fusion, so the two streams cost activations,
    not parameters:

        Y' = LN_Y(Y);  O = F(X + Y');  X <- LN_X(X + O / sqrt(l+1));  Y <- Y + O

    The `1/sqrt(l+1)` divisor applies only on the way into the X-stream -- Y
    accumulates the raw `O`. Inside `F` the sub-block is HybridNorm rather than
    our Pre-LN pair, with a learnable `gamma` mixing the normalized and raw
    attention input. This is not the paper's current reference algorithm; see
    docs/decisions.md#d018.
    """

    def __init__(self, config: ModernConfig, layer_id: int):
        super().__init__()
        self.attn_norm = RMSNorm(config.dim, config.norm_eps)
        self.attn = Attention(config)
        self.ffn_norm = RMSNorm(config.dim, config.norm_eps)
        use_moe = config.use_moe and (layer_id % config.moe_every == 0)
        self.feed_forward = (MoE(config) if use_moe
                             else SwiGLU(config.dim, config.ffn_dim, config.fuse_projections))
        self.is_moe = use_moe

        self.use_siamese = config.use_siamese_norm
        if self.use_siamese:
            self.y_norm = RMSNorm(config.dim, config.norm_eps)
            self.x_norm = RMSNorm(config.dim, config.norm_eps)
            # HybridNorm places a second norm *after* the attention sub-layer,
            # inside the residual function, which Pre-LN does not have.
            self.attn_post_norm = RMSNorm(config.dim, config.norm_eps)
            # Their learnable mixing vector on the attention input. Init 1.0 =
            # fully normalized, i.e. identical to Pre-LN at step 0, so the
            # architecture starts from the baseline and learns its way off it.
            self.gamma = nn.Parameter(torch.ones(config.dim))
            self.residual_scale = (layer_id + 1) ** -0.5
        else:
            self.residual_scale = 1.0

    def _body(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
              cache: dict | None) -> torch.Tensor:
        """The shared residual function F, in its HybridNorm form."""
        normed = self.attn_norm(x)
        # gamma interpolates between the normalized and raw attention input;
        # at gamma=1 this is exactly the Pre-LN input.
        attn_in = self.gamma * normed + (1.0 - self.gamma) * x
        h = x + self.attn_post_norm(self.attn(attn_in, cos, sin, cache))
        return (h + self.feed_forward(self.ffn_norm(h))) - x

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                cache: dict | None = None,
                y: torch.Tensor | None = None):
        if not self.use_siamese:
            x = x + self.attn(self.attn_norm(x), cos, sin, cache)
            x = x + self.feed_forward(self.ffn_norm(x))
            return x

        update = self._body(x + self.y_norm(y), cos, sin, cache)
        return self.x_norm(x + update * self.residual_scale), y + update
