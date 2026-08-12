"""ModernLM: dense decoder-only Transformer with a modern component stack.

`generate()` deliberately mirrors the DeepSeek-V4 reference's signature
(`input_ids, max_new_tokens, eos_token_id`) so the identical evaluation harness
scores both models without a per-model adapter.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModernConfig
from .layers import MoE, RMSNorm, SwiGLU, Block, build_rope_cache


@dataclass
class ModernOutput:
    logits: torch.Tensor
    aux_loss: torch.Tensor | None = None
    mtp_logits: torch.Tensor | None = None


class MTPHead(nn.Module):
    """Multi-token prediction: predict token t+2 from hidden state t and embedding t+1.

    Off by default; retained so the staged MTP arm is a flag flip, not a rewrite.
    """

    def __init__(self, config: ModernConfig, embedding: nn.Embedding, lm_head: nn.Linear):
        super().__init__()
        self.embedding, self.lm_head = embedding, lm_head
        self.h_norm = RMSNorm(config.dim, config.norm_eps)
        self.e_norm = RMSNorm(config.dim, config.norm_eps)
        self.proj = nn.Linear(2 * config.dim, config.dim, bias=False)
        self.block = Block(config, layer_id=0)
        self.norm = RMSNorm(config.dim, config.norm_eps)

    def forward(self, hidden, input_ids, cos, sin):
        if input_ids.shape[1] < 2:
            empty = hidden.new_empty(input_ids.shape[0], 0, self.lm_head.out_features)
            return empty
        h = self.h_norm(hidden[:, :-1])
        e = self.e_norm(self.embedding(input_ids[:, 1:]))
        x = self.proj(torch.cat([h, e], dim=-1))
        x = self.block(x, cos, sin)
        return self.lm_head(self.norm(x))


class ModernLM(nn.Module):
    def __init__(self, config: ModernConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.dim)
        self.blocks = nn.ModuleList([Block(config, i) for i in range(config.n_layers)])
        self.final_norm = RMSNorm(config.dim, config.norm_eps)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.embedding.weight
        self.mtp = MTPHead(config, self.embedding, self.lm_head) if config.use_mtp else None

        cos, sin = build_rope_cache(config.max_seq_len, config.head_dim,
                                    config.rope_theta, torch.device("cpu"))
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        # Scaled init on residual-path output projections (GPT-2 convention):
        # keeps residual-stream variance from growing with depth.
        for name, param in self.named_parameters():
            if name.endswith("o_proj.weight") or name.endswith("down_proj.weight"):
                nn.init.normal_(param, mean=0.0, std=0.02 / (2 * config.n_layers) ** 0.5)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def _rope(self, device, dtype_len: int):
        if self.rope_cos.device != device:
            self.rope_cos = self.rope_cos.to(device)
            self.rope_sin = self.rope_sin.to(device)
        return self.rope_cos, self.rope_sin

    def forward(self, input_ids: torch.Tensor, return_aux_loss: bool = False,
                return_mtp_logits: bool = False,
                caches: list[dict] | None = None) -> ModernOutput:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        b, t = input_ids.shape
        past = caches[0]["k"].shape[2] if caches and "k" in caches[0] else 0
        if t + past > self.config.max_seq_len:
            raise ValueError(
                f"sequence length {t + past} exceeds max_seq_len {self.config.max_seq_len}")

        cos, sin = self._rope(input_ids.device, t)
        x = self.embedding(input_ids)
        for index, block in enumerate(self.blocks):
            x = block(x, cos, sin, caches[index] if caches is not None else None)
        hidden = self.final_norm(x)
        logits = self.lm_head(hidden)

        aux_loss = None
        if return_aux_loss and self.config.use_moe:
            terms = [blk.feed_forward.last_aux_loss for blk in self.blocks
                     if blk.is_moe and blk.feed_forward.last_aux_loss is not None]
            if terms:
                aux_loss = torch.stack(terms).mean()

        mtp_logits = None
        if return_mtp_logits and self.mtp is not None:
            mtp_logits = self.mtp(hidden, input_ids, cos, sin)

        return ModernOutput(logits=logits, aux_loss=aux_loss, mtp_logits=mtp_logits)

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int,
                 eos_token_id: int | None = None,
                 use_cache: bool = False) -> torch.Tensor:
        """Greedy decode, matching the reference implementation's semantics.

        By default this recomputes the full prefix each step, exactly like the
        DeepSeek reference, so evaluation cost is comparable and no decoding
        advantage is smuggled into the benchmark comparison.

        `use_cache=True` keeps per-layer K/V instead, which produces identical
        greedy output at a fraction of the cost -- the prefix is encoded once
        and each new token attends to the stored keys. Use it when the number is
        what matters and the wall clock is not being compared against the
        reference. Once the sequence would exceed `max_seq_len` the cache can no
        longer be extended, so decoding falls back to the uncached path.
        """
        self.eval()
        finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        caches: list[dict] | None = [{} for _ in self.blocks] if use_cache else None

        for step in range(max_new_tokens):
            if caches is not None:
                if input_ids.shape[1] >= self.config.max_seq_len:
                    caches = None          # cannot extend; drop to the slow path
                    step_input = input_ids[:, -self.config.max_seq_len:]
                else:
                    step_input = input_ids if step == 0 else input_ids[:, -1:]
            else:
                step_input = input_ids[:, -self.config.max_seq_len:]

            logits = self.forward(step_input, caches=caches).logits[:, -1, :]
            next_token = logits.argmax(dim=-1)
            if eos_token_id is not None:
                next_token = torch.where(finished,
                                         torch.full_like(next_token, eos_token_id),
                                         next_token)
                finished = finished | (next_token == eos_token_id)
            input_ids = torch.cat([input_ids, next_token[:, None]], dim=1)
            if eos_token_id is not None and bool(finished.all()):
                break
        return input_ids
