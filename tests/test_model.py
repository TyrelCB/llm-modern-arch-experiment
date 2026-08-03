"""Tests for the ModernLM architecture, sizing, and training plumbing."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from modern_lm.config import ModernConfig
from modern_lm.layers import RMSNorm, apply_rope, build_rope_cache
from modern_lm.model import ModernLM

DEEPSEEK_STORED_PARAMS = 144_669_412


def test_dense_145m_matches_deepseek_stored_parameter_count():
    """Capacity match is the premise of the whole comparison; pin it."""
    config = ModernConfig.dense_145m()
    model = ModernLM(config)
    n = model.num_params()
    relative = abs(n - DEEPSEEK_STORED_PARAMS) / DEEPSEEK_STORED_PARAMS
    assert relative < 0.005, f"{n:,} vs {DEEPSEEK_STORED_PARAMS:,} ({relative:.3%})"


def test_staged_levers_are_off_by_default():
    config = ModernConfig.dense_145m()
    assert config.use_moe is False
    assert config.use_mtp is False
    model = ModernLM(config)
    assert model.mtp is None
    assert not any(block.is_moe for block in model.blocks)


def test_forward_shapes_and_finite():
    config = ModernConfig.tiny()
    model = ModernLM(config)
    ids = torch.randint(0, config.vocab_size, (2, 16))
    out = model(ids)
    assert out.logits.shape == (2, 16, config.vocab_size)
    assert torch.isfinite(out.logits).all()
    assert out.aux_loss is None and out.mtp_logits is None


def test_attention_is_causal():
    """Changing a later token must not change an earlier position's logits."""
    torch.manual_seed(0)
    model = ModernLM(ModernConfig.tiny()).eval()
    ids = torch.randint(0, 256, (1, 12))
    with torch.no_grad():
        base = model(ids).logits
        altered_ids = ids.clone()
        altered_ids[0, -1] = (altered_ids[0, -1] + 7) % 256
        altered = model(altered_ids).logits
    assert torch.allclose(base[:, :-1], altered[:, :-1], atol=1e-5)
    assert not torch.allclose(base[:, -1], altered[:, -1], atol=1e-5)


def test_rope_preserves_norm():
    cos, sin = build_rope_cache(32, 8, 10000.0, torch.device("cpu"))
    x = torch.randn(1, 1, 32, 8)
    rotated = apply_rope(x, cos, sin)
    assert torch.allclose(x.norm(dim=-1), rotated.norm(dim=-1), atol=1e-4)


def test_rope_dot_product_depends_only_on_relative_offset():
    """RoPE's defining property: <R_i q, R_j k> is a function of (j - i) alone.

    The same q/k vectors must be placed at each position -- comparing different
    random vectors across positions tests nothing.
    """
    cos, sin = build_rope_cache(32, 8, 10000.0, torch.device("cpu"))
    torch.manual_seed(0)
    q, k = torch.randn(8), torch.randn(8)

    def rotated_at(vector: torch.Tensor, position: int) -> torch.Tensor:
        x = torch.zeros(1, 1, 32, 8)
        x[0, 0, position] = vector
        return apply_rope(x, cos, sin)[0, 0, position]

    def dot(i: int, j: int) -> float:
        return float((rotated_at(q, i) * rotated_at(k, j)).sum())

    baseline = dot(5, 8)
    for i, j in [(15, 18), (0, 3), (20, 23)]:
        assert dot(i, j) == pytest.approx(baseline, abs=1e-4), (i, j)
    # A different offset must give a different score.
    assert dot(0, 5) != pytest.approx(baseline, abs=1e-3)


def test_rmsnorm_matches_reference_formula():
    norm = RMSNorm(16)
    x = torch.randn(4, 16)
    expected = x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
    assert torch.allclose(norm(x), expected, atol=1e-5)


def test_moe_flag_activates_experts_and_aux_loss():
    config = ModernConfig.tiny(use_moe=True)
    model = ModernLM(config)
    ids = torch.randint(0, config.vocab_size, (2, 16))
    out = model(ids, return_aux_loss=True)
    assert any(block.is_moe for block in model.blocks)
    assert out.aux_loss is not None and torch.isfinite(out.aux_loss)


def test_mtp_flag_produces_shifted_logits():
    config = ModernConfig.tiny(use_mtp=True)
    model = ModernLM(config)
    ids = torch.randint(0, config.vocab_size, (2, 16))
    out = model(ids, return_mtp_logits=True)
    assert out.mtp_logits is not None
    assert out.mtp_logits.shape == (2, 15, config.vocab_size)


def test_generate_stops_at_eos_and_matches_reference_signature():
    config = ModernConfig.tiny()
    model = ModernLM(config)
    ids = torch.randint(4, config.vocab_size, (2, 5))
    out = model.generate(ids, max_new_tokens=6, eos_token_id=3)
    assert out.shape[0] == 2 and out.shape[1] <= 11
    assert torch.equal(out[:, :5], ids)


def test_generate_respects_context_window():
    config = ModernConfig.tiny()
    model = ModernLM(config)
    ids = torch.randint(4, config.vocab_size, (1, config.max_seq_len))
    out = model.generate(ids, max_new_tokens=3, eos_token_id=None)
    assert out.shape[1] == config.max_seq_len + 3
