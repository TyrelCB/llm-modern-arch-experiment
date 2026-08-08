"""Tests for the Muon optimizer and its parameter split.

The split test is the important one: the trainer builds the optimizer *after*
torch.compile wraps the model, which prefixes every parameter name with
"_orig_mod.". A name-based rule that does not unwrap silently matches nothing,
which surfaces as an empty-parameter-list crash at best and a run where no
hidden matrix is actually trained by Muon at worst.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from modern_lm.config import ModernConfig
from modern_lm.model import ModernLM
from modern_lm.muon import (Muon, build_optimizer, split_muon_params,
                            zeropower_via_newtonschulz)


def _tiny_model() -> ModernLM:
    return ModernLM(ModernConfig(vocab_size=256, max_seq_len=64, dim=64, n_layers=2,
                                 n_heads=4, n_kv_heads=4, ffn_dim=192))


def test_split_covers_every_parameter_exactly_once():
    model = _tiny_model()
    muon, decay, no_decay = split_muon_params(model)
    listed = [id(p) for p in (*muon, *decay, *no_decay)]
    assert len(listed) == len(set(listed)), "a parameter landed in two groups"
    assert set(listed) == {id(p) for p in model.parameters()}


def test_split_routes_embedding_and_head_to_adamw():
    model = _tiny_model()
    muon, decay, no_decay = split_muon_params(model)
    muon_ids = {id(p) for p in muon}
    assert id(model.embedding.weight) not in muon_ids
    assert id(model.lm_head.weight) not in muon_ids
    # Every Muon parameter is a 2D matrix from inside a transformer block.
    assert all(p.ndim == 2 for p in muon)
    assert muon, "expected a non-empty Muon group"


def test_split_is_identical_under_torch_compile():
    """Guards the _orig_mod prefix: compiled and eager must split the same."""
    model = _tiny_model()
    eager = [p.numel() for p in split_muon_params(model)[0]]
    compiled = [p.numel() for p in split_muon_params(torch.compile(model))[0]]
    assert eager == compiled
    assert sum(eager) > 0


def test_tied_embeddings_are_not_double_counted():
    model = ModernLM(ModernConfig(vocab_size=256, max_seq_len=64, dim=64, n_layers=2,
                                  n_heads=4, n_kv_heads=4, ffn_dim=192,
                                  tie_embeddings=True))
    muon, decay, no_decay = split_muon_params(model)
    listed = [id(p) for p in (*muon, *decay, *no_decay)]
    assert len(listed) == len(set(listed))


def test_build_optimizer_rejects_an_empty_muon_group():
    with pytest.raises(ValueError, match="empty"):
        build_optimizer(torch.nn.Linear(4, 4), learning_rate=3e-4,
                        muon_learning_rate=0.02, weight_decay=0.1)


def test_newtonschulz_drives_singular_values_toward_one():
    torch.manual_seed(0)
    # Rectangular: full-rank in the short dimension, so every singular value
    # should be pushed into the band the quintic coefficients target.
    g = torch.randn(96, 32)
    s = torch.linalg.svdvals(zeropower_via_newtonschulz(g, 5).float())
    assert s.min() > 0.5 and s.max() < 1.5


def test_newtonschulz_preserves_shape_both_orientations():
    for shape in ((96, 32), (32, 96), (64, 64)):
        out = zeropower_via_newtonschulz(torch.randn(*shape), 5)
        assert out.shape == shape


def test_muon_step_decreases_a_quadratic_objective():
    torch.manual_seed(0)
    w = torch.nn.Parameter(torch.randn(16, 8))
    opt = Muon([w], lr=0.02)
    first = last = None
    for _ in range(25):
        opt.zero_grad(set_to_none=True)
        loss = (w ** 2).sum()
        loss.backward()
        opt.step()
        first = loss.item() if first is None else first
        last = loss.item()
    assert last < first


def test_combined_optimizer_state_dict_round_trips():
    model = _tiny_model()
    opt = build_optimizer(model, learning_rate=3e-4, muon_learning_rate=0.02,
                          weight_decay=0.1)
    for p in model.parameters():
        p.grad = torch.randn_like(p)
    opt.step()
    restored = build_optimizer(model, learning_rate=3e-4, muon_learning_rate=0.02,
                               weight_decay=0.1)
    restored.load_state_dict(opt.state_dict())
    assert len(restored.param_groups) == len(opt.param_groups)


def test_combined_optimizer_rejects_plain_adamw_state():
    model = _tiny_model()
    opt = build_optimizer(model, learning_rate=3e-4, muon_learning_rate=0.02,
                          weight_decay=0.1)
    with pytest.raises(ValueError, match="combined"):
        opt.load_state_dict(torch.optim.AdamW(model.parameters()).state_dict())


def test_param_groups_carry_their_own_lr_scale():
    model = _tiny_model()
    opt = build_optimizer(model, learning_rate=3e-4, muon_learning_rate=0.02,
                          weight_decay=0.1)
    scales = {g["lr_scale"] for g in opt.param_groups}
    assert scales == {0.02, 3e-4}
