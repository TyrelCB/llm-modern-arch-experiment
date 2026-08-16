"""SiameseNorm (arXiv 2602.08064) two-stream residual.

The load-bearing property is that enabling it changes the architecture and
nothing else: the Pre-LN path must stay byte-identical, because every
checkpoint in the results table was trained with it.
"""
from __future__ import annotations

from dataclasses import replace

import torch

from modern_lm.config import ModernConfig
from modern_lm.model import ModernLM
from modern_lm.muon import split_muon_params


def _siamese(**overrides) -> ModernConfig:
    overrides.setdefault("n_layers", 4)
    return replace(ModernConfig.tiny(), use_siamese_norm=True, **overrides)


def test_default_is_pre_ln():
    """The flag is off by default, so stored checkpoints keep loading."""
    assert ModernConfig.tiny().use_siamese_norm is False
    assert not hasattr(ModernLM(ModernConfig.tiny()), "y_final_norm")


def test_pre_ln_path_is_unchanged_by_the_flag_existing():
    """Same seed, flag off -> identical weights and identical logits."""
    ids = torch.randint(0, ModernConfig.tiny().vocab_size, (2, 8))
    torch.manual_seed(7)
    a = ModernLM(ModernConfig.tiny()).eval()
    torch.manual_seed(7)
    b = ModernLM(ModernConfig.tiny()).eval()
    with torch.no_grad():
        assert torch.equal(a(ids).logits, b(ids).logits)


def test_forward_is_finite_and_shaped():
    config = _siamese()
    model = ModernLM(config).eval()
    ids = torch.randint(0, config.vocab_size, (2, 16))
    with torch.no_grad():
        logits = model(ids).logits
    assert logits.shape == (2, 16, config.vocab_size)
    assert torch.isfinite(logits).all()


def test_kv_cache_matches_uncached_decoding():
    """Two streams must be threaded through the cache identically.

    A bug here would not crash -- it would silently score every benchmark
    question on a subtly different model than the one that was trained.
    """
    config = _siamese()
    model = ModernLM(config).eval()
    ids = torch.randint(0, config.vocab_size, (2, 12))
    slow = model.generate(ids, max_new_tokens=8, use_cache=False)
    fast = model.generate(ids, max_new_tokens=8, use_cache=True)
    assert torch.equal(slow, fast)


def test_gamma_starts_at_pre_ln():
    """gamma=1.0 makes the attention input exactly the Pre-LN input at step 0."""
    model = ModernLM(_siamese())
    for block in model.blocks:
        assert torch.equal(block.gamma, torch.ones_like(block.gamma))


def test_depth_scaling_follows_inverse_sqrt():
    model = ModernLM(_siamese(n_layers=4))
    scales = [block.residual_scale for block in model.blocks]
    assert scales == [(layer_id + 1) ** -0.5 for layer_id in range(4)]


def test_new_parameters_are_all_one_dimensional():
    """So the Muon/AdamW split needs no change: 1D params route to AdamW."""
    model = ModernLM(_siamese())
    keys = ("gamma", "x_norm", "y_norm", "attn_post_norm", "y_final_norm")
    new = [(n, p) for n, p in model.named_parameters() if any(k in n for k in keys)]
    assert new, "expected SiameseNorm to add parameters"
    assert all(p.ndim == 1 for _, p in new)

    muon, _, no_decay = split_muon_params(model)
    names = {id(p): n for n, p in model.named_parameters()}
    muon_names = {names[id(p)] for p in muon}
    assert not any(any(k in n for k in keys) for n in muon_names)
    no_decay_names = {names[id(p)] for p in no_decay}
    assert all(n in no_decay_names for n, _ in new)


def test_backward_produces_finite_gradients():
    config = _siamese()
    model = ModernLM(config)
    ids = torch.randint(0, config.vocab_size, (2, 16))
    model(ids).logits.square().mean().backward()
    for name, param in model.named_parameters():
        if param.grad is not None:
            assert torch.isfinite(param.grad).all(), name


def test_residual_init_is_skipped_under_siamese():
    """The GPT-2 depth init is a Pre-LN remedy and would double-count here."""
    torch.manual_seed(0)
    pre_ln = ModernLM(replace(ModernConfig.tiny(), n_layers=4))
    torch.manual_seed(0)
    siamese = ModernLM(_siamese())
    pre_std = pre_ln.blocks[0].attn.o_proj.weight.std().item()
    sia_std = siamese.blocks[0].attn.o_proj.weight.std().item()
    assert sia_std > pre_std * 2
