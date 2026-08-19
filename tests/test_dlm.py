"""Tests for the AR-to-dLM conversion pieces (Efficient-DLM, arXiv:2512.14067).

Two of these encode bugs that were live during implementation and would be easy
to reintroduce: the masking tilt ran backwards (early tokens masked most), and
the Gumbel term produced -inf keys that sorted to NaN and selected nothing.
"""
from __future__ import annotations

import torch

from modern_lm import dlm
from modern_lm.config import ModernConfig
from modern_lm.model import ModernLM


def _model(seq_len: int = 64) -> ModernLM:
    cfg = ModernConfig(vocab_size=256, max_seq_len=seq_len, dim=64, n_layers=2,
                       n_heads=4, n_kv_heads=4, ffn_dim=128)
    return ModernLM(cfg).eval()


def test_block_mask_is_bidirectional_within_and_causal_across():
    m = dlm.block_causal_mask(8, 4, torch.device("cpu"))
    # within block 0: every pair visible both ways
    assert m[0, 3] and m[3, 0]
    # across blocks: later sees earlier, never the reverse
    assert m[4, 0] and not m[0, 4]


def test_ar_path_is_unchanged_when_no_mask_given():
    """The dLM plumbing must not perturb ordinary autoregressive training."""
    torch.manual_seed(0)
    model = _model()
    ids = torch.randint(0, 256, (2, 16))
    assert torch.equal(model(ids).logits, model(ids, attn_mask=None).logits)


def test_block_mask_blocks_information_from_later_blocks():
    torch.manual_seed(0)
    model = _model()
    ids = torch.randint(0, 256, (2, 16))
    mask = dlm.block_causal_mask(16, 4, ids.device)
    a = model(ids, attn_mask=mask).logits
    edited = ids.clone()
    edited[:, 12:] = (edited[:, 12:] + 7) % 256
    b = model(edited, attn_mask=mask).logits
    assert torch.allclose(a[:, :12], b[:, :12], atol=1e-5)
    assert not torch.allclose(a[:, 12:], b[:, 12:], atol=1e-5)


def test_masking_tilts_toward_late_positions():
    """Regression: the tilt once ran backwards, masking position 0 the most."""
    torch.manual_seed(0)
    block = 16
    beta = dlm.beta_from_lambda(0.1, block)
    ids = torch.randint(2, 16384, (4096, 512))
    _, mask, _ = dlm.corrupt(ids, block, beta)
    rate = mask.view(-1, block).float().mean(0)
    assert rate[-1] > rate[0]
    assert bool((rate[1:] >= rate[:-1] - 0.02).all()), "should be non-decreasing"


def test_uniform_masking_when_lambda_disabled():
    torch.manual_seed(0)
    block = 16
    ids = torch.randint(2, 16384, (4096, 512))
    _, mask, _ = dlm.corrupt(ids, block, 0.0)
    rate = mask.view(-1, block).float().mean(0)
    assert float(rate.max() - rate.min()) < 0.02


def test_corrupt_selects_exactly_k_per_block():
    """Regression: -inf Gumbel keys sorted to NaN and selected zero tokens."""
    torch.manual_seed(0)
    block = 16
    beta = dlm.beta_from_lambda(0.1, block)
    ids = torch.randint(2, 16384, (256, 512))
    noisy, mask, t = dlm.corrupt(ids, block, beta)
    expected = torch.ceil(t * block).long().clamp(1, block)
    per_block = mask.view(ids.shape[0], -1, block).sum(-1)
    assert bool((per_block == expected.unsqueeze(1)).all())
    assert mask.any(), "some tokens must be masked"
    assert bool((noisy[mask] == dlm.MASK_TOKEN_ID).all())
    assert bool((noisy[~mask] == ids[~mask]).all())


def test_mdm_loss_only_scores_masked_positions():
    torch.manual_seed(0)
    logits = torch.randn(4, 32, 128, requires_grad=True)
    targets = torch.randint(0, 128, (4, 32))
    mask = torch.zeros(4, 32, dtype=torch.bool)
    mask[:, ::2] = True
    dlm.mdm_loss(logits, targets, mask, torch.rand(4).clamp_min(0.1)).backward()
    assert float(logits.grad[~mask].abs().sum()) == 0.0
    assert float(logits.grad[mask].abs().sum()) > 0.0


def test_mdm_loss_inverse_t_weighting():
    torch.manual_seed(0)
    logits = torch.randn(4, 32, 128)
    targets = torch.randint(0, 128, (4, 32))
    mask = torch.ones(4, 32, dtype=torch.bool)
    low = dlm.mdm_loss(logits, targets, mask, torch.full((4,), 0.1))
    high = dlm.mdm_loss(logits, targets, mask, torch.full((4,), 0.9))
    assert float(low) > float(high)


def test_generate_fills_every_masked_position():
    torch.manual_seed(0)
    model = _model()
    prompt = torch.randint(2, 256, (2, 8))
    out = dlm.generate(model, prompt, max_new_tokens=16, block_size=8,
                       confidence=0.999)
    assert out.shape == (2, 24)
    assert torch.equal(out[:, :8], prompt)
    assert bool((out[:, 8:] != dlm.MASK_TOKEN_ID).all())
