"""Cached decoding must be bit-identical to the uncached path.

The cache is an evaluation speedup, so the moment it changes a single token it
stops being a speedup and starts being a different model. RoPE makes this easy
to get wrong: a token decoded one at a time has to be rotated at its absolute
position, and attention over a cached prefix has to run with the causal mask
off, since a single query row legitimately attends to every stored key.
"""
import torch

from modern_lm.config import ModernConfig
from modern_lm.model import ModernLM


def _model(**overrides):
    torch.manual_seed(0)
    config = ModernConfig(**{"vocab_size": 256, "max_seq_len": 64, "dim": 64,
                             "n_layers": 3, "n_heads": 4, "n_kv_heads": 4,
                             "ffn_dim": 128, **overrides})
    return ModernLM(config).eval()


def test_cached_decode_matches_uncached():
    model = _model()
    ids = torch.randint(0, 256, (3, 7))
    assert torch.equal(model.generate(ids.clone(), max_new_tokens=20),
                       model.generate(ids.clone(), max_new_tokens=20, use_cache=True))


def test_cached_decode_matches_under_gqa():
    # n_rep > 1: K/V are cached before the repeat, so the expansion has to happen
    # after the cache read or the heads misalign.
    model = _model(n_heads=8, n_kv_heads=2)
    ids = torch.randint(0, 256, (2, 5))
    assert torch.equal(model.generate(ids.clone(), max_new_tokens=15),
                       model.generate(ids.clone(), max_new_tokens=15, use_cache=True))


def test_cached_decode_matches_with_eos():
    model = _model()
    ids = torch.randint(0, 256, (2, 5))
    assert torch.equal(model.generate(ids.clone(), max_new_tokens=15, eos_token_id=7),
                       model.generate(ids.clone(), max_new_tokens=15, eos_token_id=7,
                                      use_cache=True))


def test_cache_falls_back_past_max_seq_len():
    # Once the sequence fills the context the cache cannot be extended; decoding
    # has to drop to the sliding-window path rather than index past the RoPE table.
    model = _model(max_seq_len=48)
    ids = torch.randint(0, 256, (2, 40))
    assert torch.equal(model.generate(ids.clone(), max_new_tokens=20),
                       model.generate(ids.clone(), max_new_tokens=20, use_cache=True))


def test_rope_offset_places_a_token_at_its_real_position():
    from modern_lm.layers import apply_rope, build_rope_cache

    cos, sin = build_rope_cache(16, 8, 10000.0, torch.device("cpu"))
    x = torch.randn(1, 2, 5, 8)
    whole = apply_rope(x, cos, sin)
    # Rotating row 4 alone with offset=4 must equal its row in the batch rotation.
    single = apply_rope(x[:, :, 4:5], cos, sin, offset=4)
    assert torch.allclose(whole[:, :, 4:5], single, atol=1e-6)
