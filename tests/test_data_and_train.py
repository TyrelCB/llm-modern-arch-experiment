"""Tests for data-stream parity with the reference and training plumbing.

The parity test is the important one: if the two trainers do not walk the packed
corpus in the same order, the held-out loss comparison silently stops being a
controlled architecture comparison.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from modern_lm.config import ModernConfig
from modern_lm.data import DEEPSEEK_REPO, PackedTokenStream, default_paths
from modern_lm.model import ModernLM
from modern_lm.train import (TrainSettings, compute_loss, learning_rate_at,
                             load_checkpoint, save_checkpoint, seed_everything)

DATA_AVAILABLE = default_paths()["train"].exists()
requires_data = pytest.mark.skipif(not DATA_AVAILABLE, reason="packed FineMath corpus not present")


@requires_data
def test_stream_block_order_matches_deepseek_reference():
    """Port fidelity: identical block indices for identical positions."""
    sys.path.insert(0, str(DEEPSEEK_REPO / "src"))
    from deepseek_v4.pretrain import PackedTokenStream as ReferenceStream

    path = default_paths()["train"]
    ours = PackedTokenStream(path, 512, seed=2026)
    theirs = ReferenceStream(path, 512, seed=2026)
    assert ours.blocks == theirs.blocks
    for position in [0, 1, 17, 1000, 50_000, ours.blocks - 1, ours.blocks + 5]:
        expected_block = theirs.block_index(position) if hasattr(theirs, "block_index") else None
        if expected_block is None:
            # Reference computes the block inline in sample(); recompute it the
            # same way to compare indices without loading tensors.
            epoch, within = divmod(position, theirs.blocks)
            multiplier, offset = theirs._permutation(epoch)
            expected_block = (multiplier * within + offset) % theirs.blocks
        assert ours.block_index(position) == expected_block, position


@requires_data
def test_stream_batch_matches_reference_tokens():
    sys.path.insert(0, str(DEEPSEEK_REPO / "src"))
    from deepseek_v4.pretrain import PackedTokenStream as ReferenceStream

    path = default_paths()["train"]
    device = torch.device("cpu")
    ours = PackedTokenStream(path, 512, seed=2026)
    theirs = ReferenceStream(path, 512, seed=2026)
    a = ours.batch(0, 4, device)
    b = theirs.batch(0, 4, device)
    assert torch.equal(a, b)
    assert a.shape == (4, 513) and a.dtype == torch.int64


@requires_data
def test_packed_tokens_are_in_vocab_range():
    config = ModernConfig.dense_145m()
    stream = PackedTokenStream(default_paths()["train"], 512, seed=2026)
    batch = stream.batch(0, 8, torch.device("cpu"))
    assert int(batch.min()) >= 0
    assert int(batch.max()) < config.vocab_size


def test_lr_schedule_warms_up_then_decays_to_floor():
    settings = TrainSettings(warmup_updates=100, learning_rate=3e-4, min_lr_ratio=0.1)
    total = 1000
    assert learning_rate_at(0, settings, total) == pytest.approx(3e-6)
    assert learning_rate_at(99, settings, total) == pytest.approx(3e-4)
    peak = learning_rate_at(100, settings, total)
    assert peak == pytest.approx(3e-4, rel=1e-3)
    end = learning_rate_at(total, settings, total)
    assert end == pytest.approx(3e-5, rel=1e-6)
    # Monotone non-increasing after warmup.
    values = [learning_rate_at(u, settings, total) for u in range(100, total + 1, 50)]
    assert all(b <= a + 1e-12 for a, b in zip(values, values[1:]))


def test_checkpoint_round_trip_restores_weights_optimizer_and_rng(tmp_path):
    config = ModernConfig.tiny()
    settings = TrainSettings(microbatch_size=2, gradient_accumulation=1)
    seed_everything(123)
    model = ModernLM(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    tokens = torch.randint(0, config.vocab_size, (2, 17))
    loss, _, _ = compute_loss(model, tokens, settings)
    loss.backward()
    optimizer.step()

    state = {"micro_step": 8, "optimizer_step": 1, "tokens_seen": 32,
             "elapsed_seconds": 1.5, "next_checkpoint_tokens": 100}
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, model, optimizer, config, settings, state)

    expected_next = random.random()
    expected_torch = torch.randn(3)

    fresh = ModernLM(config)
    fresh_opt = torch.optim.AdamW(fresh.parameters(), lr=1e-3)
    restored = load_checkpoint(path, fresh, fresh_opt)

    assert restored == state
    for a, b in zip(model.parameters(), fresh.parameters()):
        assert torch.equal(a, b)
    # RNG streams must resume identically, or resumed runs diverge silently.
    assert random.random() == expected_next
    assert torch.equal(torch.randn(3), expected_torch)
    assert fresh_opt.state_dict()["state"], "optimizer moments were not restored"
    assert path.with_suffix(".json").exists()


def test_compute_loss_counts_supervised_targets_only():
    config = ModernConfig.tiny()
    settings = TrainSettings()
    model = ModernLM(config)
    tokens = torch.randint(0, config.vocab_size, (3, 33))
    loss, components, n_targets = compute_loss(model, tokens, settings)
    # 33 packed tokens -> 32 next-token labels per row.
    assert n_targets == 3 * 32
    assert torch.isfinite(loss)
    assert "main" in components and "total" in components


def test_compute_loss_includes_mtp_and_aux_when_enabled():
    config = ModernConfig.tiny(use_mtp=True, use_moe=True)
    settings = TrainSettings()
    model = ModernLM(config)
    tokens = torch.randint(0, config.vocab_size, (2, 17))
    loss, components, _ = compute_loss(model, tokens, settings)
    assert "mtp" in components and "aux" in components
    assert components["total"] >= components["main"]
    assert torch.isfinite(loss)


def test_prune_checkpoints_keeps_last_n_and_preserves_metadata(tmp_path):
    """Long runs must not fill the disk, but the trajectory must stay recoverable."""
    from modern_lm.train import prune_checkpoints

    for tokens in [50_000_000, 100_000_000, 150_000_000, 200_000_000]:
        (tmp_path / f"checkpoint-{tokens:012d}.pt").write_bytes(b"x")
        (tmp_path / f"checkpoint-{tokens:012d}.json").write_text("{}")
    (tmp_path / "latest.pt").write_bytes(b"x")

    prune_checkpoints(tmp_path, keep_last=2)

    remaining = sorted(p.name for p in tmp_path.glob("checkpoint-*.pt"))
    assert remaining == ["checkpoint-000150000000.pt", "checkpoint-000200000000.pt"]
    # latest.pt is never a pruning candidate.
    assert (tmp_path / "latest.pt").exists()
    # Every milestone's metadata survives, so the loss curve is still complete.
    assert len(list(tmp_path.glob("checkpoint-*.json"))) == 4


def test_prune_checkpoints_disabled_by_default(tmp_path):
    from modern_lm.train import prune_checkpoints

    for tokens in [10_000_000, 20_000_000]:
        (tmp_path / f"checkpoint-{tokens:012d}.pt").write_bytes(b"x")
    prune_checkpoints(tmp_path, keep_last=0)
    assert len(list(tmp_path.glob("checkpoint-*.pt"))) == 2
