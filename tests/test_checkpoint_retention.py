"""Checkpoint retention: permanent milestones plus a rotating recent window.

The failure this guards against is a rolling window quietly deleting the
scientific record -- particularly the WSD fork point, which is the only
checkpoint an extended run can legitimately branch from.
"""
from __future__ import annotations

from modern_lm.train import (classify_checkpoints, milestone_tokens,
                             prune_checkpoints)


def _touch(run_dir, tokens):
    path = run_dir / f"checkpoint-{tokens:012d}.pt"
    path.write_bytes(b"x")
    return path


def test_milestones_are_percentages_of_the_full_run():
    marks = milestone_tokens(1_000_000, [10, 20, 50, 100])
    assert marks == [100_000, 200_000, 500_000, 1_000_000]


def test_decay_start_is_its_own_milestone():
    """The fork point rarely lands on a round percentage."""
    marks = milestone_tokens(1_000_000, [10, 20], decay_start_tokens=837_194)
    assert 837_194 in marks
    assert marks == sorted(marks)


def test_duplicate_milestones_collapse():
    marks = milestone_tokens(1_000_000, [80, 80], decay_start_tokens=800_000)
    assert marks.count(800_000) == 1


def test_classify_tolerates_checkpoints_landing_past_the_mark(tmp_path):
    """Checkpoints land on the first step to CROSS a threshold, never on it."""
    _touch(tmp_path, 800_000_100)
    keep, recent = classify_checkpoints(tmp_path, [800_000_000], tolerance=1_000)
    assert len(keep) == 1 and not recent

    keep, recent = classify_checkpoints(tmp_path, [800_000_000], tolerance=10)
    assert not keep and len(recent) == 1


def test_coarse_checkpoint_interval_still_protects_every_milestone(tmp_path):
    """A one-update tolerance protects nothing when checkpoints are sparse.

    The 300M run wrote checkpoints every 30M tokens against 10% marks of a
    5,925,345,280-token budget. No mark lands on that grid -- the nearest
    checkpoint is up to half an interval away -- so with a one-update tolerance
    every milestone went unprotected and rotated out with the recent window.
    Half a checkpoint interval is the smallest tolerance that always captures
    the closest checkpoint, and it captures exactly one per milestone.
    """
    total, interval = 5_925_345_280, 30_000_000
    marks = milestone_tokens(total, [10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    grid = [3_600_000_000 + step * interval for step in range(78)]
    for tokens in grid:
        _touch(tmp_path, tokens)

    one_update = 16 * 4 * 512
    keep, _ = classify_checkpoints(tmp_path, marks, tolerance=one_update)
    assert not keep, "the sparse grid should miss every mark at one-update tolerance"

    keep, _ = classify_checkpoints(tmp_path, marks, tolerance=interval // 2)
    reachable = [m for m in marks if grid[0] - interval // 2 <= m <= grid[-1] + interval // 2]
    assert len(keep) == len(reachable)
    for mark in reachable:
        assert any(abs(int(p.stem.split("-")[-1]) - mark) <= interval // 2 for p in keep)


def test_prune_keeps_every_milestone_and_the_last_n_recent(tmp_path):
    milestones = [200, 400]
    for tokens in (100, 200, 300, 400, 500, 600, 700):
        _touch(tmp_path, tokens)

    prune_checkpoints(tmp_path, keep_last=2, protected=milestones, tolerance=0)

    survivors = sorted(int(p.stem.split("-")[-1]) for p in tmp_path.glob("checkpoint-*.pt"))
    # both milestones, plus the two newest non-milestones (600, 700)
    assert survivors == [200, 400, 600, 700]


def test_milestones_survive_an_aggressive_window(tmp_path):
    """keep_last=1 must not be able to destroy the record."""
    for tokens in range(100, 1100, 100):
        _touch(tmp_path, tokens)
    prune_checkpoints(tmp_path, keep_last=1, protected=[300, 900], tolerance=0)
    survivors = sorted(int(p.stem.split("-")[-1]) for p in tmp_path.glob("checkpoint-*.pt"))
    assert 300 in survivors and 900 in survivors
    assert survivors == [300, 900, 1000]


def test_no_protection_reproduces_the_old_behaviour(tmp_path):
    for tokens in (100, 200, 300):
        _touch(tmp_path, tokens)
    prune_checkpoints(tmp_path, keep_last=2)
    survivors = sorted(int(p.stem.split("-")[-1]) for p in tmp_path.glob("checkpoint-*.pt"))
    assert survivors == [200, 300]


def test_keep_last_zero_with_milestones_prunes_everything_else(tmp_path):
    for tokens in (100, 200, 300):
        _touch(tmp_path, tokens)
    prune_checkpoints(tmp_path, keep_last=0, protected=[200], tolerance=0)
    survivors = sorted(int(p.stem.split("-")[-1]) for p in tmp_path.glob("checkpoint-*.pt"))
    assert survivors == [200]


def test_disabled_retention_deletes_nothing(tmp_path):
    for tokens in (100, 200, 300):
        _touch(tmp_path, tokens)
    prune_checkpoints(tmp_path, keep_last=0)
    assert len(list(tmp_path.glob("checkpoint-*.pt"))) == 3
