"""Learning-rate schedules: the cosine default and the WSD alternative.

The cosine cases are regression tests. Every run before 2026-08-15 used that
path and several are still resumable, so its shape must not move.
"""
from __future__ import annotations

import math

from modern_lm.train import TrainSettings, learning_rate_at

TOTAL = 1000
WARMUP = 100


def cosine(**kw) -> TrainSettings:
    return TrainSettings(learning_rate=3e-4, warmup_updates=WARMUP, **kw)


def wsd(**kw) -> TrainSettings:
    return TrainSettings(learning_rate=3e-4, warmup_updates=WARMUP,
                         lr_schedule="wsd", **kw)


# --- warmup is shared by both schedules ----------------------------------

def test_warmup_is_linear_and_shared():
    for settings in (cosine(), wsd()):
        assert learning_rate_at(0, settings, TOTAL) == 3e-4 / WARMUP
        mid = learning_rate_at(WARMUP // 2 - 1, settings, TOTAL)
        assert math.isclose(mid, 3e-4 * 0.5, rel_tol=1e-9)
        # Last warmup update reaches the peak exactly.
        assert math.isclose(learning_rate_at(WARMUP - 1, settings, TOTAL), 3e-4)


# --- cosine: unchanged from every prior run -------------------------------

def test_cosine_starts_at_peak_and_ends_at_floor():
    s = cosine()
    assert math.isclose(learning_rate_at(WARMUP, s, TOTAL), 3e-4, rel_tol=1e-6)
    assert math.isclose(learning_rate_at(TOTAL, s, TOTAL), 3e-5, rel_tol=1e-6)


def test_cosine_is_monotonically_decreasing_after_warmup():
    s = cosine()
    values = [learning_rate_at(u, s, TOTAL) for u in range(WARMUP, TOTAL + 1)]
    assert all(a >= b for a, b in zip(values, values[1:]))


def test_cosine_midpoint_is_halfway_between_peak_and_floor():
    s = cosine()
    half = learning_rate_at(WARMUP + (TOTAL - WARMUP) // 2, s, TOTAL)
    assert math.isclose(half, (3e-4 + 3e-5) / 2, rel_tol=1e-3)


# --- wsd ------------------------------------------------------------------

def test_wsd_holds_peak_through_the_stable_phase():
    s = wsd(wsd_decay_fraction=0.2)
    # Stable spans progress 0.0 .. 0.8 of the post-warmup updates.
    span = TOTAL - WARMUP
    for progress in (0.0, 0.25, 0.5, 0.79):
        u = WARMUP + int(progress * span)
        assert learning_rate_at(u, s, TOTAL) == 3e-4


def test_wsd_decays_linearly_over_the_final_fraction():
    s = wsd(wsd_decay_fraction=0.2, min_lr_ratio=0.0)
    span = TOTAL - WARMUP
    # Halfway through a decay-to-zero tail is half the peak.
    u = WARMUP + int(0.9 * span)
    assert math.isclose(learning_rate_at(u, s, TOTAL), 1.5e-4, rel_tol=1e-2)
    assert math.isclose(learning_rate_at(TOTAL, s, TOTAL), 0.0, abs_tol=1e-12)


def test_wsd_respects_a_nonzero_floor():
    s = wsd(wsd_decay_fraction=0.2, min_lr_ratio=0.1)
    assert math.isclose(learning_rate_at(TOTAL, s, TOTAL), 3e-5, rel_tol=1e-6)


def test_wsd_spends_more_of_the_run_at_high_lr_than_cosine():
    """The property the experiment is actually testing."""
    c, w = cosine(), wsd(wsd_decay_fraction=0.2)
    area_c = sum(learning_rate_at(u, c, TOTAL) for u in range(WARMUP, TOTAL))
    area_w = sum(learning_rate_at(u, w, TOTAL) for u in range(WARMUP, TOTAL))
    assert area_w > area_c


def test_wsd_zero_decay_fraction_never_decays():
    s = wsd(wsd_decay_fraction=0.0)
    assert learning_rate_at(TOTAL, s, TOTAL) == 3e-4


def test_schedules_never_exceed_peak_or_fall_below_floor():
    """Warmup deliberately starts below the floor; the floor binds after it."""
    for s in (cosine(), wsd(min_lr_ratio=0.1), wsd(min_lr_ratio=0.0)):
        floor = 3e-4 * s.min_lr_ratio
        for u in range(0, TOTAL + 50):
            lr = learning_rate_at(u, s, TOTAL)
            assert -1e-12 <= lr <= 3e-4 + 1e-12
            if u >= WARMUP:
                assert lr >= floor - 1e-12


def test_past_the_end_stays_clamped_at_the_floor():
    """`total_updates` is planned, not enforced; a run can overshoot it."""
    for s in (cosine(), wsd(min_lr_ratio=0.1)):
        assert math.isclose(learning_rate_at(TOTAL * 2, s, TOTAL), 3e-5,
                            rel_tol=1e-6)
