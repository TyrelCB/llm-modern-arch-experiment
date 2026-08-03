"""Tests for the comparison builder.

The gate arithmetic decides the headline claim, so it is worth pinning against
hand-computed values rather than trusting it on the day the run finishes.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from modern_lm import compare


def write_log(run_dir: Path, rows: list[dict]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "train.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n")


def test_modern_trajectory_reads_only_evaluation_events(tmp_path):
    write_log(tmp_path, [
        {"event": "update", "tokens_seen": 100, "main_loss": 9.0},
        {"event": "evaluation", "tokens_seen": 1000, "main_loss": 5.0,
         "elapsed_seconds": 10.0},
        {"event": "evaluation", "tokens_seen": 2000, "main_loss": 4.0,
         "elapsed_seconds": 20.0},
    ])
    rows = compare.modern_trajectory(tmp_path)
    assert [r["tokens_seen"] for r in rows] == [1000, 2000]
    assert rows[-1]["heldout_main_loss"] == 4.0


def test_gates_find_first_crossing_of_reference_loss(tmp_path, monkeypatch):
    """Time-to-quality uses the FIRST milestone at or below the reference loss."""
    reference_loss = 2.551358178257942
    write_log(tmp_path, [
        {"event": "evaluation", "tokens_seen": 100_000_000,
         "main_loss": 2.9, "elapsed_seconds": 1000.0},
        {"event": "evaluation", "tokens_seen": 150_000_000,
         "main_loss": 2.5, "elapsed_seconds": 1500.0},
        {"event": "evaluation", "tokens_seen": 250_000_000,
         "main_loss": 2.3, "elapsed_seconds": 2500.0},
    ])
    monkeypatch.setattr(compare, "reference_trajectory",
                        lambda: [{"tokens_seen": 250_000_000,
                                  "heldout_main_loss": reference_loss}])
    monkeypatch.setattr(compare, "load_summary", lambda path: None)

    result = compare.build(tmp_path, tmp_path / "comparison.json")
    gates = result["gates"]

    # Crossing is the 150M milestone, not the final one.
    assert gates["modern_tokens_to_reference_final_loss"] == 150_000_000
    assert gates["modern_seconds_to_reference_final_loss"] == 1500.0
    assert gates["loss_delta"] == pytest.approx(2.3 - reference_loss)
    # Same-token speedup: reference would need 250M/9720.3 s.
    expected = (250_000_000 / compare.REFERENCE_TOKENS_PER_SECOND_BEST) / 2500.0
    assert gates["speedup_vs_reference_best"] == pytest.approx(expected)
    assert (tmp_path / "comparison.json").exists()


def test_gates_report_no_crossing_when_loss_never_reached(tmp_path, monkeypatch):
    write_log(tmp_path, [
        {"event": "evaluation", "tokens_seen": 250_000_000,
         "main_loss": 3.4, "elapsed_seconds": 2500.0},
    ])
    monkeypatch.setattr(compare, "reference_trajectory",
                        lambda: [{"tokens_seen": 250_000_000,
                                  "heldout_main_loss": 2.55}])
    monkeypatch.setattr(compare, "load_summary", lambda path: None)
    gates = compare.build(tmp_path, tmp_path / "c.json")["gates"]
    assert gates["modern_seconds_to_reference_final_loss"] is None
    assert "time_to_quality_speedup_vs_best" not in gates
    assert gates["loss_delta"] > 0


def test_reference_trajectory_reads_committed_checkpoints():
    """Guards against the reference run's artifacts moving or being renamed."""
    if not compare.REFERENCE_RUN.exists():
        pytest.skip("reference run artifacts not present")
    rows = compare.reference_trajectory()
    assert rows, "no reference checkpoints found"
    assert rows[-1]["tokens_seen"] == 250_000_000
    assert rows[-1]["heldout_main_loss"] == pytest.approx(2.551358178257942)


def test_reference_scorer_imports_without_pyarrow():
    """The scorer must be importable outside the reference repo's venv.

    `deepseek_v4.evaluation` imports pyarrow at module level for the benchmark
    preparation path, which scoring does not use. Importing the reference's own
    functions (rather than copying them) is what keeps the two runs' accuracy
    numbers comparable, so this import must not be fragile.
    """
    from modern_lm.evaluate_benchmarks import extract_number, numeric_equal

    assert extract_number("The answer is 42.") == "42"
    assert numeric_equal("42", "42")
    assert not numeric_equal("41", "42")
