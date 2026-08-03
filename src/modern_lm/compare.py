"""Build the head-to-head comparison against the DeepSeek-V4 reference run.

Reads both runs' recorded artifacts and emits `runs/comparison.json` plus a
markdown table. Nothing is recomputed on the GPU: the reference's numbers are
taken from its own committed checkpoint metadata and evaluation summary.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

from .data import DEEPSEEK_REPO

REFERENCE_RUN = DEEPSEEK_REPO / "runs" / "finemath-145m"
# Measured on an idle GB10 in this project's own benchmark, after the
# reference's landed loss-path optimizations (runs/finemath-145m-pretrain-
# microbatch/results-optimized-64.json). The reference's production run used
# microbatch 16x4, which its own sweep measured at 8,600 tok/s.
REFERENCE_TOKENS_PER_SECOND_BEST = 9720.3
REFERENCE_TOKENS_PER_SECOND_AS_RUN = 8600.0
REFERENCE_STORED_PARAMS = 144_669_412
REFERENCE_ACTIVE_PARAMS = 45_578_980


def reference_trajectory() -> list[dict]:
    rows = []
    for path in sorted(glob.glob(str(REFERENCE_RUN / "checkpoint-*.json"))):
        state = json.loads(Path(path).read_text())["state"]
        rows.append({"tokens_seen": state["tokens_seen"],
                     "heldout_main_loss": state.get("best_heldout_main_loss")})
    return rows


def modern_trajectory(run_dir: Path) -> list[dict]:
    rows = []
    log = run_dir / "train.jsonl"
    if not log.exists():
        return rows
    for line in log.read_text().splitlines():
        record = json.loads(line)
        if record.get("event") == "evaluation":
            rows.append({"tokens_seen": record["tokens_seen"],
                         "heldout_main_loss": record["main_loss"],
                         "elapsed_seconds": record["elapsed_seconds"]})
    return rows


def load_summary(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def build(run_dir: Path, output: Path) -> dict:
    modern = modern_trajectory(run_dir)
    reference = reference_trajectory()
    modern_final = modern[-1] if modern else None
    reference_final = reference[-1] if reference else None

    modern_eval = load_summary(run_dir / "evaluation.summary.json")
    reference_eval = load_summary(REFERENCE_RUN / "evaluation.summary.json")

    comparison: dict = {
        "reference": {
            "name": "DeepSeek-V4 145M (MoE)",
            "stored_parameters": REFERENCE_STORED_PARAMS,
            "active_parameters": REFERENCE_ACTIVE_PARAMS,
            "final_heldout_main_loss": reference_final["heldout_main_loss"] if reference_final else None,
            "tokens_per_second_best_measured": REFERENCE_TOKENS_PER_SECOND_BEST,
            "tokens_per_second_as_run": REFERENCE_TOKENS_PER_SECOND_AS_RUN,
            "trajectory": reference,
            "benchmarks": reference_eval,
        },
        "modern": {
            "name": "ModernLM 145M (dense)",
            "trajectory": modern,
            "benchmarks": modern_eval,
        },
    }

    if modern_final:
        tokens = modern_final["tokens_seen"]
        seconds = modern_final["elapsed_seconds"]
        comparison["modern"].update({
            "final_heldout_main_loss": modern_final["heldout_main_loss"],
            "training_seconds": seconds,
            "tokens_per_second": tokens / seconds,
        })
        if reference_final:
            ref_loss = reference_final["heldout_main_loss"]
            ref_seconds_best = tokens / REFERENCE_TOKENS_PER_SECOND_BEST
            ref_seconds_as_run = tokens / REFERENCE_TOKENS_PER_SECOND_AS_RUN
            # Gate 1: wall clock at which ModernLM first matched the
            # reference's *final* loss.
            crossing = next((row for row in modern
                             if row["heldout_main_loss"] <= ref_loss), None)
            comparison["gates"] = {
                "reference_final_loss": ref_loss,
                "modern_final_loss": modern_final["heldout_main_loss"],
                "loss_delta": modern_final["heldout_main_loss"] - ref_loss,
                "loss_relative_percent": 100 * (modern_final["heldout_main_loss"] - ref_loss) / ref_loss,
                "modern_seconds_to_reference_final_loss": crossing["elapsed_seconds"] if crossing else None,
                "modern_tokens_to_reference_final_loss": crossing["tokens_seen"] if crossing else None,
                "reference_seconds_for_same_tokens_best": ref_seconds_best,
                "reference_seconds_for_same_tokens_as_run": ref_seconds_as_run,
                "modern_seconds_for_same_tokens": seconds,
                "speedup_vs_reference_best": ref_seconds_best / seconds,
                "speedup_vs_reference_as_run": ref_seconds_as_run / seconds,
            }
            if crossing:
                comparison["gates"]["time_to_quality_speedup_vs_best"] = (
                    ref_seconds_best / crossing["elapsed_seconds"])
                comparison["gates"]["time_to_quality_speedup_vs_as_run"] = (
                    ref_seconds_as_run / crossing["elapsed_seconds"])

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(comparison, indent=2) + "\n")
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("runs/modern-145m"))
    parser.add_argument("--output", type=Path, default=Path("runs/comparison.json"))
    args = parser.parse_args()
    comparison = build(args.run_dir, args.output)
    print(json.dumps(comparison.get("gates", {}), indent=2))


if __name__ == "__main__":
    main()
