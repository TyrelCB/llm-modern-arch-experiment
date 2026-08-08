"""Wall-clock time-to-loss comparison between training runs.

Final loss at a fixed token budget answers "which optimizer is more sample
efficient". That is not the question that decides adoption: Muon buys fewer
steps but pays ~5% more wall-clock per step, so the two effects work against
each other. This script prices them together by asking, for each held-out loss
threshold, how long each run took to first reach it.

Reads the `evaluation` records from each run's train.jsonl, so its resolution is
whatever --checkpoint-tokens the run used. Sweep-resolution runs (one eval at
the end) cannot produce a curve; use a small --checkpoint-tokens for the
comparison runs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_evaluations(run_dir: Path) -> list[dict]:
    """Return the evaluation records from a run, oldest first."""
    log = run_dir / "train.jsonl"
    if not log.exists():
        raise FileNotFoundError(f"no train.jsonl in {run_dir}")
    records = []
    for line in log.read_text().splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("event") == "evaluation":
            records.append(record)
    # A resumed run appends to the same log, so a later pass can repeat earlier
    # token counts; keep the last record for each and re-sort.
    by_tokens = {r["tokens_seen"]: r for r in records}
    return [by_tokens[k] for k in sorted(by_tokens)]


def loss_of(record: dict) -> float:
    """Held-out loss, tolerating the two key spellings the trainers emit."""
    for key in ("main_loss", "loss"):
        if key in record:
            return float(record[key])
    raise KeyError(f"no loss field in evaluation record: {sorted(record)}")


def first_crossing(records: list[dict], threshold: float) -> dict | None:
    """First evaluation at or below `threshold`, linearly interpolated in time.

    Without interpolation the answer quantizes to the eval interval, which at
    coarse checkpointing is a large fraction of the very gap being measured.
    """
    previous = None
    for record in records:
        if loss_of(record) <= threshold:
            if previous is None:
                # Already below at the first eval: no crossing to interpolate.
                return {"seconds": record["elapsed_seconds"],
                        "tokens": record["tokens_seen"], "exact": False}
            span = loss_of(previous) - loss_of(record)
            frac = 1.0 if span <= 0 else (loss_of(previous) - threshold) / span
            return {
                "seconds": previous["elapsed_seconds"]
                + frac * (record["elapsed_seconds"] - previous["elapsed_seconds"]),
                "tokens": previous["tokens_seen"]
                + frac * (record["tokens_seen"] - previous["tokens_seen"]),
                "exact": True,
            }
        previous = record
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", type=Path,
                        help="run directories, each containing train.jsonl")
    parser.add_argument("--label", action="append", default=None,
                        help="display name per run, in the same order")
    parser.add_argument("--thresholds", type=float, nargs="+",
                        help="held-out losses to time; default spans the range "
                             "all runs actually reached")
    parser.add_argument("--baseline", type=int, default=0,
                        help="index of the run used as the speedup denominator")
    parser.add_argument("--json", type=Path, help="also write results here")
    args = parser.parse_args()

    labels = args.label or [r.name for r in args.runs]
    if len(labels) != len(args.runs):
        raise SystemExit("--label count must match the number of runs")
    series = {label: load_evaluations(run) for label, run in zip(labels, args.runs)}

    for label, records in series.items():
        if not records:
            raise SystemExit(f"{label}: no evaluation records")
        if len(records) < 2:
            print(f"warning: {label} has only {len(records)} evaluation point(s); "
                  f"time-to-loss needs a finer --checkpoint-tokens")

    if args.thresholds:
        thresholds = sorted(args.thresholds, reverse=True)
    else:
        # Only thresholds every run reached are comparable.
        floor = max(min(loss_of(r) for r in recs) for recs in series.values())
        ceiling = min(max(loss_of(r) for r in recs) for recs in series.values())
        if floor >= ceiling:
            raise SystemExit("runs share no common loss range to compare")
        step = (ceiling - floor) / 5
        thresholds = [round(ceiling - i * step, 3) for i in range(6)]

    baseline = labels[args.baseline]
    width = max(len(l) for l in labels)
    print(f"\nWall-clock time to reach a held-out loss (baseline: {baseline})\n")
    header = f"{'loss':>7} | " + " | ".join(f"{l:>{max(width, 16)}}" for l in labels)
    print(header)
    print("-" * len(header))

    results = []
    for threshold in thresholds:
        crossings = {l: first_crossing(recs, threshold) for l, recs in series.items()}
        cells, row = [], {"threshold": threshold, "runs": {}}
        base = crossings[baseline]
        for label in labels:
            hit = crossings[label]
            if hit is None:
                cells.append(f"{'never':>{max(width, 16)}}")
                row["runs"][label] = None
                continue
            minutes = hit["seconds"] / 60
            if label == baseline or base is None:
                cells.append(f"{minutes:>10.1f} min  ")
            else:
                speedup = base["seconds"] / hit["seconds"]
                cells.append(f"{minutes:>10.1f} ({speedup:.2f}x)")
            row["runs"][label] = {"seconds": hit["seconds"], "tokens": hit["tokens"],
                                  "minutes": minutes}
        print(f"{threshold:>7.3f} | " + " | ".join(cells))
        results.append(row)

    print("\nFinal state")
    for label, records in series.items():
        last = records[-1]
        print(f"  {label:>{width}}: loss {loss_of(last):.4f} at "
              f"{last['tokens_seen']:,} tok in {last['elapsed_seconds']/60:.1f} min "
              f"({last['tokens_seen']/last['elapsed_seconds']:,.0f} tok/s)")

    if args.json:
        args.json.write_text(json.dumps(
            {"baseline": baseline, "thresholds": results,
             "final": {l: {"loss": loss_of(r[-1]), "tokens": r[-1]["tokens_seen"],
                           "seconds": r[-1]["elapsed_seconds"]}
                       for l, r in series.items()}}, indent=2) + "\n")
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
