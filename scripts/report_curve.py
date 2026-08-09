#!/usr/bin/env python3
"""Print the capability curve across checkpoint evaluations.

Reads the *.summary.json files a curve sweep writes and reports accuracy per
benchmark against tokens seen, with a binomial 95% CI on the overall figure.

The CI matters here: the per-checkpoint scatter on these benchmarks is a
sizeable fraction of the total drift across a whole run, so a bare accuracy
column invites reading noise as progress. algebra is only 100 questions -- one
question is a full percentage point -- so it is flagged as low-N.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

BENCHMARKS = ("asdiv", "svamp", "gsm8k", "algebra", "arithmetic")
LOW_N = {"algebra", "arithmetic"}


def load(directory: Path) -> list[tuple[int, dict]]:
    rows = []
    for path in sorted(directory.glob("*.summary.json")):
        match = re.search(r"(\d+)", path.name)
        if not match:
            continue
        rows.append((int(match.group(1)), json.loads(path.read_text())))
    rows.sort(key=lambda item: item[0])
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", nargs="+", type=Path)
    args = parser.parse_args()

    for directory in args.directories:
        rows = load(directory)
        if not rows:
            print(f"{directory}: no summaries yet")
            continue
        print(f"\n=== {directory} ({len(rows)} checkpoints) ===")
        header = f"{'tokens':>9}  {'overall':>16}  " + "  ".join(f"{b:>10}" for b in BENCHMARKS) + "  numeric"
        print(header)
        for tokens, summary in rows:
            overall = summary["overall"]
            accuracy = overall["accuracy"]
            total = overall["total"]
            half_width = 1.96 * math.sqrt(accuracy * (1 - accuracy) / total) if total else 0.0
            cells = []
            for name in BENCHMARKS:
                entry = summary.get(name)
                cells.append("       n/a" if entry is None
                             else f"{entry['accuracy'] * 100:9.2f}{'*' if name in LOW_N else ' '}")
            rate = summary.get("numeric_completion_rate")
            suffix = f"  {rate * 100:5.1f}%" if rate is not None else ""
            print(f"{tokens / 1e6:8.0f}M  {accuracy * 100:6.2f}% +/-{half_width * 100:5.2f}  "
                  + "  ".join(cells) + suffix)
        print("* low-N benchmark: algebra=100, arithmetic=300 questions; treat single-point moves as noise.")

        first, last = rows[0][1]["overall"], rows[-1][1]["overall"]
        delta = (last["accuracy"] - first["accuracy"]) * 100
        se = math.sqrt(
            first["accuracy"] * (1 - first["accuracy"]) / first["total"]
            + last["accuracy"] * (1 - last["accuracy"]) / last["total"])
        print(f"  drift {rows[0][0] / 1e6:.0f}M -> {rows[-1][0] / 1e6:.0f}M: "
              f"{delta:+.2f}pp (95% CI +/-{1.96 * se * 100:.2f}pp)")


if __name__ == "__main__":
    main()
