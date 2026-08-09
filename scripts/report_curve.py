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


ARITHMETIC = re.compile(r"\d\s*[-+x*/×]\s*\d")


def shows_work_rate(jsonl: Path) -> dict[str, float]:
    """Fraction of completions per benchmark that contain an actual calculation.

    At the 2B baseline the model emits the word "Explanation" on ~70% of word
    problems while showing arithmetic on only ~14-16% -- it learned the shape of
    a worked solution without the substance. This rate should move before
    accuracy does, so it is the earlier signal that continued pretraining is
    teaching multi-step composition rather than more formatting.
    """
    if not jsonl.exists():
        return {}
    totals: dict[str, int] = {}
    working: dict[str, int] = {}
    with jsonl.open() as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            name = record.get("benchmark")
            if name is None:
                continue
            totals[name] = totals.get(name, 0) + 1
            if ARITHMETIC.search(str(record.get("completion") or "")):
                working[name] = working.get(name, 0) + 1
    return {k: working.get(k, 0) / v for k, v in totals.items() if v}


def load(directory: Path) -> list[tuple[int, dict]]:
    rows = []
    for path in sorted(directory.glob("*.summary.json")):
        match = re.search(r"(\d+)", path.name)
        if not match:
            continue
        summary = json.loads(path.read_text())
        summary["_shows_work"] = shows_work_rate(
            path.with_name(path.name.replace(".summary.json", ".jsonl")))
        rows.append((int(match.group(1)), summary))
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
            work = summary.get("_shows_work") or {}
            word_problems = [work[k] for k in ("asdiv", "svamp", "gsm8k") if k in work]
            suffix += (f"  {sum(word_problems) / len(word_problems) * 100:5.1f}%"
                       if word_problems else "      -")
            print(f"{tokens / 1e6:8.0f}M  {accuracy * 100:6.2f}% +/-{half_width * 100:5.2f}  "
                  + "  ".join(cells) + suffix)
        print("* low-N benchmark: algebra=100, arithmetic=300 questions; treat single-point moves as noise.")
        print("  last column: fraction of word-problem completions showing real arithmetic")
        print("  (2B baseline ~14-16%); expected to move before accuracy does.")

        first, last = rows[0][1]["overall"], rows[-1][1]["overall"]
        delta = (last["accuracy"] - first["accuracy"]) * 100
        se = math.sqrt(
            first["accuracy"] * (1 - first["accuracy"]) / first["total"]
            + last["accuracy"] * (1 - last["accuracy"]) / last["total"])
        print(f"  drift {rows[0][0] / 1e6:.0f}M -> {rows[-1][0] / 1e6:.0f}M: "
              f"{delta:+.2f}pp (95% CI +/-{1.96 * se * 100:.2f}pp)")


if __name__ == "__main__":
    main()
