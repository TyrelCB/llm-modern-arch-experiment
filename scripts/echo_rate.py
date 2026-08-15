#!/usr/bin/env python3
"""Split benchmark hits into echoes and real answers.

A model that has learned the *format* of math text without learning to compute
scores above zero anyway, because it copies a number out of the question and
`extract_number` finds it. On the 5M run, 48 of 64 "correct" answers had the gold
value already sitting in the question text -- the headline 1.27% was really
0.32%, and that residue is small enough to be coincidence.

This is not the same failure as the truncation bug that `answer_segment` fixes.
That one scored numbers from a rambling continuation; this one scores a number
the model was handed. Both inflate accuracy, and neither is visible in the
summary.json.

An echo is a hit whose gold answer already appears as a number in the question.
That is deliberately conservative in one direction -- a model that genuinely
computes "9" for a question containing "9" is counted as an echo -- so the
non-echo count is a LOWER bound on real capability, and the echo count is an
UPPER bound on how much is fake. When non-echo accuracy is at noise, as it is
for every model below ~50M body params so far, the distinction does not matter:
both readings say the same thing.

Usage:
    scripts/echo_rate.py runs/eval-5m-100x.jsonl runs/eval-10m-100x.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


def numbers_in(text: str) -> set[str]:
    """Every number in `text`, as written and normalized.

    Both forms are kept because the gold answer and the question often disagree
    on formatting -- "9" vs "9.0" vs "9.00" -- and a spelling mismatch would
    count a genuine echo as a real answer.
    """
    found = set()
    for token in NUMBER.findall(text or ""):
        found.add(token)
        try:
            value = float(token)
        except ValueError:
            continue
        found.add(f"{value:g}")
        if value.is_integer():
            found.add(str(int(value)))
    return found


def normalize(answer: str) -> set[str]:
    return numbers_in(answer) or {(answer or "").strip()}


def analyze(path: Path) -> dict:
    records = [json.loads(line) for line in path.open() if line.strip()]
    total = len(records)
    hits = [r for r in records if r.get("correct")]

    echoes, real = [], []
    for record in hits:
        gold = normalize(str(record.get("answer", "")))
        asked = numbers_in(record.get("question", ""))
        (echoes if gold & asked else real).append(record)

    by_benchmark: dict[str, dict[str, int]] = {}
    for record in records:
        name = record.get("benchmark", "?")
        row = by_benchmark.setdefault(name, {"total": 0, "hits": 0, "real": 0})
        row["total"] += 1
    for record in hits:
        by_benchmark[record.get("benchmark", "?")]["hits"] += 1
    for record in real:
        by_benchmark[record.get("benchmark", "?")]["real"] += 1

    return {"path": path, "total": total, "hits": len(hits),
            "echoes": len(echoes), "real": len(real),
            "by_benchmark": by_benchmark, "examples": real[:3]}


def wilson(correct: int, total: int) -> tuple[float, float]:
    """Wilson 95% interval -- stays in [0,1] at the tiny rates this script sees."""
    if not total:
        return (0.0, 0.0)
    z, p, n = 1.96, correct / total, total
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    spread = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return (max(0.0, center - spread), min(1.0, center + spread))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("evals", nargs="+", type=Path, help="eval .jsonl files")
    parser.add_argument("--examples", action="store_true",
                        help="print sample non-echo hits, to eyeball whether they are real")
    args = parser.parse_args()

    reports = [analyze(path) for path in args.evals]

    print(f"{'eval':>26} {'scored':>8} {'echo':>7} {'real':>7} {'real %':>8} {'95% CI':>16}")
    for report in reports:
        low, high = wilson(report["real"], report["total"])
        print(f"{report['path'].stem:>26} {report['hits']:>8} {report['echoes']:>7} "
              f"{report['real']:>7} {report['real'] / report['total'] * 100:>7.2f}% "
              f"  [{low * 100:.2f}, {high * 100:.2f}]")

    print("\nnon-echo hits per benchmark (hits -> real):")
    names = sorted({n for r in reports for n in r["by_benchmark"]})
    print(f"{'benchmark':>12} " + " ".join(f"{r['path'].stem[:14]:>16}" for r in reports))
    for name in names:
        cells = []
        for report in reports:
            row = report["by_benchmark"].get(name)
            cells.append("               -" if row is None
                         else f"{row['hits']:>6} -> {row['real']:<6}")
        print(f"{name:>12} " + " ".join(cells))

    if args.examples:
        for report in reports:
            print(f"\n--- non-echo examples: {report['path'].stem} ---")
            for record in report["examples"]:
                print(f"  Q: {record.get('question', '')[:88]}")
                print(f"  A: {(record.get('answer_segment') or '')[:88]!r}")
                print(f"     predicted {record.get('prediction')!r}  gold {record.get('answer')!r}")

    print("\nAn echo is a hit whose gold answer already appears as a number in the question.")
    print("Real = hits - echoes, a LOWER bound on genuine capability.")


if __name__ == "__main__":
    main()
