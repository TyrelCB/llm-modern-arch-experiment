#!/usr/bin/env python3
"""Recompute every `arithmetic` gold answer and report mismatches.

The arithmetic subset asks `Calculate A op B.`, which is fully checkable: parse
the two operands and the operator, evaluate, compare against the stored gold.

This was added after an audit mistakenly matched examples by operand pair while
ignoring the operator: the suite intentionally contains addition, subtraction,
and multiplication questions with the same operands. Recomputing all 300 from
the full expression distinguishes a real bad gold from that lookup error. A bad
gold would make a correct model answer score as wrong and contaminate every
recorded arithmetic result.

Read-only: reports, never rewrites the benchmark.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from modern_lm.data import default_paths  # noqa: E402

# `Calculate 99 × 3.` -- the corpus uses unicode × and ÷ as well as ascii.
PATTERN = re.compile(
    r"Calculate\s+(-?[\d,]+(?:\.\d+)?)\s*([+\-*/x×÷])\s*(-?[\d,]+(?:\.\d+)?)")

OPERATIONS = {
    "+": lambda a, b: a + b,
    "-": lambda a, b: a - b,
    "*": lambda a, b: a * b,
    "x": lambda a, b: a * b,
    "×": lambda a, b: a * b,
    "/": lambda a, b: a / b if b else None,
    "÷": lambda a, b: a / b if b else None,
}


def to_number(text: str) -> Fraction:
    return Fraction(text.replace(",", ""))


def render(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{float(value):g}"


def main() -> None:
    path = default_paths()["evaluation"]
    records = [json.loads(line) for line in path.open()]
    arithmetic = [r for r in records if r["benchmark"] == "arithmetic"]

    unparsed, mismatches = [], []
    operators = Counter()

    for record in arithmetic:
        match = PATTERN.search(record["question"])
        if not match:
            unparsed.append(record)
            continue
        left, operator, right = match.groups()
        operators[operator] += 1
        computed = OPERATIONS[operator](to_number(left), to_number(right))
        if computed is None:
            unparsed.append(record)
            continue
        try:
            stored = to_number(str(record["answer"]))
        except (ValueError, ZeroDivisionError):
            mismatches.append((record, computed, "unparseable gold"))
            continue
        if stored != computed:
            mismatches.append((record, computed, render(stored)))

    total = len(arithmetic)
    print(f"arithmetic questions: {total}")
    print(f"parsed:               {total - len(unparsed)}")
    print(f"operators seen:       {dict(operators)}")
    print(f"GOLD MISMATCHES:      {len(mismatches)} "
          f"({100 * len(mismatches) / max(1, total):.1f}%)\n")

    if unparsed:
        print(f"unparsed ({len(unparsed)}):")
        for record in unparsed[:5]:
            print(f"  {record['question'][:70]}")
        print()

    if mismatches:
        print(f"{'question':<34} {'stored':>12} {'correct':>12}")
        for record, computed, stored in mismatches[:40]:
            print(f"  {record['question'][:32]:<32} {stored:>12} "
                  f"{render(computed):>12}")
        if len(mismatches) > 40:
            print(f"  ... and {len(mismatches) - 40} more")

        by_operator = Counter(
            PATTERN.search(r["question"]).group(2) for r, _, _ in mismatches
            if PATTERN.search(r["question"]))
        print(f"\nmismatches by operator: {dict(by_operator)}")
        print(f"share of the 5,024-question benchmark: "
              f"{100 * len(mismatches) / len(records):.2f}%")


if __name__ == "__main__":
    main()
