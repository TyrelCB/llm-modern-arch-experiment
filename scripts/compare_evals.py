#!/usr/bin/env python3
"""Compare benchmark eval runs: aggregate counts *and* the completions behind them.

Numbers alone have been wrong about this project repeatedly. A short list from
one session: a +12 algebra "win" that was the 32-token budget truncating both
models before they could talk themselves out of a correct answer; two scored
"correct" arithmetic answers where the model wrote `196 - 98 = -100` and the
scorer extracted 98 from the question echoed in a run-on; and a 2B benchmark
result read as a capability gap that turned out to sit inside the ±25 swing
between adjacent checkpoints of the same run. Each was caught by reading
completions, and none was visible in the accuracy column.

So this tool refuses to print a comparison without samples. `--samples 0` is
not accepted. The samples are stratified deliberately:

  gained   B right, A wrong   -- what the change bought
  lost     A right, B wrong   -- what it cost, never omitted
  both     both right         -- checks the wins are real, not scorer artifacts
  neither  both wrong         -- shows the failure mode the change did not fix

Reading `lost` and `both` is the point. `gained` alone is a highlight reel, and
a highlight reel is how a truncation artifact gets written into a README.

Usage:
  compare_evals.py a.jsonl b.jsonl --labels "AdamW 2B" "Muon 2B"
  compare_evals.py *.jsonl --benchmark algebra --samples 5
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
from pathlib import Path

WRAP = 100


def load(path: Path) -> dict:
    """Read per-example records. The .jsonl is authoritative -- evaluate_benchmarks
    writes progress and summary to the same descriptor, so a redirected
    .summary.json can be corrupted, but these rows are always intact."""
    rows = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "identifier" in d and "correct" in d:
            rows[(d["benchmark"], d["identifier"])] = d
    if not rows:
        raise SystemExit(f"no per-example records in {path}")
    return rows


def mcnemar(a: dict, b: dict, keys) -> tuple[int, int, float]:
    """Exact paired test. Only discordant pairs carry information, so this is a
    two-sided sign test on them -- the same test used throughout the project's
    other comparisons, kept identical so numbers stay comparable."""
    n01 = n10 = 0
    for k in keys:
        x, y = bool(a[k]["correct"]), bool(b[k]["correct"])
        if y and not x:
            n10 += 1
        elif x and not y:
            n01 += 1
    n = n01 + n10
    if n == 0:
        return 0, 0, 1.0
    tail = sum(math.comb(n, i) for i in range(min(n01, n10) + 1))
    return n10, n01, min(1.0, 2 * tail / 2 ** n)


def flat(text: str, limit: int = WRAP) -> str:
    s = re.sub(r"\s+", " ", (text or "")).strip()
    return s if len(s) <= limit else s[: limit - 1] + "…"


def stats(rows: dict) -> dict:
    n = len(rows)
    correct = sum(bool(d["correct"]) for d in rows.values())
    numeric = sum(d.get("prediction") is not None for d in rows.values())
    per: dict[str, list[int]] = {}
    for (b, _), d in rows.items():
        per.setdefault(b, [0, 0])
        per[b][1] += 1
        per[b][0] += bool(d["correct"])
    return {"n": n, "correct": correct, "numeric": numeric, "per": per}


def show_samples(a: dict, b: dict, keys, la: str, lb: str, k: int, seed: int) -> None:
    buckets = {"gained": [], "lost": [], "both": [], "neither": []}
    for key in keys:
        x, y = bool(a[key]["correct"]), bool(b[key]["correct"])
        buckets["gained" if (y and not x) else
                "lost" if (x and not y) else
                "both" if x else "neither"].append(key)

    rng = random.Random(seed)  # fixed seed: same run shows the same examples
    notes = {
        "gained": f"{lb} right, {la} wrong -- what the change bought",
        "lost": f"{la} right, {lb} wrong -- what it cost",
        "both": "both right -- verify the wins are real, not scorer artifacts",
        "neither": "both wrong -- the failure mode neither fixed",
    }
    for name in ("gained", "lost", "both", "neither"):
        pool = buckets[name]
        print(f"\n{'=' * WRAP}\n{name.upper()}  (n={len(pool)})  {notes[name]}\n{'=' * WRAP}")
        if not pool:
            print("  (none)")
            continue
        for key in rng.sample(pool, min(k, len(pool))):
            da, db = a[key], b[key]
            print(f"\n  [{key[0]}] gold={da['answer']}")
            print(f"  Q: {flat(da['question'])}")
            for lab, d in ((la, da), (lb, db)):
                mark = "OK " if d["correct"] else "   "
                print(f"    {mark}{lab:<22.22s} pred={str(d.get('prediction')):>10.10s} | {flat(d['completion'], 88)}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("evals", type=Path, nargs="+", help="per-example .jsonl files")
    p.add_argument("--labels", nargs="*", default=None)
    p.add_argument("--benchmark", default=None, help="restrict to one benchmark")
    p.add_argument("--samples", type=int, default=3,
                   help="completions per bucket (must be >=1; the point of this tool)")
    p.add_argument("--seed", type=int, default=2026)
    args = p.parse_args()

    if args.samples < 1:
        raise SystemExit(
            "--samples must be >= 1. Aggregate counts have been wrong about this "
            "project repeatedly and the errors were only ever visible in the "
            "completions; printing a table without them is how those errors got "
            "written down as findings. Use the module's load()/stats() directly "
            "if you truly need numbers alone.")

    labels = args.labels or [e.stem for e in args.evals]
    if len(labels) != len(args.evals):
        raise SystemExit("--labels must match the number of eval files")

    data = {lab: load(e) for lab, e in zip(labels, args.evals)}
    shared = set.intersection(*(set(r) for r in data.values()))
    if args.benchmark:
        shared = {k for k in shared if k[0] == args.benchmark}
        if not shared:
            raise SystemExit(f"no shared items for benchmark {args.benchmark!r}")

    benches = sorted({b for b, _ in shared})
    print(f"shared items: {len(shared)}"
          + (f"  (benchmark={args.benchmark})" if args.benchmark else ""))
    head = f"{'arm':<24}{'correct':>9}{'acc':>8}{'numeric':>9}  " + "".join(f"{b[:9]:>10}" for b in benches)
    print(f"\n{head}\n{'-' * len(head)}")
    for lab, rows in data.items():
        sub = {k: rows[k] for k in shared}
        s = stats(sub)
        line = (f"{lab:<24}{s['correct']:>9}{s['correct'] / s['n'] * 100:>7.2f}%"
                f"{s['numeric'] / s['n'] * 100:>8.1f}%  ")
        line += "".join(f"{s['per'].get(b, [0, 0])[0]:>10}" for b in benches)
        print(line)

    if len(data) > 1:
        print("\npaired exact McNemar (later arm vs earlier):")
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                la, lb = labels[i], labels[j]
                w, l, pv = mcnemar(data[la], data[lb], shared)
                sig = "*" if pv < 0.05 else " "
                print(f"  {lb:<22.22s} vs {la:<22.22s} gained={w:>4} lost={l:>4} "
                      f"net={w - l:>+5} p={pv:<10.4g}{sig}")

    # Samples for the first and last arm: with two files that is the comparison,
    # with more it frames the span the sweep covers.
    if len(data) > 1:
        show_samples(data[labels[0]], data[labels[-1]], sorted(shared),
                     labels[0], labels[-1], args.samples, args.seed)
    else:
        lab = labels[0]
        rows = data[lab]
        rng = random.Random(args.seed)
        for name, want in (("CORRECT", True), ("WRONG", False)):
            pool = [k for k in sorted(shared) if bool(rows[k]["correct"]) is want]
            print(f"\n{'=' * WRAP}\n{name}  (n={len(pool)})\n{'=' * WRAP}")
            for key in rng.sample(pool, min(args.samples, len(pool))):
                d = rows[key]
                print(f"\n  [{key[0]}] gold={d['answer']}  pred={d.get('prediction')}")
                print(f"  Q: {flat(d['question'])}")
                print(f"  A: {flat(d['completion'], 88)}")


if __name__ == "__main__":
    main()
