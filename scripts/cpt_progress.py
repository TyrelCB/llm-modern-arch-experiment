#!/usr/bin/env python3
"""Compare CPT checkpoints against the 2B anchor: benchmarks, loss, completions.

Everything here is measured at the same 96-token budget as the anchor, so a
difference reflects the extra training rather than the eval settings.

Reports a two-proportion z-test against the anchor because the per-checkpoint
scatter on these benchmarks is a large fraction of the drift across a whole run
-- without it, ordinary noise reads as progress.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

ANCHOR_TOKENS = 2_000_000_000
# The 2B run's final 2.0863 was measured on the *2B* corpus's heldout split; the
# CPT reports against the 8B corpus's split. Scoring the untouched 2B checkpoint
# on both gives 2.0863 and 2.1435, so the corpus alone is worth 0.057. Compare
# against 2.1435 or the regression is overstated. See
# docs/cpt-8b-heldout-baseline.md.
JOIN_HELDOUT_LOSS = 2.1435
BENCHMARKS = ("asdiv", "svamp", "gsm8k", "algebra", "arithmetic")


def load_summaries(directory: Path) -> dict[int, dict]:
    out = {}
    for path in sorted(directory.glob("*.summary.json")):
        match = re.search(r"(\d+)", path.name)
        if match:
            out[int(match.group(1))] = json.loads(path.read_text())
    return out


def z_test(a_correct: int, a_total: int, b_correct: int, b_total: int) -> float:
    """Two-proportion z for b (later) vs a (anchor); + means the later one is better."""
    if not a_total or not b_total:
        return 0.0
    pooled = (a_correct + b_correct) / (a_total + b_total)
    se = math.sqrt(pooled * (1 - pooled) * (1 / a_total + 1 / b_total))
    if se == 0:
        return 0.0
    return (b_correct / b_total - a_correct / a_total) / se


def heldout_losses(train_log: Path) -> dict[int, float]:
    out = {}
    if not train_log.exists():
        return out
    with train_log.open() as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("event") == "evaluation":
                out[record["tokens_seen"]] = record["main_loss"]
    return out


def show_completions(jsonl: Path, benchmark: str, limit: int) -> None:
    if not jsonl.exists():
        return
    shown = 0
    with jsonl.open() as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("benchmark") != benchmark:
                continue
            mark = "OK " if record.get("correct") else "MISS"
            print(f"  [{mark}] {str(record.get('question', ''))[:96]}")
            print(f"         gold={record.get('answer')!r} pred={record.get('prediction')!r}")
            print(f"         {str(record.get('completion', ''))[:150]!r}")
            shown += 1
            if shown >= limit:
                return


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curve-dir", type=Path, default=Path("runs/curve-cpt"))
    parser.add_argument("--train-log", type=Path,
                        default=Path("runs/muon-cpt-8b/train.jsonl"))
    parser.add_argument("--completions", default="gsm8k",
                        help="benchmark to print sample completions from")
    parser.add_argument("--n-completions", type=int, default=4)
    args = parser.parse_args()

    summaries = load_summaries(args.curve_dir)
    if ANCHOR_TOKENS not in summaries:
        print("No 2B anchor evaluation yet; nothing to compare against.")
        return
    anchor = summaries[ANCHOR_TOKENS]
    losses = heldout_losses(args.train_log)

    print(f"anchor: 2B endpoint, overall {anchor['overall']['accuracy'] * 100:.2f}% "
          f"({anchor['overall']['correct']}/{anchor['overall']['total']}), "
          f"heldout loss {JOIN_HELDOUT_LOSS:.4f}")
    for name in BENCHMARKS:
        entry = anchor.get(name)
        if entry:
            print(f"        {name:11s} {entry['accuracy'] * 100:6.2f}% "
                  f"({entry['correct']}/{entry['total']})")

    later = sorted(t for t in summaries if t != ANCHOR_TOKENS)
    if not later:
        print("\nNo CPT checkpoints evaluated yet.")
    else:
        print(f"\n{'tokens':>9}  {'overall':>8}  {'vs anchor':>10}  {'z':>6}  {'heldout':>8}  vs join")
        for tokens in later:
            summary = summaries[tokens]
            overall = summary["overall"]
            delta = (overall["accuracy"] - anchor["overall"]["accuracy"]) * 100
            z = z_test(anchor["overall"]["correct"], anchor["overall"]["total"],
                       overall["correct"], overall["total"])
            loss = losses.get(tokens)
            loss_cell = f"{loss:8.4f}" if loss is not None else "       -"
            join_cell = f"{loss - JOIN_HELDOUT_LOSS:+.4f}" if loss is not None else "     -"
            print(f"{tokens / 1e6:8.0f}M  {overall['accuracy'] * 100:7.2f}%  "
                  f"{delta:+9.2f}pp  {z:6.2f}  {loss_cell}  {join_cell}")
        print("  |z| > 1.96 is significant at 95%; anything less is within noise.")

    print(f"\n--- sample {args.completions} completions "
          f"({'latest CPT checkpoint' if later else 'anchor'}) ---")
    tokens = later[-1] if later else ANCHOR_TOKENS
    stem = "cpt-{}".format(tokens)
    show_completions(args.curve_dir / f"{stem}.jsonl", args.completions, args.n_completions)


if __name__ == "__main__":
    main()
