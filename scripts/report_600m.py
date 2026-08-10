#!/usr/bin/env python3
"""Current state of the 600M/8B run: loss trajectory and benchmark curve.

Compares against the 145M baselines on the same scorer and token budget:
  145M @ 2B        6.61% (332/5024)
  145M @ 2B + SFT  9.26% at its proper 32-token budget

Note the fair size comparison is 600M@8B vs 145M@2B -- matched tokens/param
(13.3 vs 13.8). At 2B the 600M has only 3.3 tokens/param and is still badly
under-trained, so an early shortfall there is not evidence against scaling.
"""
import json
import math
import re
from pathlib import Path

RUN = Path("runs/muon-600m-8b/train.jsonl")
CURVE = Path("runs/curve-600m")
BASE_145M = 332 / 5024


def main() -> None:
    if not RUN.exists():
        print("no run yet")
        return
    updates, evals = [], []
    with RUN.open() as handle:
        for line in handle:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("event") == "update":
                updates.append(r)
            elif r.get("event") == "evaluation":
                evals.append(r)

    if updates:
        last = updates[-1]
        seen = last["tokens_seen"]
        rate = last.get("tokens_per_second", 0) or 1
        remaining = (8_000_000_000 - seen) / rate
        print(f"tokens {seen:,} / 8,000,000,000  ({seen/8e9*100:.1f}%)")
        print(f"loss {last['loss_main']:.3f}  grad {last.get('grad_norm', 0):.2f}  "
              f"{rate:,.0f} tok/s")
        print(f"eta {remaining/86400:.1f} days")

    if evals:
        print(f"\nheldout loss ({len(evals)} checkpoints):")
        for r in evals[-6:]:
            print(f"  {r['tokens_seen']/1e6:7.0f}M  {r['main_loss']:.4f}")

    rows = []
    for path in sorted(CURVE.glob("m600-*.summary.json")):
        # Filenames carry the full token count: m600-200015872.summary.json.
        # A bare \d+ would stop at the "600" in the prefix.
        match = re.search(r"m600-(\d+)", path.name)
        if match:
            rows.append((int(match.group(1)), json.loads(path.read_text())))
    rows.sort(key=lambda row: row[0])
    if rows:
        print(f"\nbenchmarks (96 tokens, run-on fix) vs 145M@2B = 6.61%:")
        print(f"{'tokens':>9}  {'overall':>8}  {'vs 145M':>8}  {'z':>6}  alg  arith")
        for tokens, s in rows:
            o = s["overall"]
            acc = o["accuracy"]
            pooled = (o["correct"] + 332) / (o["total"] + 5024)
            se = math.sqrt(pooled * (1 - pooled) * (1 / o["total"] + 1 / 5024)) or 1
            print(f"{tokens/1e6:8.0f}M  {acc*100:7.2f}%  {(acc-BASE_145M)*100:+7.2f}pp  "
                  f"{(acc-BASE_145M)/se:6.2f}  {s['algebra']['correct']:3d}  "
                  f"{s['arithmetic']['correct']:3d}")


if __name__ == "__main__":
    main()
