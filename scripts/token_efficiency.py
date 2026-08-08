"""Token efficiency: how many tokens each run needs to reach a given loss.

`time_to_loss.py` prices wall-clock, which is what decides adoption. This asks
the different question of sample efficiency: at each held-out loss threshold,
how many training tokens did each run consume to first reach it, and what is the
ratio.

A ratio below 1.0 means the comparison run needed fewer tokens than the
baseline for the same loss. Note that a token-efficiency win is not automatically
a wall-clock win: Muon pays ~3.3% more per token, so a <3.3% token saving is
cancelled at equal hardware.

Thresholds are interpolated linearly between the two bracketing evaluations,
since evaluation cadence is coarse (10M tokens) relative to the differences
being measured.

Usage:
  python scripts/token_efficiency.py \
      --baseline runs/modern-145m --run runs/muon-250m-lr0.005
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def evaluations(run_dir: Path) -> list[tuple[int, float]]:
    log = run_dir / "train.jsonl"
    if not log.exists():
        raise FileNotFoundError(f"no train.jsonl in {run_dir}")
    out = []
    for line in log.read_text().splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("event") == "evaluation":
            out.append((r["tokens_seen"], r["main_loss"]))
    return sorted(out)


def tokens_to_reach(evals: list[tuple[int, float]], target: float) -> float | None:
    """Tokens at which loss first drops to `target`, linearly interpolated."""
    prev = None
    for tokens, loss in evals:
        if loss <= target:
            if prev is None:
                return float(tokens)
            ptok, ploss = prev
            if ploss == loss:
                return float(tokens)
            frac = (ploss - target) / (ploss - loss)
            return ptok + frac * (tokens - ptok)
        prev = (tokens, loss)
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", type=Path, default=Path("runs/modern-145m"))
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--throughput-tax", type=float, default=3.3,
                    help="percent slower per token; used to flag wall-clock wins")
    ap.add_argument("--step", type=float, default=0.1,
                    help="loss threshold spacing")
    args = ap.parse_args()

    base = evaluations(args.baseline)
    comp = evaluations(args.run)
    if not comp:
        print("no evaluations in the comparison run yet")
        return

    # Only thresholds both runs could plausibly have reached.
    floor = max(min(l for _, l in base), min(l for _, l in comp))
    ceiling = min(max(l for _, l in base), max(l for _, l in comp))
    step = args.step
    targets = []
    t = round(ceiling - (ceiling % step), 2)
    while t > floor:
        targets.append(round(t, 2))
        t = round(t - step, 2)

    print(f"baseline: {args.baseline.name}   run: {args.run.name}")
    print(f"{'loss':>7}{'baseline':>12}{'run':>12}{'ratio':>9}{'saving':>9}")
    print("-" * 49)
    for target in targets:
        b = tokens_to_reach(base, target)
        c = tokens_to_reach(comp, target)
        if b is None or c is None:
            continue
        ratio = c / b
        saving = (1 - ratio) * 100
        flag = " *" if saving > args.throughput_tax else ""
        print(f"{target:>7.2f}{b/1e6:>11.1f}M{c/1e6:>11.1f}M"
              f"{ratio:>9.3f}{saving:>+8.1f}%{flag}")

    print(f"\n  ratio < 1.000 = fewer tokens for the same loss")
    print(f"  * = token saving exceeds the ~{args.throughput_tax}% throughput "
          f"tax, i.e. also a wall-clock win")


if __name__ == "__main__":
    main()
