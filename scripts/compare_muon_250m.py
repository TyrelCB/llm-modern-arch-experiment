"""Compare the Muon 0.005 250M run against the AdamW baselines.

Three arms, and only the first pair is a controlled comparison:

- AdamW 250M   -- same schedule, seed, and data order as the Muon run. This is
                  the optimizer comparison.
- Muon 250M    -- muon_learning_rate=0.005, weight_decay shared with AdamW.
- AdamW 2B     -- 8x the tokens on a different schedule. Not variable-isolated;
                  it is a ceiling, useful only for "how much of the 2B gap does
                  a better optimizer close".

Usage: PYTHONPATH=src python scripts/compare_muon_250m.py
"""
from __future__ import annotations

import json
from pathlib import Path

RUNS = Path(__file__).resolve().parent.parent / "runs"

ARMS = [
    ("AdamW 250M", "modern-145m"),
    ("Muon 0.005 250M", "muon-250m-lr0.005"),
    ("AdamW 2B", "modern-145m-2b"),
]


def evaluations(run: str) -> list[tuple[int, float, float]]:
    path = RUNS / run / "train.jsonl"
    if not path.exists():
        return []
    out = []
    with path.open() as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("event") == "evaluation":
                out.append((r["tokens_seen"], r["main_loss"],
                            r.get("elapsed_seconds", float("nan"))))
    return out


def main() -> None:
    data = {label: evaluations(run) for label, run in ARMS}

    adamw = {t: l for t, l, _ in data["AdamW 250M"]}
    muon = {t: l for t, l, _ in data["Muon 0.005 250M"]}

    print("Controlled comparison (same schedule, seed, data order)\n")
    print(f"{'Tokens':>8}{'AdamW':>10}{'Muon':>10}{'Muon-AdamW':>12}")
    print("-" * 40)
    # Token boundaries differ by a few thousand between runs; bucket to 10M.
    for bucket in sorted({round(t / 1e7) for t in list(adamw) + list(muon)}):
        a = next((l for t, l in adamw.items() if round(t / 1e7) == bucket), None)
        m = next((l for t, l in muon.items() if round(t / 1e7) == bucket), None)
        if a is None and m is None:
            continue
        row = f"{bucket * 10:>7}M"
        row += f"{a:>10.4f}" if a is not None else f"{'-':>10}"
        row += f"{m:>10.4f}" if m is not None else f"{'-':>10}"
        row += f"{m - a:>+12.4f}" if (a is not None and m is not None) else f"{'-':>12}"
        print(row)

    print("\nFinal losses\n")
    for label, _ in ARMS:
        e = data[label]
        if not e:
            print(f"  {label:<18} (no data)")
            continue
        t, l, sec = e[-1]
        print(f"  {label:<18} {l:.4f} at {t/1e6:.0f}M tokens "
              f"({sec/60:.0f} min)")

    if not (muon and adamw):
        return

    mt, ml, msec = data["Muon 0.005 250M"][-1]
    # Compare like with like: find AdamW's evaluation at the same token bucket
    # the Muon run has actually reached, not AdamW's final value.
    at, al, asec = min(data["AdamW 250M"],
                       key=lambda e: abs(e[0] - mt))
    if abs(at - mt) > 5e6:
        print(f"\n  no AdamW evaluation near {mt/1e6:.0f}M tokens; skipping delta")
        return

    if mt < 249e6:
        print(f"\n  [Muon run in progress: {mt/1e6:.0f}M / 250M tokens]")
    print(f"\n  At {mt/1e6:.0f}M tokens: Muon {ml:.4f} vs AdamW {al:.4f} "
          f"({ml - al:+.4f} nats)")
    print(f"  wall clock: Muon {msec/60:.1f} min vs AdamW {asec/60:.1f} min "
          f"({(msec/asec - 1)*100:+.1f}%)")

    b = data["AdamW 2B"]
    if b and mt >= 249e6:
        gap = al - b[-1][1]
        closed = (al - ml) / gap * 100 if gap else float("nan")
        print(f"\n  AdamW 250M -> 2B gap: {gap:.4f} nats")
        print(f"  Muon closes {closed:.1f}% of it at equal tokens "
              f"(not a controlled comparison; 2B is a ceiling)")


if __name__ == "__main__":
    main()
