"""Evaluate a GRPO checkpoint against the four pre-registered gates.

Gates are defined in `docs/grpo-protocol.md` and are not restated loosely here;
each check below names the gate it implements and the number it was registered
against. The point of running them from one script is that the verdict is
computed the same way every time rather than assembled by hand per checkpoint.

Usage:
  python3 scripts/evaluate_grpo_gates.py --run-dir runs/modern-145m-2b-grpo
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import types
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from modern_lm.data import DEEPSEEK_REPO  # noqa: E402

sys.path.insert(0, str(DEEPSEEK_REPO / "src"))
sys.modules.setdefault("pyarrow", types.ModuleType("pyarrow"))
_stub = types.ModuleType("pyarrow.parquet")
_stub.read_table = None
sys.modules["pyarrow.parquet"] = _stub
from deepseek_v4.evaluation import extract_number, numeric_equal  # noqa: E402

# Registered baselines, all measured on runs/modern-145m-2b-sft/latest.pt.
SFT_SCORE_32 = 412          # gate 1 bar
SFT_SCORE_256 = 358         # gate 2 clause 3 reference
SFT_FIRSTLINE_GAP = 3.12    # gate 2 clause 2, in points
SFT_SPURIOUS_RATE = 0.0     # gate 2 clause 1 (threshold 1.0%)
SFT_HELDOUT_LOSS = 0.2199   # gate 3: must stay below 0.2199 * 1.25
GATE3_LIMIT = 0.2749


def load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def first_line_score(rows: list[dict]) -> int:
    """Pinned definition from the protocol: split on the first newline."""
    return sum(1 for r in rows
               if numeric_equal(extract_number(r["completion"].split("\n")[0]),
                                r["answer"]))


def run_eval(checkpoint: Path, output: Path, max_new_tokens: int) -> dict:
    if output.with_suffix(".summary.json").exists():
        existing = output.with_suffix(".summary.json").read_text().strip()
        if existing:
            return json.loads(existing)
    subprocess.run([sys.executable, "-u", "-m", "src.modern_lm.evaluate_benchmarks",
                    "--checkpoint", str(checkpoint), "--output", str(output),
                    "--max-new-tokens", str(max_new_tokens), "--device", "cuda"],
                   cwd=REPO, check=True, stdout=subprocess.DEVNULL)
    return json.loads(output.with_suffix(".summary.json").read_text())


def analyze(rows: list[dict]) -> dict:
    total = len(rows)
    correct = sum(1 for r in rows if r["correct"])
    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / total,
        "first_line": first_line_score(rows),
        "spurious_rate": sum(1 for r in rows if "Question:" in r["completion"]) / total,
        "numeric_rate": sum(1 for r in rows if r["prediction"] is not None) / total,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoints", nargs="*", type=Path)
    parser.add_argument("--skip-256", action="store_true",
                        help="Only score at 32 tokens (the registered budget).")
    args = parser.parse_args()

    checkpoints = args.checkpoints or sorted(args.run_dir.glob("checkpoint-*.pt"))
    if not checkpoints:
        print(f"no checkpoints in {args.run_dir}")
        return

    train_log = args.run_dir / "train.jsonl"
    rewards = []
    if train_log.exists():
        for line in train_log.read_text().splitlines():
            record = json.loads(line)
            if record.get("event") == "update":
                rewards.append(record["reward_mean"])

    results = []
    for checkpoint in checkpoints:
        step = checkpoint.stem.split("-")[-1]
        path32 = args.run_dir / f"evaluation-{step}.jsonl"
        run_eval(checkpoint, path32, 32)
        entry = {"checkpoint": checkpoint.name, "tokens_32": analyze(load_rows(path32))}
        if not args.skip_256:
            run_eval(checkpoint, args.run_dir / f"evaluation-{step}-256.jsonl", 256)
            entry["tokens_256"] = analyze(load_rows(
                args.run_dir / f"evaluation-{step}-256.jsonl"))
        results.append(entry)
        print(json.dumps(entry), flush=True)

    print("\n" + "=" * 72)
    print("PRE-REGISTERED GATES (docs/grpo-protocol.md)")
    print("=" * 72)

    best = max(results, key=lambda e: e["tokens_32"]["correct"])
    b32 = best["tokens_32"]
    print(f"\nBest checkpoint by 32-token score: {best['checkpoint']}")
    print(f"  32-token : {b32['correct']}/{b32['total']} ({100*b32['accuracy']:.3f}%)")

    # Gate 1
    passed1 = b32["correct"] > SFT_SCORE_32
    print(f"\nGate 1 (primary) benchmark accuracy > {SFT_SCORE_32}: "
          f"{'PASS' if passed1 else 'FAIL'} "
          f"({b32['correct']} vs {SFT_SCORE_32}, {b32['correct']-SFT_SCORE_32:+d})")

    # Gate 2
    spurious_ok = b32["spurious_rate"] <= 0.01
    gap = 100 * (b32["first_line"] - b32["correct"]) / b32["total"]
    gap_ok = gap <= SFT_FIRSTLINE_GAP
    print(f"\nGate 2 (blocking) reward hacking:")
    print(f"  spurious 'Question:' {100*b32['spurious_rate']:.2f}% <= 1.00%: "
          f"{'ok' if spurious_ok else 'FIRED'}")
    print(f"  first-line gap {gap:+.2f}pt <= {SFT_FIRSTLINE_GAP}pt: "
          f"{'ok' if gap_ok else 'FIRED'}")
    divergence_ok = True
    if "tokens_256" in best:
        b256 = best["tokens_256"]
        print(f"  256-token: {b256['correct']}/{b256['total']} "
              f"({100*b256['accuracy']:.3f}%)")
        rose32 = b32["correct"] > SFT_SCORE_32
        fell256 = b256["correct"] < SFT_SCORE_256
        divergence_ok = not (rose32 and fell256)
        print(f"  32 up while 256 down: {'FIRED' if not divergence_ok else 'ok'} "
              f"(32 {b32['correct']-SFT_SCORE_32:+d}, 256 {b256['correct']-SFT_SCORE_256:+d})")
    gate2 = spurious_ok and gap_ok and divergence_ok
    print(f"  => Gate 2 {'PASS' if gate2 else 'DISQUALIFIED'}")

    # Gate 4
    if len(rewards) >= 10:
        first5 = sum(rewards[:5]) / 5
        last5 = sum(rewards[-5:]) / 5
        print(f"\nGate 4 (sanity) reward rose: "
              f"{'PASS' if last5 > first5 else 'FAIL'} "
              f"(first5 {first5:.4f} -> last5 {last5:.4f})")

    print("\nGate 3 (SFT regression) requires an SFT held-out loss pass; "
          f"limit is {GATE3_LIMIT:.4f} (SFT was {SFT_HELDOUT_LOSS:.4f}).")

    (args.run_dir / "gates.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
