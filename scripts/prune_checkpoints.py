#!/usr/bin/env python
"""Apply the milestone retention policy to runs that predate it.

Existing runs wrote a checkpoint every N tokens and kept all of them, which is
why runs/ is 884GB. This reclassifies each run's checkpoints under the policy
new runs use -- keep every 10% milestone plus the WSD fork point forever, keep
the last few recent ones for resume, delete the rest.

DEFAULTS TO A DRY RUN. Nothing is deleted without --apply, because a deleted
checkpoint is unrecoverable: reproducing one costs the GPU-hours of the run up
to that point. The per-checkpoint .json metadata is never touched, so the loss
trajectory survives regardless.

    scripts/prune_checkpoints.py runs/size100m-20x           # show the plan
    scripts/prune_checkpoints.py runs/*/ --apply             # execute it
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from modern_lm.train import milestone_tokens  # noqa: E402

DEFAULT_PERCENTS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]


def run_settings(run_dir: Path) -> dict | None:
    """Recover a run's settings from any checkpoint's sidecar metadata."""
    for candidate in sorted(run_dir.glob("checkpoint-*.json")) + [run_dir / "latest.json"]:
        if candidate.exists():
            try:
                return json.loads(candidate.read_text())
            except (ValueError, OSError):
                continue
    return None


def decay_start_tokens(settings: dict) -> int | None:
    """Where WSD leaves peak LR, in tokens.

    Mirrors the trainer: decay is keyed to progress through the post-warmup
    updates, so a plain fraction of the token budget lands warmup-length early.
    """
    if settings.get("lr_schedule") != "wsd":
        return None
    fraction = settings.get("wsd_decay_fraction") or 0.0
    if not 0.0 < fraction < 1.0:
        return None
    planned = settings.get("planned_total_tokens") or 0
    tokens_per_update = (settings.get("microbatch_size", 0)
                         * settings.get("gradient_accumulation", 0)
                         * settings.get("sequence_length", 0))
    if not planned or not tokens_per_update:
        return None
    total_updates = max(1, planned // tokens_per_update)
    warmup = settings.get("warmup_updates", 0)
    stable = warmup + (total_updates - warmup) * (1.0 - fraction)
    return int(stable * tokens_per_update)


def plan(run_dir: Path, percents: list[int], keep_last: int) -> dict | None:
    checkpoints = sorted(run_dir.glob("checkpoint-*.pt"))
    if not checkpoints:
        return None
    meta = run_settings(run_dir)
    if meta is None:
        return {"run": run_dir, "error": "no metadata; skipped", "keep": checkpoints,
                "delete": [], "freed": 0}
    settings = meta.get("settings", {})
    planned = settings.get("planned_total_tokens") or 0
    if not planned:
        return {"run": run_dir, "error": "no planned_total_tokens; skipped",
                "keep": checkpoints, "delete": [], "freed": 0}

    fork = decay_start_tokens(settings)
    marks = milestone_tokens(planned, percents, fork)

    # Existing runs checkpointed on a round token grid (every 50M), which does
    # not divide their budget, so 10% of the run falls BETWEEN checkpoints --
    # by up to 23M tokens on the 100M rung. Matching within a tolerance would
    # therefore protect almost nothing. For historical runs the useful rule is
    # the nearest surviving checkpoint to each milestone; only new runs, which
    # force a write exactly at the mark, can rely on exact matching.
    #
    # The fork point is the exception: a checkpoint AFTER decay begins is not a
    # fork point at all, so only an earlier one may stand in for it.
    protected: set[Path] = set()
    by_tokens = {}
    for path in checkpoints:
        try:
            by_tokens[int(path.stem.split("-")[-1])] = path
        except ValueError:
            protected.add(path)  # unparseable: keep rather than guess
    for mark in marks:
        candidates = list(by_tokens)
        if fork is not None and mark == fork:
            candidates = [t for t in candidates if t <= mark]
        if candidates:
            protected.add(by_tokens[min(candidates, key=lambda t: abs(t - mark))])

    recent = [p for p in checkpoints if p not in protected]
    delete = recent[:-keep_last] if keep_last > 0 and len(recent) > keep_last else recent
    kept = [p for p in checkpoints if p not in set(delete)]
    return {"run": run_dir, "error": None, "keep": kept, "delete": delete,
            "freed": sum(p.stat().st_size for p in delete), "fork": fork,
            "milestones": marks}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--percents", default=",".join(str(p) for p in DEFAULT_PERCENTS))
    parser.add_argument("--keep-last", type=int, default=3,
                        help="recent non-milestone checkpoints to retain (default 3)")
    parser.add_argument("--apply", action="store_true",
                        help="actually delete; omit for a dry run")
    args = parser.parse_args()

    percents = [int(p) for p in args.percents.split(",") if p.strip()]
    total_freed = 0
    for run_dir in args.run_dirs:
        if not run_dir.is_dir():
            continue
        result = plan(run_dir, percents, args.keep_last)
        if result is None:
            continue
        if result["error"]:
            print(f"{run_dir.name:<34} {result['error']}")
            continue
        freed_gb = result["freed"] / 1e9
        total_freed += result["freed"]
        print(f"{run_dir.name:<34} keep {len(result['keep']):>3}  "
              f"delete {len(result['delete']):>3}  free {freed_gb:>6.1f} GB"
              + (f"  fork@{result['fork']/1e6:.0f}M" if result["fork"] else ""))
        if args.apply:
            for path in result["delete"]:
                path.unlink(missing_ok=True)

    verb = "freed" if args.apply else "would free"
    print(f"\n{verb} {total_freed/1e9:.1f} GB total")
    if not args.apply and total_freed:
        print("dry run -- re-run with --apply to delete")


if __name__ == "__main__":
    main()
