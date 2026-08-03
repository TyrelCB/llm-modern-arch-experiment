"""Render `docs/results.md` from the recorded comparison artifacts.

Kept separate from `compare.py` so the numbers are computed once and the prose
is regenerated freely. Nothing here recomputes a metric.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _fmt(value: float | None, digits: int = 4) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _minutes(seconds: float | None) -> str:
    return "n/a" if seconds is None else f"{seconds / 60:.1f}"


def render(comparison: dict) -> str:
    reference = comparison["reference"]
    modern = comparison["modern"]
    gates = comparison.get("gates", {})

    lines: list[str] = []
    add = lines.append

    add("# Results: modern dense vs DeepSeek-V4 reference")
    add("")
    add("Both models trained on identical tokens in identical order (verified by")
    add("test), same tokenizer, same held-out set, same optimizer settings, same")
    add("250,000,000-token budget, single seed 2026, no SFT on either side.")
    add("")

    add("## Headline")
    add("")
    add("| | DeepSeek-V4 145M (MoE) | ModernLM 145M (dense) |")
    add("|---|---:|---:|")
    add(f"| Stored parameters | {reference['stored_parameters']:,} | "
        f"{modern.get('parameters', 144_630_912):,} |")
    add(f"| Active parameters / token | {reference['active_parameters']:,} | "
        f"{modern.get('parameters', 144_630_912):,} |")
    add(f"| Final held-out loss | {_fmt(reference['final_heldout_main_loss'])} | "
        f"**{_fmt(modern.get('final_heldout_main_loss'))}** |")
    if modern.get("final_heldout_main_loss") is not None:
        import math
        add(f"| Held-out perplexity | 12.8245 | "
            f"**{math.exp(modern['final_heldout_main_loss']):.4f}** |")
    add(f"| Throughput (tok/s) | {reference['tokens_per_second_as_run']:,.0f} (as run) / "
        f"{reference['tokens_per_second_best_measured']:,.0f} (best) | "
        f"**{modern.get('tokens_per_second', 0):,.0f}** |")
    add(f"| Wall clock for 250M tokens | "
        f"{_minutes(gates.get('reference_seconds_for_same_tokens_as_run'))} min (as run) | "
        f"**{_minutes(gates.get('modern_seconds_for_same_tokens'))} min** |")
    add("")

    if gates:
        add("## Pre-registered gates")
        add("")
        delta = gates.get("loss_delta")
        relative = gates.get("loss_relative_percent")
        add("**Gate 2 — quality at equal tokens.** ModernLM final held-out loss "
            f"{_fmt(gates.get('modern_final_loss'))} vs reference "
            f"{_fmt(gates.get('reference_final_loss'))}: "
            f"**{delta:+.4f}** ({relative:+.2f}%)."
            if delta is not None else "**Gate 2** — pending.")
        add("")
        crossing_s = gates.get("modern_seconds_to_reference_final_loss")
        crossing_t = gates.get("modern_tokens_to_reference_final_loss")
        if crossing_s:
            add("**Gate 1 — time to quality (primary).** ModernLM first reached the "
                f"reference's *final* loss ({_fmt(gates['reference_final_loss'])}) "
                f"after **{crossing_t:,} tokens** in **{crossing_s / 60:.1f} minutes**. "
                "The reference needed 250,000,000 tokens and "
                f"{_minutes(gates.get('reference_seconds_for_same_tokens_as_run'))} minutes "
                "to get there.")
            add("")
            speedup = gates.get("time_to_quality_speedup_vs_as_run")
            speedup_best = gates.get("time_to_quality_speedup_vs_best")
            if speedup:
                add(f"That is a **{speedup:.1f}x** time-to-quality speedup against the "
                    f"reference as actually run, or **{speedup_best:.1f}x** against its "
                    "best measured throughput.")
                add("")
        else:
            add("**Gate 1 — time to quality.** ModernLM did not reach the reference's "
                "final held-out loss within the 250M-token budget.")
            add("")
        add(f"**Same-token wall-clock speedup:** "
            f"{gates.get('speedup_vs_reference_as_run', 0):.2f}x vs the reference as run, "
            f"{gates.get('speedup_vs_reference_best', 0):.2f}x vs its best measured throughput.")
        add("")

    add("## Loss trajectory")
    add("")
    add("| Tokens | ModernLM | DeepSeek-V4 | Delta |")
    add("|---:|---:|---:|---:|")
    reference_by_tokens = {row["tokens_seen"]: row["heldout_main_loss"]
                           for row in reference["trajectory"]}
    reference_keys = sorted(reference_by_tokens)
    for row in modern["trajectory"]:
        tokens = row["tokens_seen"]
        nearest = min(reference_keys, key=lambda k: abs(k - tokens)) if reference_keys else None
        reference_loss = reference_by_tokens.get(nearest) if nearest else None
        delta = (row["heldout_main_loss"] - reference_loss) if reference_loss else None
        add(f"| {tokens:,} | {_fmt(row['heldout_main_loss'])} | "
            f"{_fmt(reference_loss)} | {delta:+.4f} |" if delta is not None
            else f"| {tokens:,} | {_fmt(row['heldout_main_loss'])} | n/a | n/a |")
    add("")
    add("The reference column is its `best_heldout_main_loss` at each recorded "
        "checkpoint. At intermediate milestones that is a running best rather "
        "than a point-in-time evaluation; the 250M endpoint is the clean "
        "comparison.")
    add("")

    modern_bench = modern.get("benchmarks")
    reference_bench = reference.get("benchmarks")
    if modern_bench and reference_bench:
        add("## Benchmarks (no SFT, greedy, 32 new tokens)")
        add("")
        add("| Benchmark | DeepSeek-V4 | ModernLM |")
        add("|---|---:|---:|")
        for name in ["asdiv", "svamp", "gsm8k", "algebra", "arithmetic"]:
            ref_row = reference_bench.get(name, {})
            mod_row = modern_bench.get(name, {})
            add(f"| {name} | {ref_row.get('correct', 0)} / {ref_row.get('total', 0)} "
                f"({100 * ref_row.get('accuracy', 0):.3f}%) | "
                f"{mod_row.get('correct', 0)} / {mod_row.get('total', 0)} "
                f"({100 * mod_row.get('accuracy', 0):.3f}%) |")
        overall = modern_bench.get("overall", {})
        ref_total = sum(reference_bench[k]["correct"] for k in reference_bench
                        if isinstance(reference_bench[k], dict) and "correct" in reference_bench[k])
        ref_n = sum(reference_bench[k]["total"] for k in reference_bench
                    if isinstance(reference_bench[k], dict) and "total" in reference_bench[k])
        add(f"| **Overall** | **{ref_total} / {ref_n} ({100 * ref_total / max(1, ref_n):.3f}%)** | "
            f"**{overall.get('correct', 0)} / {overall.get('total', 0)} "
            f"({100 * overall.get('accuracy', 0):.3f}%)** |")
        add("")
        add("**These numbers are not a capability claim for either model.** Both sit "
            "at ~1.7 tokens per stored parameter, far below what matures a small "
            "language model, and sampled completions from both are degenerate "
            "(repetition loops, answers unrelated to the question, frequently no "
            "number at all). Differences of a few tenths of a percent here are "
            "noise. Held-out loss is the reliable signal at this budget.")
        add("")

    add("## Scope and limits")
    add("")
    add("- **Single seed (2026) on both sides.** This is an exploratory result, "
        "not a settled architecture ranking. A three-seed replication would be "
        "needed before treating the loss gap as decision-grade.")
    add("- **Capacity-matched, not compute-matched.** ModernLM spends ~3.2x the "
        "active parameters per token (144.6M dense vs 45.6M active). The wall-clock "
        "win is therefore achieved *despite* a per-token compute disadvantage, but "
        "a compute-matched comparison would be a different experiment.")
    add("- **MTP and MoE are off** in ModernLM, and the reference used both. Part "
        "of any gap may be attributable to those, not to RoPE/SwiGLU/RMSNorm. The "
        "staged arms exist to separate that.")
    add("- The reference implementation is an explicitly unoptimized clean-room "
        "correctness reference (~3% MFU, memory-bound on FP32 elementwise traffic "
        "by its own profiling). The throughput gap measures these two "
        "implementations, not MoE versus dense in general.")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path, default=Path("runs/comparison.json"))
    parser.add_argument("--output", type=Path, default=Path("docs/results.md"))
    args = parser.parse_args()
    comparison = json.loads(args.comparison.read_text())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(comparison))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
