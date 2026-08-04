# Results: modern dense vs DeepSeek-V4 reference

Both models trained on identical tokens in identical order (verified by
test), same tokenizer, same held-out set, same optimizer settings, same
250,000,000-token budget, single seed 2026, no SFT on either side.

## Headline

| | DeepSeek-V4 145M (MoE) | ModernLM 145M (dense) |
|---|---:|---:|
| Stored parameters | 144,669,412 | 144,630,912 |
| Active parameters / token | 45,578,980 | 144,630,912 |
| Final held-out loss | 2.5514 | **2.4049** |
| Held-out perplexity | 12.8245 | **11.0773** |
| Throughput (tok/s) | 8,600 (as run) / 9,720 (best) | **35,975** |
| Wall clock for 250M tokens | 484.5 min (as run) | **115.8 min** |

## Pre-registered gates

**Gate 2 — quality at equal tokens.** ModernLM final held-out loss 2.4049 vs reference 2.5514: **-0.1465** (-5.74%).

**Gate 1 — time to quality (primary).** ModernLM first reached the reference's *final* loss (2.5514) after **170,000,384 tokens** in **78.8 minutes**. The reference needed 250,000,000 tokens and 484.5 minutes to get there.

That is a **6.1x** time-to-quality speedup against the reference as actually run, or **5.4x** against its best measured throughput.

**Same-token wall-clock speedup:** 4.18x vs the reference as run, 3.70x vs its best measured throughput.

## Loss trajectory

| Tokens | ModernLM | DeepSeek-V4 | Delta |
|---:|---:|---:|---:|
| 10,027,008 | 5.7493 | 6.5859 | -0.8366 |
| 20,021,248 | 4.6953 | 5.4421 | -0.7468 |
| 30,015,488 | 4.0069 | 4.7534 | -0.7465 |
| 40,009,728 | 3.5607 | 4.2434 | -0.6827 |
| 50,003,968 | 3.3059 | 3.8770 | -0.5712 |
| 60,030,976 | 3.1503 | 3.6009 | -0.4506 |
| 70,025,216 | 3.0388 | 3.4192 | -0.3804 |
| 80,019,456 | 2.9372 | 3.2545 | -0.3173 |
| 90,013,696 | 2.8609 | 3.1172 | -0.2564 |
| 100,007,936 | 2.7983 | 3.0386 | -0.2403 |
| 110,002,176 | 2.7469 | 2.9698 | -0.2229 |
| 120,029,184 | 2.6999 | 2.8923 | -0.1924 |
| 130,023,424 | 2.6625 | 2.8416 | -0.1791 |
| 140,017,664 | 2.6237 | 2.7911 | -0.1674 |
| 150,011,904 | 2.5897 | 2.7514 | -0.1617 |
| 160,006,144 | 2.5593 | 2.7099 | -0.1506 |
| 170,000,384 | 2.5311 | 2.6854 | -0.1543 |
| 180,027,392 | 2.5069 | 2.6525 | -0.1456 |
| 190,021,632 | 2.4826 | 2.6291 | -0.1464 |
| 200,015,872 | 2.4629 | 2.6093 | -0.1465 |
| 210,010,112 | 2.4464 | 2.5918 | -0.1454 |
| 220,004,352 | 2.4311 | 2.5802 | -0.1490 |
| 230,031,360 | 2.4187 | 2.5674 | -0.1488 |
| 240,025,600 | 2.4108 | 2.5581 | -0.1472 |
| 250,000,000 | 2.4049 | 2.5514 | -0.1465 |

The reference column is its `best_heldout_main_loss` at each recorded checkpoint. At intermediate milestones that is a running best rather than a point-in-time evaluation; the 250M endpoint is the clean comparison.

## Benchmarks (no SFT, greedy, 32 new tokens)

| Benchmark | DeepSeek-V4 | ModernLM |
|---|---:|---:|
| asdiv | 22 / 2305 (0.954%) | 50 / 2305 (2.169%) |
| svamp | 15 / 1000 (1.500%) | 23 / 1000 (2.300%) |
| gsm8k | 18 / 1319 (1.365%) | 22 / 1319 (1.668%) |
| algebra | 1 / 100 (1.000%) | 0 / 100 (0.000%) |
| arithmetic | 0 / 300 (0.000%) | 0 / 300 (0.000%) |
| **Overall** | **56 / 5024 (1.115%)** | **95 / 5024 (1.891%)** |

**These numbers are not a capability claim for either model.** Both sit at ~1.7 tokens per stored parameter, far below what matures a small language model, and sampled completions from both are degenerate (repetition loops, answers unrelated to the question, frequently no number at all). Differences of a few tenths of a percent here are noise. Held-out loss is the reliable signal at this budget.

**Reading the completions confirms the accuracy number is not measuring
arithmetic.** Every inspected "correct" answer is a case where the gold answer
happens to appear verbatim in the prompt, so copying an operand scores a point:

| Question | Gold | ModernLM completion |
|---|---:|---|
| "David has zero fewer apples than Marin. Marin has three apples..." | 3 | "The number of apples in Marin is 3." |
| "Brian has zero fewer oranges than Marcie. Marcie has 12 oranges..." | 12 | "Brian has 12 oranges." |
| "1 lonely pigeons was eating breadcrumbs. Another pigeon came..." | 2 | "...Question: 2 lonely pigeons was eating breadcrum" |

Zero-difference and 1+1 problems are exactly the cases operand-copying solves.
Incorrect answers show the same mechanism failing: "Seven red apples and two
green apples" → "There are seven apples in the basket" (copies 7, gold 9);
"Ellen has six more balls than Marin" → "The number of balls that Ellen has is
equal to the number of balls that Ellen has."

ModernLM's numeric completion rate is 85.0% versus the reference's 60.6%, so
much of the +39-answer gap is plausibly *more attempts that contain a number
at all*, not better reasoning. **The 1.891% vs 1.115% difference should not be
reported as a reasoning improvement.** Both models score near the floor of what
operand-copying yields on this suite. Algebra and arithmetic — the two
benchmarks where no operand-copy shortcut exists — are 0/100 and 0/300 for
ModernLM.

## Scope and limits

- **Single seed (2026) on both sides.** This is an exploratory result, not a settled architecture ranking. A three-seed replication would be needed before treating the loss gap as decision-grade.
- **Capacity-matched, not compute-matched.** ModernLM spends ~3.2x the active parameters per token (144.6M dense vs 45.6M active). The wall-clock win is therefore achieved *despite* a per-token compute disadvantage, but a compute-matched comparison would be a different experiment.
- **MTP and MoE are off** in ModernLM, and the reference used both. Part of any gap may be attributable to those, not to RoPE/SwiGLU/RMSNorm. The staged arms exist to separate that.
- The reference implementation is an explicitly unoptimized clean-room correctness reference (~3% MFU, memory-bound on FP32 elementwise traffic by its own profiling). The throughput gap measures these two implementations, not MoE versus dense in general.
