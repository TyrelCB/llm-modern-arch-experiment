# Results: the 5M-300M capability ladder, post-SFT

Six pretraining runs on the FineMath 8B corpus (Muon, cosine schedule), each
SFT'd on `sft-math-words` for 1,000 updates and scored on the same 5,024-question
benchmark at a 32-token greedy budget. All numbers postdate the `extract_number`
scorer fix, so they are directly comparable to each other.

An HTML version with charts sits alongside this file:
[`results-ladder-5m-300m-2026-08-19.html`](results-ladder-5m-300m-2026-08-19.html).

## Headline

| Rung | Body params | Tokens | TPP | GPU h | Heldout loss | Overall | ASDiv | SVAMP | GSM8K | Algebra | Arith |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5M | 4,754,816 | 0.48B | 100 | 0.68 | 2.810 | 115 / 5,024 (2.29%) | 2.73% | 1.90% | 1.97% | 3.00% | 1.33% |
| 10M | 10,184,256 | 1.02B | 100 | 2.02 | 2.575 | 245 / 5,024 (4.88%) | 6.68% | 4.70% | 1.90% | 7.00% | 4.00% |
| 20M | 20,078,016 | 2.01B | 100 | 5.26 | 2.407 | 460 / 5,024 (9.16%) | 11.02% | 11.80% | 3.34% | 20.00% | 8.00% |
| 50M | 52,324,672 | 1.05B | 20 | 4.71 | 2.297 | 474 / 5,024 (9.43%) | 11.06% | 12.70% | 3.56% | 24.00% | 7.00% |
| 100M | 97,793,728 | 1.96B | 20 | 13.13 | 2.170 | 632 / 5,024 (12.58%) | 14.14% | 15.00% | 4.17% | 49.00% | 17.33% |
| **300M** | 296,267,264 | 5.93B | 20 | 92.83 | 1.908 | **851 / 5,024 (16.94%)** | 16.96% | 17.10% | 5.91% | 78.00% | 44.33% |

Body parameters exclude the embedding table and `lm_head`, per the project's
size convention.

## The overall figure hides the result

Overall accuracy fits `accuracy ~ params^0.445` — a clean-looking power law. But
it is an average over five benchmarks that behave completely differently:

| Category | Exponent | 5M -> 300M | Multiple |
|---|---:|---|---:|
| algebra | **0.774** | 3.00% -> 78.00% | 26x |
| arithmetic | **0.762** | 1.33% -> 44.33% | 33x |
| svamp | 0.498 | 1.90% -> 17.10% | 9x |
| asdiv | 0.391 | 2.73% -> 16.96% | 6x |
| gsm8k | **0.278** | 1.97% -> 5.91% | 3x |

Exponents are OLS fits on log-log against body parameters.

**Capacity buys procedure execution, not comprehension.** The models learn to
carry a digit and isolate a variable roughly twice as fast — in exponent terms —
as they learn to parse what a word problem is asking. Algebra and arithmetic were
still climbing steeply at the last rung with no saturation visible; asdiv and
svamp have effectively flattened near 17%.

## Two confounds that limit the conclusions

**The token budget changes mid-ladder.** 5M/10M/20M trained at 100 tokens per
parameter; 50M/100M/300M at 20. So 20M -> 50M is 2.6x the capacity on *half* the
tokens, which is very likely why 50M lands on top of 20M (9.16% -> 9.43%, well
inside noise) rather than above it. Read that plateau as a budget artifact, not
as evidence that capacity stopped paying.

**GSM8K is scored at a budget its answers exceed.** Multi-step solutions
routinely need more than 32 tokens, so the harness truncates answers the model
might otherwise complete. `^0.278` is a floor, not a ceiling. Re-run at 96 tokens
before drawing any conclusion about multi-step reasoning.

A third, smaller caveat: the 5M-100M rungs are **single evaluations**, while the
300M figure is the best of a 3x3 seed/checkpoint grid spanning 601-851 correct
(sd 79). Applying that variance to the lower rungs would likely blur the
20M-vs-50M ordering entirely. Treat mid-ladder ranking as provisional.

## The 300M grid

3 seeds x 3 checkpoints, 9 readings:

| Seed | 600 | 800 | 1000 |
|---|---:|---:|---:|
| 2027 | 727 | 744 | 752 |
| 2028 | 690 | 784 | **840** |
| 2029 | 601 | **851** | 816 |
| mean | 672.7 | 793.0 | 802.7 |
| sd | 64.8 | 54.1 | 45.5 |

Best is **seed 2029 at update 800** (851/5,024, 16.94%), and note that update 800
beat update 1000 on that seed — evaluating only endpoints would have reported 840
and missed the peak. This is the second time the grid has shown update 1000 is
not reliably best; see [`results-sft.md`](results-sft.md) and the 5280M grid.

851 is **selection-biased**: the max of 9 draws from a distribution with sd 79.
The mean of the best checkpoint column (~803) is the better estimate of what the
recipe yields on a fresh seed. Compare arms at a fixed grid shape, never
best-against-single-reading.

## Cost

Pretraining time grew far faster than accuracy. The 300M rung consumed **92.83 of
the ladder's 118.63 GPU-hours** (78%) to move overall accuracy 4.36 points past
100M.

| Rung | GPU h | milli-GPU-h per correct answer |
|---|---:|---:|
| 5M | 0.68 | 5.9 |
| 10M | 2.02 | 8.2 |
| 20M | 5.26 | 11.4 |
| 50M | 4.71 | 9.9 |
| 100M | 13.13 | 20.8 |
| 300M | 92.83 | 109.1 |

Cost per point rises 18x across the ladder.

## What to run next

Held-out loss improved monotonically at every rung (2.810 -> 1.908) and never
showed the plateau the benchmark did. Loss still tracks capacity cleanly; it is
the translation from loss to measurable capability that is uneven, and it is
uneven in a specific direction.

- **Add a 200M rung at 20 TPP.** That makes 100M -> 200M -> 300M the first clean
  three-point capacity series in the set, free of the token-budget confound.
  Shape is already in the ladder table: dim 896, 17 layers, 14 heads, ffn 3072.
- **Re-score GSM8K at 96 tokens** across all rungs before concluding anything
  about multi-step reasoning.
- **Word problems need something other than parameters.** Three rungs of capacity
  moved asdiv about 3 points. More scale is unlikely to be the cheapest fix.

## Provenance

Pretraining logs `runs/size{5m,10m,20m}-100x.log` and
`runs/size{50m,100m,300m}-20x.log`. Eval summaries `runs/eval-sft-5m.summary.json`,
`runs/eval-sft-10m.summary.json`, `runs/eval-sft-20m.summary.json`,
`runs/eval-sft-50m-cosine.summary.json`, `runs/eval-sft-100m.summary.json`, and
`runs/eval-sft-300m-final-seed2029-step000800.summary.json`. The full 300M grid
is `runs/eval-sft-300m-final-seed-grid.summary.json`.
