# Results: concise-response SFT on the 2B ModernLM

Eighth arm. Same pretrained checkpoint (`runs/modern-145m-2b/latest.pt`), same
SFT hyperparameters, same scorer, same greedy 32-token budget as the 412
baseline. **The only change is the SFT supervision target.**

**Result: 412 -> 473 of 5,024 (8.20% -> 9.41%), +61 questions, +14.8% relative.**

## The diagnosis that motivated it

Scoring the existing 2B+SFT checkpoint at a 256-token budget -- where 95% of
completions terminate on their own rather than being cut off -- shows accuracy
collapsing monotonically in the number of reasoning lines the model emits:

| Reasoning lines emitted | n | Accuracy |
|---:|---:|---:|
| 1 | 650 | **22.00%** |
| 2 | 2,207 | 6.57% |
| 3 | 1,112 | 3.87% |
| 4 | 393 | 1.78% |
| 5 | 203 | 3.45% |
| 6+ | 221 | ~2% |

For a model this small, every additional generated line is another opportunity
to corrupt a result it had already computed. The extra steps are not reasoning;
they are drift.

The gap this opens is large. Counting a completion as an "oracle hit" when the
gold answer appears *anywhere* in it:

| Budget | Oracle | Scored | Wasted |
|---|---:|---:|---:|
| 32 tokens | 17.93% | 8.20% | 9.73 pts |
| 256 tokens | 21.42% | 7.13% | 14.29 pts |

The baseline model was computing the right number two to three times more often
than it got credit for, then talking itself out of it. That -- not arithmetic
ability -- was the binding constraint.

It also explains the otherwise odd 256-token regression recorded in
`docs/results-sft.md` (358 < 412): the 32-token wall was accidentally acting as
a regularizer, truncating completions before the model could append a spurious
step. Removing the wall removed the accident.

## The change

`scripts/prepare_concise_sft.py` rewrites the SFT target to "compute, state the
answer, stop", touching neither the questions nor the train/heldout split:

- One-line responses (all 9,606 synthetic-math-v1 records) pass through
  unchanged -- already the target format.
- A multi-step response is replaced by its **last reasoning line** plus its
  unchanged `Final answer: N`, but only when that line is *grounded*: it
  contains an `=`, names no bare algebraic unknown, and every number in it
  appears in the question, in the answer, or earlier in that same response.
- Every other multi-step record is dropped.

| Split | Source | Kept | unchanged | quoted final step | dropped |
|---|---:|---:|---:|---:|---:|
| train | 16,679 | 13,780 | 9,606 | 4,174 | 2,899 |
| heldout | 878 | 722 | 478 | 244 | 156 |

**Every emitted response is verbatim source text.** This constraint did real
work. The first version of the script collapsed non-groundable records to a
bare `Final answer: 18`, which turns "Megan is delivering meals ..." into
supervision for stating an unjustified number -- training confident guessing,
a worse failure than rambling. An intermediate version quoted last lines like
`x=22 quarters`, referencing a variable the concise response never introduces.
Both were discarded on inspection before training; dropping 2,899 records costs
less than teaching 2,899 fabrications.

Allowing the final line to reference intermediates derived earlier in its own
response is what keeps word problems in the corpus: 4,174 gsm8k-train records
survive under that rule against 146 under a stricter question-only rule. The
supervision stays truthful, and the demand it places on the model is exactly
the target behaviour -- do the intermediate work implicitly, emit one step,
stop.

## Result

Greedy, 32 new tokens, reference scorer, 5,024 questions.

| | Baseline SFT | Concise SFT | Change |
|---|---:|---:|---:|
| **Overall** | **412 (8.20%)** | **473 (9.41%)** | **+61** |
| ASDiv | 213 (9.24%) | 241 (10.46%) | +28 |
| SVAMP | 74 (7.40%) | 116 (11.60%) | **+42** |
| GSM8K | 56 (4.25%) | 42 (3.18%) | **-14** |
| Algebra | 34 (34.0%) | 37 (37.0%) | +3 |
| Arithmetic | 35 (11.67%) | 37 (12.33%) | +2 |

Held-out SFT loss 0.1914 (perplexity 1.211). Training cost 194 s on GB10.

## The mechanism, verified

| | Baseline | Concise |
|---|---:|---:|
| Completions emitting 1 reasoning line | 978 | **4,921** |
| Completions emitting 2 | 3,071 | 103 |
| Completions emitting 3+ | 974 | 0 |
| Reached `Final answer:` within 32 tokens | 25.1% | **90.4%** |
| Oracle (answer appears anywhere) | **17.93%** | 14.49% |
| Scored | 8.20% | **9.41%** |

The decisive row is the last two together: **oracle fell while accuracy rose.**
The concise model produces the correct number *less* often in total, and
converts what it does produce into a scored answer far more reliably. The
baseline's higher oracle was not latent capability being unlocked -- it was
correct results being buried under a spurious final step.

Typical repairs:

```
Q: 20 peaches are in the basket. 25 more are put in. How many now?
  baseline: "20 + 25 = 45 peaches ... So, 20 + 45 = 65 peaches."   -> 65  wrong
  concise:  "The total ... is 20 + 25 = 45 peaches. Final answer: 45"  -> 45  right

Q: 2 birds were sitting on the fence. 4 more came. How many birds?
  baseline: "2 + 4 = 6 birds ... So, there are 6 x 2 = 12 birds."  -> 12  wrong
  concise:  "there are 8 - 2 = 6 birds. Final answer: 6"           ->  6  right
```

## The control: two steps is already too many

An obvious objection to the above is that shorter targets are generically
better -- that any truncation of the SFT responses would have helped, and one
step is not special. A second arm tests exactly that, built by the same script
and the same grounding rule but keeping the **last two** reasoning lines
instead of one (14,556 train records, 764 heldout), trained and scored
identically.

| Arm | SFT target | Benchmark |
|---|---|---:|
| Baseline SFT | full chain (1-8 lines) | 412 (8.20%) |
| **Concise SFT** | **1 line** | **473 (9.41%)** |
| Two-step SFT | 2 lines | 361 (7.19%) |

**Two steps is worse than the untouched baseline.** The effect is not
monotone in target length, so it is not "shorter data is better" -- it is
specifically that the second line is where this model starts overwriting its
own correct result. That is the finding the one-line arm rests on, and this
control is what makes it a claim about the mechanism rather than about
truncation.

The per-benchmark split is consistent with that reading:

| Benchmark | Baseline | Concise (1 line) | Two-step |
|---|---:|---:|---:|
| ASDiv | 213 | **241** | 173 |
| SVAMP | 74 | **116** | 59 |
| GSM8K | **56** | 42 | 52 |
| Algebra | 34 | **37** | 36 |
| Arithmetic | 35 | 37 | **41** |

GSM8K climbs back to 52 with two lines available, which supports the
interpretation that its concise-arm regression is genuinely about losing
multi-step decomposition rather than about corpus size. The single-step
benchmarks move the other way and dominate the total.

## The token budget stops being a trap

The 32-token figure above is the one that is comparable to every prior arm, and
it is the headline. But the diagnosis predicts something further: once the
model terminates on its own, a larger budget should stop costing accuracy.

For the baseline it cost a great deal — going 32 → 256 tokens *lost* 54
questions (412 → 358), because the extra room was spent on spurious steps. For
the concise model the same change gains:

| Budget | Baseline SFT | Concise SFT |
|---:|---:|---:|
| 32 | 412 | **473** |
| 64 | — | **480** |
| 256 | 358 | — |

At 64 tokens GSM8K also partially recovers (42 → 45), consistent with a few
genuinely two-step problems needing the room.

This is a **decoding-budget change, and therefore a separate claim** from the
473 headline: it is not comparable to the README table, whose rows are all
fixed at 32 tokens. It is reported because it settles the mechanism — the
baseline's budget regression was caused by run-on, not by decoding length as
such, and removing the run-on removes the regression.

## How solid is +61?

The two arms are scored on the identical 5,024 questions, so the comparison is
paired and the churn underneath the net is visible:

| | Count |
|---|---:|
| Wrong -> right (fixed by concise) | 347 |
| Right -> wrong (broken by concise) | 286 |
| **Net** | **+61** |

McNemar's test on the 633 discordant pairs gives **two-sided p ~ 0.017**. The
improvement is unlikely to be chance, but the churn is the more honest part of
the picture: the concise model is not the baseline plus 61 extra solves, it is
a substantially different predictor that wins 347 and loses 286. A model this
weak sits near chance on a large share of these items, and both arms flip many
of them. Per benchmark:

| Benchmark | Fixed | Broken | Net |
|---|---:|---:|---:|
| SVAMP | 106 | 64 | **+42** |
| ASDiv | 180 | 152 | **+28** |
| GSM8K | 34 | 48 | **-14** |

Only SVAMP has a fixed/broken ratio far from 1. ASDiv's +28 comes off 332
flips, so the point estimate there is the least stable of the three.

## Scope and limits

- **GSM8K regressed (56 -> 42).** It is the benchmark most dependent on genuine
  multi-step decomposition, and it lost the most supervision (2,899 dropped
  records are overwhelmingly gsm8k-train). The gain is concentrated in
  single-step-solvable problems: SVAMP +42, ASDiv +28. This arm buys accuracy
  on problems within one step's reach by giving up ground on problems that are
  not, and at this scale that trade is strongly positive -- but it is a trade,
  not a free improvement, and it would likely invert at a capacity where
  multi-step chains actually execute correctly.
- Single seed (2027), as with every prior arm. The McNemar test above addresses
  sampling over *questions*, not over training runs: it says the two fitted
  models differ, not that a re-run with a different seed would land at 473. The
  per-benchmark moves on algebra (+3 of 100) and arithmetic (+2 of 300) are
  individually within noise; only SVAMP is separated cleanly from the churn.
- The evaluation harness was not modified. Same scorer, same prompt, same
  greedy decoding, same 32-token budget as every row in the README table.
- This does not show the model reasons better. It shows the model was being
  scored on its worst output when it had a better one available, and that
  supervising it to stop recovers part of that gap.
