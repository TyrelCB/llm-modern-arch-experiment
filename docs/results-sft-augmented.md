# Results: arithmetic-augmented concise SFT

Ninth arm, built on the eighth. Same pretrained checkpoint, same hyperparameters,
same scorer, same greedy 32-token budget. The concise rewrite of
[`results-sft-concise.md`](results-sft-concise.md) is held fixed; the only
change is **6,000 additional arithmetic records** covering magnitudes and
operations the corpus barely contained.

**Result: 473 -> 497 of 5,024 (9.41% -> 9.89%). Against the original 412
baseline: +85, +20.6% relative.**

**Read the +24 cautiously.** Paired McNemar against the concise arm gives
**p = 0.30** — 255 questions fixed against 231 broken, which is what chance
looks like. The cumulative +85 over the baseline is solid (p = 0.00086), but
this arm's own increment is not separated from noise by this evidence. The
mechanism evidence below is stronger than the accuracy delta, and the
conclusion rests on it rather than on 497 > 473.

## The diagnosis

Two measurements on the concise arm's own eval output pointed the same way.

**Accuracy collapses with the magnitude of the answer**, and the corpus explains
why:

| Answer magnitude | Concise accuracy | Records in corpus |
|---|---:|---:|
| < 10 | 14.05% | 314 |
| 10-99 | 10.46% | 3,367 |
| 100-999 | 5.12% | 5,814 |
| **1000+** | **0.28%** | **111** |

The model had seen almost no arithmetic resolving above 1,000, and scored
essentially zero there. Multiplication was similarly thin: 892 records against
3,776 additions.

**Half the errors are computation, not comprehension.** Classifying the concise
arm's 4,551 wrong answers by whether the equation it wrote is internally
consistent:

| Error type | Count |
|---|---:|
| Right operation, arithmetic wrong | **1,746** |
| Arithmetic right, wrong problem setup | 1,607 |
| No parseable equation | 1,198 |

The first row is a coverage problem and can be attacked with data. The second is
not — a model that misreads the problem does not need more examples of long
division.

## The change

`scripts/augment_arithmetic_sft.py` generates 6,000 records from **the six
templates already present in synthetic-math-v1**, reproducing their response
strings exactly and sampling toward the gaps:

```
Calculate {a} + {b}.                         Calculate {a} - {b}.
Calculate {a} × {b}.                         Solve for x: {c}x + {k} = {n}.
A collection starts with {a} objects and receives {b} more. ...
A total of {n} items are split equally into {g} groups. ...
```

No new task format, phrasing, or reasoning is invented. These are arithmetic
ground truth in the corpus's existing wording. `verify` re-derives every stated
equation before it is written, which caught the linear template immediately --
its response states `69x = 21114` rather than a plain binary operation, and the
naive checker would have silently skipped it.

Magnitude coverage after augmentation:

| Bucket | Before | After |
|---|---:|---:|
| < 10 | 314 | 1,081 |
| 10-99 | 3,367 | 5,519 |
| 100-999 | 5,814 | 8,368 |
| **1000+** | **111** | **4,812** |

### Decontamination

Three distinct leaks were found by checking all three split pairs rather than
assuming any of them were safe:

1. Generated questions colliding with the 5,023 evaluation questions — **3
   caught and discarded**.
2. Heldout generation reproducing a question already generated for train —
   closed with `--exclude`.
3. The subtle one: **train generation reproducing a question that exists only
   in the base *heldout* split**. A different RNG seed does not prevent this,
   because both draws sample the same small template space; `Calculate 17 × 19`
   collided this way. Caught only because the train/heldout pair was checked
   explicitly.

Final corpus: **zero overlap on all three pairs** (train/eval, heldout/eval,
train/heldout).

## Result

Greedy, 32 new tokens, reference scorer, 5,024 questions.

| | Baseline | Concise | **Augmented** |
|---|---:|---:|---:|
| **Overall** | 412 (8.20%) | 473 (9.41%) | **497 (9.89%)** |
| ASDiv | 213 | 241 | **253** |
| SVAMP | 74 | 116 | **118** |
| GSM8K | **56** | 42 | 47 |
| Algebra | 34 | **37** | 35 |
| Arithmetic | 35 | 37 | **44** |

Nothing regressed against the concise arm except algebra (37 -> 35, 2 questions
of 100 — noise). GSM8K recovers 42 -> 47, roughly half the ground the concise
rewrite gave up.

## The mechanism, verified

The intervention was aimed at computation, and computation is what moved:

| | Concise | Augmented |
|---|---:|---:|
| Equations stated in completions | 3,710 | **4,034** |
| Of those, arithmetically correct | 1,911 (**51.5%**) | 2,227 (**55.2%**) |

The model both attempts more explicit computation and gets more of it right.
By magnitude:

| Bucket | Concise | Augmented |
|---|---:|---:|
| < 10 | 14.05% | **17.00%** |
| 10-99 | 10.46% | 9.67% |
| 100-999 | 5.12% | 5.23% |
| 1000+ | 0.28% | **1.69%** |

The targeted 1000+ bucket improved 6x, which confirms the coverage hypothesis
directly — but it holds only 355 of 5,024 questions, so it contributes about 5
of the +24. **Most of the gain came from the `<10` bucket instead** (+2.95
points over 1,253 questions), which the augmentation did not target. The honest
reading is that training on harder arithmetic improved general computational
competence, and the specific magnitude gap was the smaller half of the story.

## At a 64-token budget

Same checkpoint, decode budget raised to 64 — the point at which the concise
arm's curve saturated:

| Budget | Concise | Augmented |
|---:|---:|---:|
| 32 | 473 | 497 |
| 64 | 480 | **505 (10.05%)** |

**505 of 5,024 is +93 over the original 412 baseline**, and the first arm in
the project to clear 10%. As with the concise arm this is a decoding-budget
change and therefore not comparable to the README's 32-token table, which is
why 497 remains the headline figure.

## How solid is it?

Paired on the identical 5,024 questions:

| Comparison | Fixed | Broken | Net | McNemar (2-sided) |
|---|---:|---:|---:|---:|
| baseline -> augmented | 361 | 276 | **+85** | **p = 0.00086** |
| concise -> augmented | 255 | 231 | +24 | p = 0.30 |

The cumulative improvement over the original SFT arm is significant. **This
arm's marginal contribution is not.** 255 fixed against 231 broken is the churn
signature of a model that is near chance on a large share of these items, and
+24 sits comfortably inside it.

What survives that caveat is the mechanism measurement: the arithmetic
correctness rate of the model's own stated equations rose 51.5% -> 55.2% over
324 more equations attempted, and the targeted 1000+ bucket improved 6x. Those
are direct measurements of the thing the intervention was aimed at, and they do
not depend on the benchmark delta clearing a significance bar.

## Scope and limits

- Single seed (2029 for generation, 2027 for training), as with every prior arm.
- The augmentation is **synthetic and templated**. It teaches arithmetic on six
  fixed sentence patterns; it does not broaden the model's exposure to natural
  problem phrasing, which the error analysis says is the other half of the
  failure mass (1,607 wrong-setup errors). That half is untouched here.
- Generated data was verified arithmetically but is not human-reviewed. The
  guarantee is that every equation is correct and every response ends in the
  answer it derives — not that the phrasing is pedagogically ideal.
- The evaluation harness was not modified.
