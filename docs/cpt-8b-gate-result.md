# The 8B CPT fails its gate at 2050M, even with a correct schedule

Third attempt, with `--muon-learning-rate 0.001` actually reaching the optimizer
(the first two were the same run -- see docs/cpt-8b-reheat-failure.md and the
`load_checkpoint` fix). Gate criteria were written down before the numbers.

## Results after 50M CPT tokens

| check | threshold | result | |
|---|---|---|---|
| weight drift | < 40% | **28.27%** | PASS |
| benchmarks vs anchor | \|z\| < 2, low-N non-zero | **4.14% vs 6.61%, z = -5.49** | FAIL |
| repetition loops | < 10% | **24.1%** | FAIL |

The schedule fix worked exactly as intended: drift fell from 114.9% to 28.27%,
heldout loss from 2.577 to 2.2761 (baseline 2.1435), and training loss dropped
*below* the 2.0863 join. **The optimization is healthy and the capability still
regresses.**

## What is actually being lost

266 questions right at 2B are wrong after 50M CPT tokens; 142 go the other way.
The losses are not arithmetic failures -- they are a style shift:

| question | 2B | after CPT |
|---|---|---|
| Sandra took 6 cups, Marcie 2. Total? | `6 + 2 = 8 cups of coffee.` | `Sandra took six cups... Explanation: Given, Sandra...` |
| Adam has 5 more apples than Jackie, who has 9. | `5 + 9 = 14 apples.` | `Let the number of apples be x. 5x + 9 = 5x + 9 ... x = 0` |
| Allan brought 2 balloons, Jake 4. Total? | `Jake had two balloons. Explanation:...` | `The balloons were in the park.` (looping) |

The model is regressing toward the finemath corpus's register: restate the
problem, write "Given,", write "Explanation:", and often never commit to an
answer. It even applies spurious algebra scaffolding to one-step addition.

## Reading

The 2B model's arithmetic ability came from the same corpus, so more of that
corpus is not obviously wrong as a plan -- but at 145M parameters the
question-answering behaviour and the corpus's expository style compete for the
same capacity, and 8B more tokens of exposition wins. Continued pretraining on
the *pretraining* distribution pulls a model back toward that distribution and
away from the answer-shaped behaviour the benchmarks reward.

This is a capacity/objective mismatch, not a tuning bug, so a fourth LR is not
indicated. Getting answer-shaped behaviour out of this checkpoint is what
supervised fine-tuning is for, and `runs/muon-2b-sft` already exists.
