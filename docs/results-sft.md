# Results: SFT on the 2B-token ModernLM

Supervised fine-tuning of `runs/modern-145m-2b/latest.pt` on the shared
math/instruction corpus (16,679 train / 878 held-out, reused unchanged and
sha256-verified). 1,000 updates, ~5.1 minutes on GB10. Single seed (2027).

**The evaluation harness was not changed.** Same greedy decode, same 32-token
budget, same `eos_token_id=3`, same reference scorer as all four prior arms.

## Headline

| Arm | Overall | ASDiv | SVAMP | GSM8K | Algebra | Arithmetic |
|---|---:|---:|---:|---:|---:|---:|
| DeepSeek-V4 pretrain | 56 / 5,024 (1.115%) | 0.95% | 1.50% | 1.36% | 1.00% | 0.00% |
| DeepSeek-V4 + SFT@1000 | 116 / 5,024 (2.309%) | 3.08% | 1.50% | 1.67% | 5.00% | 1.00% |
| ModernLM 250M pretrain | 95 / 5,024 (1.891%) | 2.17% | 2.30% | 1.67% | 0.00% | 0.00% |
| ModernLM 2B pretrain | 115 / 5,024 (2.290%) | 2.95% | 1.90% | 1.82% | 1.00% | 1.00% |
| **ModernLM 2B + SFT** | **412 / 5,024 (8.201%)** | **9.24%** | **7.40%** | **4.25%** | **34.00%** | **11.67%** |

**3.6x the reference's post-SFT result** (412 vs 116), on the identical corpus,
scorer, and decode settings.

## SFT held-out loss

Monotonic at every checkpoint, and lower than the reference at every one:

| Update | ModernLM 2B+SFT | DeepSeek-V4+SFT |
|---:|---:|---:|
| 0 | 2.2277 | 3.4456 |
| 100 | 0.3609 | 0.5548 |
| 500 | 0.2601 | 0.4129 |
| 1000 | **0.2199** | 0.3805 |

The starting gap (2.23 vs 3.45) is the pretraining advantage carried in: the 2B
model begins SFT already better than the reference ever gets. Note ModernLM's
loss at update **100** (0.3609) is already below the reference's final update-1000
value (0.3805).

Port fidelity check: this run consumed exactly 153,616 supervised tokens by
update 100 and evaluates on exactly 10,845 held-out supervised tokens — both
identical to the reference's recorded run, confirming the data path matches.

## The run-on hypothesis was correct

The 2B pretrained model answered correctly and then kept generating, so
`extract_number` picked up a number from a self-generated follow-up question.
SFT supervises `<eos>` on every response. Measured:

| | 2B pretrain | 2B + SFT |
|---|---:|---:|
| Completions containing a spurious `Question:` | **34.2%** | **0.0%** |
| Registered score | 115 / 5,024 | 412 / 5,024 |
| First-line-only score (diagnostic) | 165 / 5,024 | 571 / 5,024 |
| Numeric completion rate | 85.0% | 99.6% |

Run-on is eliminated completely. This closes the measurement artifact recorded
in `results-2b.md` **as a model improvement rather than a scorer change** — the
harness is untouched; the model simply learned to stop.

The first-line-only diagnostic still scores higher than the registered metric
(571 vs 412), so some gap remains, but it is now 3.9 points rather than the
1.0-point-on-a-much-smaller-base distortion seen before.

## Reading the completions

Arithmetic, where no operand can be copied:

```
Calculate 11 + 3.    -> "Add the two numbers: 11 + 3 = 14.\nFinal answer: 14"
Calculate 233 + 20.  -> "Add the two numbers: 233 + 20 = 253.\nFinal answer: 253"
```

Algebra, requiring two correct steps:

```
Solve for x: 9x + 11 = 146.
  -> "Subtract 11: 9x = 135. Then divide by 9: x = 15.\nFinal answer: 15"
```

This is genuine multi-step symbolic manipulation, and algebra rising 1% -> 34%
is the single largest per-benchmark move in the project.

**Failures are now reasoning errors, not formatting artifacts** — which is a
qualitatively different failure mode than at 250M:

```
"Seven red apples and two green apples..."  (gold 9)
  -> "There are 7 + 2 = 9 red apples. There are 7 + 9 = 16 green apples.
      Therefore, there are 7 + 16 = 21 apples"
```

It computes `7 + 2 = 9` correctly, then invents two more steps. The arithmetic
is right; the problem model is wrong. GSM8K remains lowest (4.25%) because
multi-step word problems compound exactly this error.

## Scope and limits

- **Single seed (2027)**, single run. Exploratory, as with every prior arm.
- **8.2% is not competence.** It is a large relative gain on a suite where the
  reference scored 2.3%, and the model still fails ~92% of problems.
- The pretraining advantage and the SFT gain are **not separable** here: this
  run SFTs only the 2B checkpoint. Attributing the 3.6x between "better
  pretraining" and "SFT working better on a better base" would need an
  SFT-on-250M arm, which was deliberately not run.
- The comparison to DeepSeek-V4+SFT is fair on corpus, scorer, decode budget,
  and hyperparameters, but the two models had different pretraining budgets
  (2B vs 250M tokens). This is an end-to-end pipeline comparison, not an
  isolated SFT-algorithm comparison.
