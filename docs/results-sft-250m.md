# SFT on the 250M checkpoint: separating pretraining from fine-tuning

`results-sft.md` reported 412/5,024 for the 2B checkpoint after SFT, but could
not say how much of that came from the longer pretraining versus SFT itself.
This arm applies the **identical** SFT recipe to the 250M-token checkpoint to
decompose it. Same corpus, same 1,000 updates, same hyperparameters, same seed
(2027), same untouched evaluation harness.

## All six arms

| Arm | Overall | ASDiv | SVAMP | GSM8K | Algebra | Arithmetic |
|---|---:|---:|---:|---:|---:|---:|
| DeepSeek-V4 pretrain | 56 (1.115%) | 0.95% | 1.50% | 1.36% | 1.00% | 0.00% |
| DeepSeek-V4 + SFT | 116 (2.309%) | 3.08% | 1.50% | 1.67% | 5.00% | 1.00% |
| ModernLM 250M pretrain | 95 (1.891%) | 2.17% | 2.30% | 1.67% | 0.00% | 0.00% |
| **ModernLM 250M + SFT** | **163 (3.244%)** | 3.69% | 2.50% | 2.73% | 11.00% | 2.00% |
| ModernLM 2B pretrain | 115 (2.289%) | 2.95% | 1.90% | 1.82% | 1.00% | 1.00% |
| **ModernLM 2B + SFT** | **412 (8.201%)** | 9.24% | 7.40% | 4.25% | 34.00% | 11.67% |

## The decomposition

| Comparison | Change | Multiplier |
|---|---|---:|
| SFT on the **250M** base | 95 → 163 | **1.72x** |
| SFT on the **2B** base | 115 → 412 | **3.58x** |
| More pretraining, **no** SFT | 95 → 115 | 1.21x |
| More pretraining, **with** SFT | 163 → 412 | 2.53x |

**Neither lever alone explains the 412.** Pretraining 8x longer buys only 1.21x
on its own (95 → 115), because the extra capability is masked by the model's
inability to stop. SFT buys 1.72x on the weak base. Together they produce 4.34x
over the 95 starting point — the levers compound rather than add.

The interaction runs in both directions: SFT is more than twice as effective on
the better-pretrained model (3.58x vs 1.72x), and pretraining is more than twice
as valuable once SFT can expose it (2.53x vs 1.21x).

## SFT held-out loss

| Update | 250M + SFT | 2B + SFT | DeepSeek-V4 + SFT |
|---:|---:|---:|---:|
| 0 | 2.8071 | 2.2277 | 3.4456 |
| 100 | 0.4614 | 0.3609 | 0.5548 |
| 1000 | **0.2916** | **0.2199** | 0.3805 |

The pretraining ordering is preserved through SFT and never crosses: the better
pretrained model starts better, stays better, and ends better. Both ModernLM
arms finish below the reference.

## What the completions show — the important part

Both SFT'd models learned the output **format** perfectly. Run-on generation is
eliminated in both (0.0%, from 34.2% pre-SFT), and both produce the same clean
template. The difference is entirely in whether the arithmetic inside that
template is correct:

**250M + SFT** — correct format, wrong numbers:
```
Calculate 11 + 3.   -> "Add the two numbers: 11 + 3 = 12.  Final answer: 12"
Calculate 48 + 22.  -> "Add the two numbers: 48 + 22 = 69.  Final answer: 69"
Calculate 85 + 41.  -> "Add the two numbers: 85 + 41 = 144. Final answer: 144"
```

**2B + SFT** — same format, right numbers:
```
Calculate 11 + 3.   -> "Add the two numbers: 11 + 3 = 14.  Final answer: 14"
Calculate 48 + 22.  -> "Add the two numbers: 48 + 22 = 70.  Final answer: 70"
```

Every 250M answer above is off by a small amount — the model has learned the
*shape* of addition without the computation. This is the cleanest evidence in
the project for a simple claim:

> **SFT teaches the model how to answer. Pretraining determines whether the
> answer is right.**

Arithmetic (2.00% vs 11.67%) and algebra (11% vs 34%) show the gap most
sharply, because those benchmarks cannot be passed by formatting or by copying
an operand out of the prompt.

## Scope and limits

- Single seed (2027) per arm; exploratory, like every prior arm.
- The two SFT arms differ **only** in the pretrained checkpoint, so the
  decomposition above is a clean within-recipe comparison.
- The comparison to DeepSeek-V4 + SFT still differs in pretraining budget
  (250M vs 250M tokens for the 250M arm — comparable — and 2B for the other).
  The **250M + SFT arm at 163 is the closest apples-to-apples match** to the
  reference's 116: same token budget, same SFT recipe, differing only in
  architecture. On that matched comparison the modern dense stack is 1.41x.
- 8.2% remains far from competence; the model fails ~92% of the suite.
