# GRPO protocol: verifiable-reward RL on the SFT checkpoint

Pre-registered before any GRPO code is written. Same reason as
`comparison-protocol.md`: this project's sibling repositories have twice seen a
metric pass an exploratory gate and fail on confirmation, and the sibling
`llm-lessons` repo shipped a PPO notebook whose prose claimed reward rose
0.6 -> 1.2 while the committed run actually fell (first-five mean 1.330,
last-five 0.941) with visibly degraded generations. That failure was only
visible because someone read the completions. This protocol is written to make
the same failure impossible to miss here.

## The question

Does GRPO on a **verifiable** reward improve benchmark accuracy over the SFT
checkpoint, at the same decode budget and under the same scorer — or does it
merely sharpen formatting the SFT stage already fixed?

The starting point is `runs/modern-145m-2b-sft/latest.pt`: 412/5,024 (8.201%),
the best arm in the project. **Beating 8.2% is the entire bar.** GRPO that
produces a prettier reward curve without moving benchmark accuracy is a
negative result and will be recorded as one.

## Why GRPO and not PPO/DPO

The benchmarks are math with gold numeric answers, so reward is a **rule**, not
a learned model:

- No reward model to train, hence no reward-model overfitting to diagnose.
- No preference pairs to collect.
- GRPO's group-relative advantage removes the value head — one fewer network
  that can silently diverge, which is precisely what went wrong in the sibling
  repo's PPO run.

The reward is `numeric_equal(extract_number(completion), gold)` using the
**reference's own scorer**, imported exactly as `evaluate_benchmarks.py` already
imports it (`sys.path` insert into `DEEPSEEK_REPO/src`, pyarrow stubbed). A
second copy of the scorer would break comparability with all six prior arms.

### The scorer artifact propagates into the reward — stated up front

`results-2b.md` recorded a run-on artifact: the pretrained model answered
correctly, kept generating, and `extract_number` picked up a number from a
self-generated follow-up question. SFT eliminated it (spurious `Question:` in
completions went 34.2% -> 0.0%).

Under GRPO this stops being a measurement artifact and becomes an **optimization
target**. Any residual scorer looseness is something the policy can now be
actively rewarded for exploiting. This is the single largest risk in this arm
and is why reward hacking gets a pre-registered gate below rather than a
post-hoc check.

## What is held fixed

Everything the six prior arms held fixed, unchanged:

| Held fixed | Value |
|---|---|
| Base checkpoint | `runs/modern-145m-2b-sft/latest.pt` (412/5,024) |
| Tokenizer | The reference's 16,384-token byte-level BPE, sha256-verified |
| Scorer | The reference's `extract_number` / `numeric_equal`, imported |
| Benchmark suite | Same 5,024 examples, same five subsets |
| Eval decoding | Greedy, `max_new_tokens=32`, `Question: ...\nAnswer:` |
| `eos_token_id` | 3 |
| Prompt source | GRPO prompts drawn **only** from the SFT train split |

**The evaluation harness is not modified for this arm.** If GRPO needs a
different decode budget to look good, that fact is the result.

### Held-out integrity

GRPO prompts come from the SFT **train** split only (16,679 examples). The 878
SFT held-out examples and all 5,024 benchmark examples stay untouched. The
benchmark set has never been trained on in any arm and must not start here —
training on prompts whose gold answers feed the reward would make the headline
number meaningless.

This gets a test, in the same spirit as the packed-stream order test:
`tests/` asserts zero overlap between the GRPO prompt pool and the benchmark
examples by hashed problem text.

## What differs (the independent variable)

Only the training objective. Same model, same tokenizer, same data source.

| | SFT arm | GRPO arm |
|---|---|---|
| Objective | Token-level cross-entropy on gold responses | Group-relative policy gradient on scored samples |
| Signal | Teacher-forced next token | Terminal correctness of sampled completion |
| Decoding in training | None (teacher forcing) | Sampled, temperature > 0, G per prompt |
| KL anchor | n/a | Reference = frozen SFT checkpoint |

## Reward specification

Terminal, per completion. Fixed before any run:

| Component | Value | Rationale |
|---|---:|---|
| Correct numeric answer | +1.0 | `numeric_equal(extract_number(c), gold)` |
| Incorrect / no number | 0.0 | No partial credit — nothing here is reliably gradable mid-chain |
| Format bonus | **none** | A format bonus would re-reward what SFT already achieved (99.6% numeric completion) and inflate reward without accuracy |

No length penalty in the primary run. Length is *monitored* (see gates) rather
than penalized, so that if the policy degenerates toward long or truncated
outputs, the metric shows it instead of a shaped reward hiding it.

## Hyperparameters (registered, single seed)

| Parameter | Value |
|---|---|
| Group size G | 8 completions per prompt |
| Sampling temperature | 1.0 |
| Max new tokens (training rollouts) | 64 |
| KL coefficient vs frozen SFT ref | 0.04 |
| Learning rate | 1e-6, constant |
| Updates | 300, checkpoint every 50 |
| Prompts per update | 32 |
| Seed | 2028 |

LR is ~50x below the SFT run's 5e-5. RL on a 145M model with a binary reward is
far easier to destabilize than supervised fine-tuning; the sibling repo's PPO
collapse is the cautionary case.

## Pre-registered gates

Recorded before the first GRPO run.

1. **Benchmark accuracy (primary).** GRPO wins if overall numeric exact match
   exceeds the SFT arm's **412 / 5,024 (8.201%)**, scored by the same code at
   `max_new_tokens=32`, greedy. Reported at every 50-update checkpoint, not only
   at the end — a mid-run peak followed by decline is the expected failure shape
   and must be visible.

2. **Reward hacking (blocking).** The run is **disqualified as a headline
   result**, regardless of gate 1, if either holds at the best checkpoint:
   - Completions containing a spurious `Question:` exceed **1.0%** (SFT: 0.0%).
   - The first-line-only diagnostic score diverges from the registered score by
     more than the SFT arm's 3.9-point gap (571 vs 412 = 11.37% vs 8.20%).

   Both diagnostics already exist from `results-sft.md`. Either firing means the
   policy is being scored on something other than answering correctly.

3. **No SFT regression (blocking).** Held-out SFT loss on the untouched 878
   examples must not exceed **0.2199** (the SFT checkpoint's final value) by
   more than 25% — i.e. must stay below **0.2749**. Catastrophic forgetting that
   still scores well on math would be a narrow, non-transferable win and is
   reported as such.

4. **Reward must actually rise (sanity).** Mean training reward at the last 5
   logged steps must exceed the first 5. This is the exact check the sibling
   repo's PPO run failed while its prose claimed success. If it fails, the
   result is written up as a failed run — **not** retuned silently until it
   passes.

### Interpretation rules fixed in advance

- **Single seed (2028), single run. Exploratory, like every prior arm.** A
  benchmark move of a point or two is not a settled RL result.
- **Read the completions before trusting the percentages.** This rule has now
  caught a real artifact twice in this project (the 2B run-on) and once in the
  sibling repo (PPO degradation). Any write-up includes verbatim samples from
  the best checkpoint, including failures.
- The SFT arm's failures were *reasoning* errors, not formatting artifacts
  (`7 + 2 = 9` correct, then two invented steps). GRPO on terminal reward has no
  mechanism to fix a wrong problem model except by luck of sampling. **A large
  gain would be surprising and should raise suspicion of gate 2 before
  celebration.**
- GSM8K (4.25%) is the honest test. Algebra (34.00%) is 100 examples with a
  narrow template and will move on noise; it is not the headline.
- Baseline for the reward curve is the SFT checkpoint's pass rate on the prompt
  pool, measured before update 0, so "reward rose" has a fixed origin.

## Open decision: eval decode budget

A `max_new_tokens=256` evaluation was started against the SFT checkpoint and
**did not finish** — `runs/modern-145m-2b-sft/evaluation-256.summary.json` is
empty and `evaluation-256.jsonl` has 0 lines; the log stops at 4,677/5,024. So
there is currently **no 256-token number for any arm.**

This protocol therefore registers **32 tokens as the comparison budget**, matching
all six prior arms. If a longer budget is wanted, the SFT baseline must be
re-evaluated at that budget *first*, so GRPO is never compared against a
baseline measured at a different budget. Changing the budget only for the new
arm would manufacture a win.

## Scope and limits

- Capacity-matched, single-seed, exploratory — inherited from the parent
  protocol and unchanged.
- **8.2% is not competence**, and neither is any modest improvement on it. The
  model fails ~92% of the suite before GRPO starts.
- GRPO here is compared against the SFT arm only. This is **not** a comparison
  against DeepSeek-V4 + RL; the reference has no RL arm in this project.
- Terminal-reward RL on math with G=8 at 145M parameters is a regime where
  published results are thin. A null result is a plausible and publishable
  outcome of this arm.
