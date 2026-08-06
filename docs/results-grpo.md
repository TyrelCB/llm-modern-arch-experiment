# Results: GRPO on the 2B+SFT ModernLM

Seventh arm. Protocol pre-registered in `docs/grpo-protocol.md` before any code
was written; this document records what happened against it.

**Status: the registered configuration produces no learning signal.** This is a
negative result about the configuration, established by measurement rather than
by running it to completion, and it is reported as the primary finding rather
than being retuned away.

## What was run

`src/modern_lm/grpo.py` was launched from `runs/modern-145m-2b-sft/latest.pt`
with the registered hyperparameters (G=8, temperature 1.0, 64-token rollouts,
LR 1e-6, KL 0.04, seed 2028), targeting 150 updates — cut from the registered
300 at a checkpoint boundary to fit the available window, as the protocol's cost
note requires.

It was stopped after one update. Update 1 logged:

```json
{"event": "update", "optimizer_step": 1, "reward_mean": 0.0,
 "reward_baseline": 0.0078125, "grad_norm": 0.0,
 "zero_variance_groups": 32, "completion_tokens": 15247.0}
```

`zero_variance_groups: 32` out of 32 groups, and `grad_norm: 0.0`. Every group
of 8 sampled completions was unanimously wrong, so every group-relative
advantage was exactly zero and the optimizer step was a no-op.

## Why: the pass rate is below what group-relative advantage can use

GRPO's advantage is `(r - mean(group)) / std(group)`. A group whose completions
all score the same carries no preference information and contributes no
gradient — correct behavior, but it means the usable fraction of a rollout is
governed by how often a group contains *both* a right and a wrong answer.

Measured on **256 prompts** drawn from the full pool with the registered seed
(2028), G=8, temperature 1.0, 64-token rollouts — i.e. one full update's worth
of rollout:

| Metric | Value |
|---|---:|
| Mean reward | **0.59%** (12 correct completions of 2,048) |
| Groups with nonzero variance (usable gradient) | **12 / 256 (4.7%)** |
| Groups unanimously wrong | **244 / 256 (95.3%)** |
| Groups unanimously right | 0 / 256 |

**95.3% of rollout compute produces exactly zero gradient.** Since rollout is
89% of update wall-clock (below), roughly 85% of the arm's total compute would
do nothing at all.

Per-source breakdown, 24 prompts each, same settings:

| Source | Pool | Mean reward | Groups with nonzero variance |
|---|---:|---:|---|
| gsm8k-train | 7,073 | 1.04% | 2/24 (8%) |
| synthetic-math-v1 | 9,606 | 6.25% | 6/24 (25%) |

The small per-source samples are optimistic relative to the 256-prompt
measurement — 25% versus the 4.7% seen at scale — which is itself a caution
about sizing a diagnostic off a couple of dozen prompts. The 256-prompt figure
is the one to trust.

This is a known GRPO failure mode, and the registered hyperparameters walked
into it. Recording it is the point of having registered them.

## Cost structure

Measured on GB10, 145M parameters, one update at the registered settings:

| Phase | Time | Share |
|---|---:|---:|
| Rollout (256 sequences x 64 tokens) | 50.8 s | 89% |
| Backward (policy + frozen reference) | ~6.2 s | 11% |
| **Total per update** | **~57 s** | |

`ModernLM` has no KV cache — `generate` recomputes the full prefix at every
decode step, deliberately, so that evaluation cost stays comparable to the
DeepSeek reference. `rollout` inherits this, which is why sampling dominates so
heavily. At 57 s/update the registered 300-update schedule is ~4.75 h.

## What would restore signal

Measured on synthetic-math-v1 (the higher-pass-rate source), 24 prompts:

| Configuration | Mean reward | Groups with nonzero variance |
|---|---:|---|
| temp 1.0, G=8 *(registered)* | 6.25% | 25% |
| temp 1.0, G=16 | 5.99% | 33% |
| temp 0.8, G=8 | 7.81% | 38% |
| temp 0.8, G=16 | 11.20% | 46% |

Lowering temperature raises the pass rate (fewer incoherent samples) and a
larger group raises the chance of catching a disagreement; together they roughly
double the usable fraction. G=16 also doubles rollout cost to ~114 s/update.

**Read these with the sample-size caution above.** They are 24-prompt
measurements on the higher-pass-rate source, and the equivalent 24-prompt figure
for the registered config (25%) overstated the 256-prompt result (4.7%) by more
than fivefold. The honest reading is a *relative* one — temperature 0.8 with
G=16 is better than the registered config — not that it reaches 46% at scale.
Even a fivefold improvement over 4.7% leaves most groups contributing nothing.

**These numbers are diagnostic, not a result.** Adopting them would be a
post-hoc hyperparameter change, which is precisely what the pre-registration
exists to prevent. Any run using them must be labelled an amended,
exploratory follow-up and reported separately from the registered arm.

### The deeper problem is the base model, not the sampler

At 0.59% sampled reward, no amount of temperature or group-size tuning
manufactures a learning signal from nothing: GRPO can only amplify a
distinction the policy already sometimes gets right. The SFT checkpoint scores
7.13% greedy at 256 tokens, and its *sampled* pass rate at temperature 1.0 is
an order of magnitude below that — sampling at temperature 1.0 from a 145M model
this weak mostly produces incoherent text, as the recorded samples show.

That points at prompt selection as the higher-leverage fix rather than sampler
settings: restricting the pool to problems the policy solves *sometimes* (say, a
measured per-prompt pass rate in 0.1-0.9) would raise the usable fraction far
more than any temperature change. That is a substantially different experiment
from the registered one and is noted here as the natural next arm, not run.

## Scope and limits

- The registered arm was **not** run to completion. The claim here is narrow and
  mechanical: at a measured 0.59% sampled pass rate, group-relative advantage is
  zero for 95.3% of groups, so the configuration cannot learn. It is **not** a
  claim that GRPO fails at this scale in general — only that it fails from this
  checkpoint, on this prompt pool, at these settings.
- Single seed (2028), as with every prior arm.
- The headline zero-variance measurement is 256 prompts (2,048 completions), one
  full update's worth of rollout, on the full pool at the registered seed. The
  per-source and alternative-setting tables are 24-prompt samples and are
  demonstrably optimistic; treat them as directional only.
- Nothing here evaluates whether GRPO *would* have improved benchmark accuracy.
  Gates 1-3 were never reached, because gate 4's precondition — that there be a
  gradient at all — was not met.
