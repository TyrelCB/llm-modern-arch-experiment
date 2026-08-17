# Local Siamese/HybridNorm at 50M: pre-registered protocol

Registered 2026-08-16, **before** the run. The prediction below is recorded so
the result cannot be reinterpreted after the fact — the same discipline the
GRPO arm used, which is what made its negative result reportable rather than
quietly dropped.

> **Implementation-fidelity correction, 2026-08-16:** a post-launch audit against
> the paper's current official implementation found that this branch is not a
> faithful implementation of published SiameseNorm. It performs one combined
> update per Transformer block with `1/sqrt(layer+1)` scaling and a different
> final fusion; the reference path updates after attention and feed-forward
> separately, uses different depth scaling, and adds input/final normalization.
> The run remains useful as a controlled test of the local topology and is retained
> under `siamese-local-hybrid`. It cannot support a paper-level SiameseNorm claim.
> See [`D018`](decisions.md#d018). The preregistration below is preserved as the
> record that governed the launch.

## What is being tested

The tested local branch, inspired by SiameseNorm (arXiv 2602.08064, Alibaba),
replaces the single Pre-LN residual stream with two coupled streams over one
shared block body:

```
Y' = LN_Y(Y)                    # normalized read of the Pre-LN-like stream
O  = F(X + Y')                  # shared block, fed the fusion
X <- LN_X(X + O / sqrt(l+1))    # Post-LN-like stream, bounded, depth-scaled
Y <- Y + O                      # Pre-LN-like stream, raw accumulation
X_out = X_N + LN_final(Y_N)
```

The claim is that gradients then inherit both Pre-LN's identity path (`I`) and
Post-LN's normalization Jacobian (`J_LN`), rather than one or the other.

`F` is HybridNorm, not our Pre-LN sub-block: a second norm sits *after* the
attention sub-layer, inside the residual function, and a learnable per-channel
`gamma` mixes the normalized and raw attention input. Both are included in this
arm rather than testing the two-stream residual in isolation.

## Their result

1.3B OLMo, FineWeb-Edu, AdamW, QK-norm on. The margin grows with learning rate:

| LR | Baseline | SiameseNorm |
|---|---|---|
| 4e-4 | HybridNorm 10.91 PPL | **10.57** |
| 1e-3 | HybridNorm diverges | **10.43** |
| 2e-3 | Pre-Norm 10.89 PPL, arith 28.1% | **10.48**, arith **39.6%** |
| 2e-3, 350B tok | Pre-Norm 9.67, arith 36.2% | **9.42**, arith **43.4%** |

## Design decisions and why

**gamma initializes to 1.0.** That makes the attention input exactly the Pre-LN
input at step 0, so the model starts at the baseline and has to learn its way
off it. An init that started elsewhere would confound "SiameseNorm helps" with
"this particular init helps".

**The `0.02/sqrt(2L)` residual-output init is disabled on this path.** It is a
Pre-LN remedy for residual-variance growth, which is the same problem the
`1/sqrt(l+1)` divisor and per-layer `LN_X` solve structurally. The paper is
explicit that it initializes all norm scales to 1.0 "without the
Pre-Norm-biased initialization used in prior multi-path methods". Applying both
would damp the residual twice.

**All new parameters are 1D** (four RMSNorm gains and one `gamma` per layer,
plus one final Y-stream norm). The Muon/AdamW split keys on `ndim == 2`, so
they route to AdamW alongside every other norm gain with no change to
`muon.py`.

**Evaluation is post-SFT only.** Base models at this rung are mostly echo — SFT
lifts every rung 8–17x (`results-sft.md`) — so a base benchmark would measure
rambling, not capability.

## Throughput cost: -7.7%, not "negligible"

Measured before the run, 50M shape, microbatch 16, seq 512, compiled, 30 timed
steps after 12 warmup, Muon+AdamW as in the real arm:

| | tok/s | |
|---|---:|---|
| Pre-LN | 26,440 | |
| SiameseNorm | 24,397 | **-7.7%** |

The absolute numbers are below the real run's 59,465 tok/s because this
microbench uses no gradient accumulation and an unfused loss; only the ratio
transfers.

The paper calls its overhead "negligible" on the grounds that normalization is
cheap next to attention and MLP. That reasoning holds for FLOPs and fails for
wall clock at this size: the extra cost is four more RMSNorm passes over the
residual stream per layer plus a second `[B, T, D]` tensor carried through every
block, which is memory-bandwidth and kernel-launch bound, not FLOP bound. At
1.3B the ratio would be gentler, since attention and MLP grow faster than the
norms do.

For context, this is larger than the ~4.6% per-token cost Muon pays
(`results-muon.md`) — and Muon repaid that by reaching AdamW's final loss in 80%
of the tokens. So -7.7% sets a real bar: SiameseNorm has to win enough loss to
be worth ~8% more wall clock per token, and the decision rule below is about
capability rather than loss precisely because the Muon arm showed a clean loss
win can repay nothing at all.

## The comparison

One variable. `run_50m_20x_siamese.sh` is flag-for-flag identical to
`run_50m_20x_wsd.sh` except `--siamese-norm`: same 1,046,493,440 tokens, same
shape, same seed 2026, same Muon LR 0.005, same WSD schedule and zero floor.
The SFT is likewise identical to the WSD arm's, recovered from its stored
checkpoint metadata: `sft-math-words`, 1000 updates, lr 5e-5, seed 2027.

Anchor to beat, same 5,024 questions, same 32-token greedy budget, same scorer:

| Arm | Overall | ASDiv | GSM8K | Algebra | Arithmetic |
|---|---:|---:|---:|---:|---:|
| WSD (Pre-LN) + SFT | **459 (9.14%)** | 11.02% | 3.49% | 28.0% | 6.67% |
| SiameseNorm + SFT | — | — | — | — | — |

## Pre-registered prediction

**A null result is the expected outcome, and it is still worth the ~5 GPU-hours.**

Three reasons the paper's effect may not transfer to this rung:

1. **The regime is wrong for it.** Their margin is largest at aggressive LRs
   where the Pre-LN baseline is straining. Our stability budget is already
   spent: QK-norm is on, and Muon's orthogonalized updates control per-layer
   update scale — plausibly the same insurance the `1/sqrt(l+1)` divisor buys.
   The paper never tested Muon.
2. **The honest effect size is small.** Their PPL delta is ~0.04 nats. The Muon
   arm won 0.0935 nats here and produced **no** benchmark capability (88 vs 95,
   p = 0.59). Loss and benchmark decouple routinely in this repo.
3. **Scale.** 50M is 26x smaller than their 1.3B, and normalization pathologies
   are depth- and width-dependent.

**Decision rule, fixed in advance.** Per-arm SFT increments in the README range
from +24 (p = 0.30, not significant) to +71 (p = 0.0021), single seed. So:

- **< ±30 questions**: no detectable effect at 50M with Muon. Report as null.
  Do *not* promote to a larger rung on the strength of a sub-threshold trend.
- **> +30 questions**: promising on the canonical trajectory. First implement and
  test the faithful reference equations; then transfer the selected path to 300M
  or another token budget rather than routinely repeating the seed. This applies
  the current policy in [`D002`](decisions.md#d002).
- **Any divergence or loss regression**: report it. A negative architecture
  result at a known rung is worth as much as a positive one.

## Known risk

The paper reports "more significant emergence of 'massive activations'" than
baseline. This repo has been burnt by exactly that shape of failure — the CPT
reheat showed healthy training loss while algebra capability went to zero
(`cpt-8b-reheat-failure.md`). That is why the gate is a post-SFT benchmark and
not a pretrain loss curve.

## Files

- `scripts/run_50m_20x_siamese.sh` — pretraining
- `scripts/sft_50m_siamese.sh` — SFT + the single benchmark
- `src/modern_lm/layers.py` — `Block`, two-stream branch
- `src/modern_lm/model.py` — stream init and final fusion
- `tests/test_siamese_norm.py` — KV-cache equivalence, Muon split, init policy
