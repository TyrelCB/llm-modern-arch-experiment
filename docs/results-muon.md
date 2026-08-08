# Review: Muon versus AdamW

Review date: 2026-08-07, updated 2026-08-08 with the completed LR probes. This
note records the review of the uncommitted Muon implementation and the completed
optimizer experiments. It is a recipe review, not a claim that Muon is
intrinsically inferior to AdamW.

## Decision

Do not adopt the current Muon recipe (`muon_learning_rate=0.02`,
`weight_decay=0.1`) and do not treat the 25M-token sweep as a confirmed win.
Muon is clearly better early in this experiment, but the selected recipe crosses
over and is worse than AdamW later in the actual 250M schedule. Note that this
verdict is about the *screened recipe*, not about Muon.

The completed `0.005` and `0.01` probes (see [Completed LR
probes](#completed-lr-probes-120m-tokens)) confirm the crossover was an
LR artifact rather than a property of Muon: at `0.005` Muon ends *ahead* of
AdamW at 120M tokens. The margin is small enough to sit inside the measured
throughput tax, so this is a reason to run the pre-registered screen, not a
reason to switch optimizers yet.

The implementation is reasonable enough to keep as an experimental branch. The
next comparison should tune Muon weight decay separately, use the real 250M
schedule during screening, and measure training time independently of checkpoint
I/O.

## What was measured

The 25M sweep used the same seed, data order, model, and held-out evaluation for
all arms. AdamW was run at the existing `3e-4` recipe; Muon was screened at four
learning rates:

| Arm | Peak Muon LR | Held-out loss | Elapsed time | Tokens/s |
|---|---:|---:|---:|---:|
| AdamW control | — | 3.5882 | 684s | 36,539 |
| Muon | 0.005 | 3.2383 | 708s | 35,302 |
| Muon | 0.010 | 3.1892 | 707s | 35,383 |
| **Muon** | **0.020** | **3.1585** | **707s** | **35,381** |
| Muon | 0.050 | 3.1786 | 737s | 33,942 |

The useful Muon arms therefore paid approximately a 3.3% throughput cost while
showing a large early loss advantage. The raw records are in
[`runs/muon-sweep-lr0.02/train.jsonl`](../runs/muon-sweep-lr0.02/train.jsonl) and
[`runs/adamw-sweep-control/train.jsonl`](../runs/adamw-sweep-control/train.jsonl).

The selected `0.02` arm was then run under the actual 250M-token schedule and
stopped at 110M tokens:

| Tokens | AdamW loss | Muon 0.02 loss | Muon − AdamW |
|---:|---:|---:|---:|
| 20M | 4.6953 | **3.9164** | −0.7789 |
| 50M | 3.3059 | **3.2112** | −0.0947 |
| 60M | 3.1503 | **3.1449** | −0.0054 |
| 70M | **3.0388** | 3.0788 | +0.0400 |
| 110M | **2.7469** | 2.8679 | +0.1209 |

Muon reaches loss 3.5 in about 15.4 minutes versus AdamW’s 19.6 minutes, but
AdamW reaches loss 2.9 in about 39.3 minutes versus Muon’s 49.2 minutes. See
[`runs/muon-250m-lr0.02-ABORTED/train.jsonl`](../runs/muon-250m-lr0.02-ABORTED/train.jsonl),
[`runs/modern-145m/train.jsonl`](../runs/modern-145m/train.jsonl), and the
[`time_to_loss.py`](../scripts/time_to_loss.py) harness.

## Completed LR probes (120M tokens)

The `0.005` and `0.01` probes have since finished. Both ran the real 250M
schedule (2000-update warmup, `min_lr_ratio=0.1`, seed 2026, identical data
order) and stopped at 120M tokens. All three arms reach 120M at
`optimizer_step=3663`, so this is a matched-token and matched-step comparison.

| Arm | Peak Muon LR | Held-out loss @120M | Perplexity | vs AdamW |
|---|---:|---:|---:|---:|
| AdamW control | — | 2.6999 | 14.879 | — |
| **Muon** | **0.005** | **2.6753** | **14.516** | **−0.0246** |
| Muon | 0.010 | 2.7628 | 15.845 | +0.0629 |
| Muon | 0.020 | (aborted at 110M) | — | +0.1209 |

Loss is monotone in Muon LR over the probed range — 0.02 → 0.01 → 0.005
improves steadily — so the crossover documented above is an artifact of an
excessive learning rate, not an intrinsic property of the optimizer. At `0.005`
Muon ends ahead of AdamW.

Two things keep this from being a decision:

- **The margin is inside the throughput tax.** Muon's 0.0246-nat advantage is
  earned at roughly 3.3% lower tokens/s. The `0.005` arm took 3467s to reach
  120M against AdamW's 3337s, so on wall-clock the two arms are close to a wash.
- **The 25M sweep's LR ranking is inverted.** The sweep selected `0.02` and
  ranked `0.005` worst; at 120M on the real schedule that ordering is exactly
  reversed. The sweep used a shorter warmup, so its LR ranking does not transfer
  to the full schedule. This is the more useful methodological finding: screen
  Muon LR under the schedule the recipe will actually run.

The weight-decay confound described below is still unfixed in these probes — all
three arms share `weight_decay=0.1` while their Muon LRs differ by 4x, so
lowering the LR also lowered the effective decay. The `0.005` result is
therefore a joint LR-and-decay effect, not a clean LR measurement.

### Benchmark scores at 120M tokens

The three matched checkpoints were scored on the full benchmark suite with the
unchanged harness (greedy, 32 new tokens). Scores are recomputed from the
per-example `.jsonl` records; see the evaluator note under
[Reproducibility](#reproducibility-and-resume-behavior).

| Arm | Loss | Correct | Accuracy | Numeric completion |
|---|---:|---:|---:|---:|
| AdamW control | 2.6999 | 73 / 5024 | 1.45% | 68.11% |
| **Muon 0.005** | **2.6753** | **93 / 5024** | **1.85%** | 87.74% |
| Muon 0.01 | 2.7628 | 89 / 5024 | 1.77% | **89.65%** |

Paired exact McNemar on accuracy (n = 5023 shared examples):

| Comparison | net | p |
|---|---:|---:|
| AdamW vs Muon 0.005 | +20 | 0.088 |
| AdamW vs Muon 0.01 | +16 | 0.160 |
| Muon 0.01 vs Muon 0.005 | +4 | 0.794 |

**No accuracy comparison reaches significance.** None clears the 0.05 threshold
this project has used for its other paired comparisons, so on task accuracy the
three arms are not separable at 120M tokens.

Numeric completion rate separates the arms decisively, and this is where the
result becomes interesting:

| Comparison | net | p |
|---|---:|---:|
| AdamW vs Muon 0.005 | +986 | 1.4e-162 |
| AdamW vs Muon 0.01 | +1082 | 2.5e-195 |
| Muon 0.01 vs Muon 0.005 | −96 | 5.9e-04 |

Both Muon arms emit a parseable numeric answer far more often than AdamW, and
the effect holds on every benchmark independently (+16 to +40 points, largest on
arithmetic). But **completion rate does not track held-out loss**: Muon `0.01`
has the worst loss of the three arms and the *highest* completion rate,
significantly above Muon `0.005`. The ordering by loss is 0.005 < AdamW < 0.01;
the ordering by completion rate is AdamW ≪ 0.005 < 0.01. These are measuring
different things, and the completion-rate difference is an optimizer effect, not
a consequence of the loss advantage.

Read this cautiously in both directions. At 120M tokens neither optimizer
produces a model that can do the arithmetic — spot-checking GSM8K completions
shows all arms collapsing into degenerate repetition loops (`$2 per dry duck
eggs = $2 per dry duck eggs = ...`, `$80,000,000,000,...`). A higher completion
rate here means the model more reliably emits *something numeric* before
degenerating; it is a format/output-distribution property, not evidence of
better reasoning, and it is equally consistent with Muon arms simply degenerating
into digit loops more readily than AdamW's word loops. The summary:

- Held-out loss: Muon `0.005` slightly better (−0.0246 nats); Muon `0.01` worse.
- Task accuracy: no arm separable from any other.
- Numeric completion: both Muon arms decisively higher, but uncorrelated with
  loss and not interpretable as capability at this scale.

A benchmark comparison at 120M tokens cannot settle whether Muon produces a
better model. It is reported here because the checkpoints existed and the
comparison was matched; the pre-registered 250M screen remains the decision
point. If the completion-rate gap is real it should be re-checked there, where
the models are past the degenerate-output regime and the metric can mean
something.

## Implementation review

The implementation in [`src/modern_lm/muon.py`](../src/modern_lm/muon.py) follows
the standard hybrid pattern:

- 105 hidden 2D matrices, 119.4M parameters (82.6%), use Muon.
- Embeddings, the output head, norms, and routers use AdamW.
- Tied parameters are deduplicated.
- The compiled model wrapper is unwrapped before name-based routing.
- Newton–Schulz coefficients, momentum, Nesterov behavior, and aspect-ratio
  scaling are consistent with the canonical Muon implementation.

All 66 project tests passed, including the 11 Muon tests. The new tests are good
unit coverage, but they do not yet prove CUDA integration, exact save/resume
next-step equivalence, or parity with `torch.optim.Muon`.

## Important confounds and risks

### Weight decay and schedule horizon

Muon applies decoupled decay as `p *= 1 - lr * weight_decay`. The trainer passes
the same `weight_decay=0.1` to AdamW and Muon, even though the Muon LR is roughly
67 times larger than AdamW’s. The 25M sweep also takes ten times fewer optimizer
steps than the 250M run, so cumulative decay and update dynamics are not
transferable. The sweep tuned Muon LR but not Muon weight decay.

The canonical Muon guidance says both learning rate and weight decay should be
tuned. PyTorch also documents a `match_rms_adamw` adjustment intended to reuse
AdamW LR and decay values:

- [KellerJordan/Muon](https://github.com/KellerJordan/Muon)
- [PyTorch Muon implementation](https://github.com/pytorch/pytorch/blob/main/torch/optim/_muon.py)

### Unequal hyperparameter search

Four Muon LRs were compared with one inherited AdamW setting. That is acceptable
for deciding whether the current AdamW recipe has an attractive replacement, but
not for a general claim that Muon beats a tuned AdamW. The single seed also makes
the result exploratory, consistent with the project’s existing
[`comparison-protocol.md`](comparison-protocol.md).

### Wall-clock accounting

The trainer evaluates and writes both a milestone checkpoint and `latest.pt` at
each evaluation boundary. AdamW checkpoints are about 1.736GB; Muon checkpoints
are about 1.258GB. A fine-grained time-to-loss run can therefore include
optimizer-dependent checkpoint I/O. `elapsed_seconds` is recorded before the
current evaluation/save and resumed runs omit boundary costs. The current curves
are directionally useful, but small wall-clock differences should not be treated
as precise until evaluation, checkpointing, and training timing are separated.

### Reproducibility and resume behavior

The Muon changes and run artifacts are currently uncommitted. Run metadata does
not record the command, git revision, dirty diff, Torch/CUDA versions, GPU, data
paths, compile setting, or evaluation batch count. Resume also does not validate
checkpoint settings against command-line settings, so changed Muon LR/decay can
silently disagree with loaded optimizer groups.

Two further gaps surfaced while scoring the probes:

- **No environment of its own.** This project has no venv; `python -m
  modern_lm...` fails outright. Training and scoring only run under the
  reference repo's environment, as
  `PYTHONPATH=src ../llm-deepseek-v4-experiment/.venv/bin/python -m modern_lm...`.
  That is undocumented, and it means the exact Torch build behind every number
  in this note is defined by another repository's uncommitted venv.
- **The evaluator corrupts its own summary.** `evaluate_benchmarks` writes its
  progress stream and its final summary to the same descriptor, so a redirected
  `.summary.json` ends up with the summary overwriting the head of the progress
  output, separated by NUL padding. The per-example `.jsonl` is unaffected and is
  the authoritative record; scores in this note are recomputed from it. Progress
  events should go to stderr.

## Recommended next experiment

1. ~~Let the existing `0.005` probe finish; use it only as a diagnostic.~~ Done;
   see [Completed LR probes](#completed-lr-probes-120m-tokens). It is a
   diagnostic, not a decision: Muon `0.005` wins by 0.0246 nats at 120M, which
   the throughput tax roughly cancels.
2. Add separate Muon weight decay and log actual AdamW and Muon LRs per update.
   The probes cannot separate LR from decay, so this now blocks interpretation
   rather than merely improving it.
3. Separate evaluation cadence from checkpoint cadence and record compute-only and
   end-to-end elapsed time.
4. Screen under the real 250M schedule to 120M tokens:
   - fresh AdamW control (`3e-4`, `0.1`);
   - Muon `0.005` with Muon decay held at `0.1` (reproduces the current
     best arm and pins the decay confound);
   - Muon `0.005` with Muon decay `0.01`;
   - RMS-matched Muon using AdamW-scale LR/decay.
   Extend the LR probe downward — `0.0025` — since `0.005` is the edge of the
   probed range and loss is still improving as LR falls.
5. Pre-register time-to-loss and loss at 120M. Require a margin larger than the
   measured ~3.3% Muon throughput tax before funding a full 250M confirmation.
   The current best arm does not clear this bar.
6. Confirm the selected recipe with at least one additional seed before making a
   general optimizer claim.

