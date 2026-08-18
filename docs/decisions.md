# Decision ledger

Last reconciled: **2026-08-18**

This is the authoritative, append-only record of project decisions. Result notes
and protocols remain historical evidence; when they disagree with this ledger,
the newest non-superseded decision here controls current work.

## How to use this ledger

- Give each decision a stable, monotonically increasing ID.
- Record the decision, why it was made, evidence, consequences, and anything it
  supersedes.
- Do not rewrite an old decision to make history look cleaner. Add a new decision,
  point its `Supersedes` field at the old ID, and update the index status.
- `Accepted` means the policy is current. It does not imply all implementation
  work is complete; that is stated explicitly in the entry.
- `Provisional` means it is the current operational choice but its causal claim is
  not established.
- `Planned`, `deferred`, and `rejected` describe work disposition, not whether the
  experiment was worth doing.

## Current index

| ID | Date | Status | Decision |
|---|---|---|---|
| [D001](#d001) | 2026-08-16 | Accepted | Reframe the repository as an optimization testbed |
| [D002](#d002) | 2026-08-16 | Accepted | One canonical trajectory; no routine multiple-seed gate |
| [D003](#d003) | 2026-08-16 | Accepted | Separate validation lanes by change semantics |
| [D004](#d004) | 2026-08-16 | Accepted; implementation pending | Split discovery evaluation from sealed confirmation |
| [D005](#d005) | 2026-08-03 | Accepted | Dense single-stream Pre-RMSNorm is the canonical family |
| [D006](#d006) | 2026-08-03 | Accepted | RMSNorm residual layout and scaled residual initialization |
| [D007](#d007) | 2026-08-03 | Accepted | RoPE, QK-norm, causal SDPA, full MHA by default |
| [D008](#d008) | 2026-08-03 | Accepted | Dense SwiGLU feed-forward |
| [D009](#d009) | 2026-08-03 | Accepted | Bias-free, untied embedding and output head |
| [D010](#d010) | 2026-08-11 | Accepted | Optional parity-tested KV cache |
| [D011](#d011) | 2026-08-03 | Accepted | MTP and MoE remain disabled experimental branches |
| [D012](#d012) | 2026-08-08 | Provisional | Hybrid Muon/AdamW is the working pretraining optimizer |
| [D013](#d013) | 2026-08-16 | Accepted | Cosine remains canonical; 50M WSD is rejected |
| [D014](#d014) | 2026-08-16 | Superseded by D026 | 300M-body 3.45B checkpoint is capability champion |
| [D015](#d015) | 2026-08-16 | Accepted | Current SFT recipe and token-matched comparison policy |
| [D016](#d016) | 2026-08-16 | Accepted; implementation pending | Replace body-only efficiency accounting |
| [D017](#d017) | 2026-08-16 | Superseded by D033 | Defer low precision until a fused supported path exists |
| [D018](#d018) | 2026-08-16 | Accepted | Relabel current Siamese arm as a local HybridNorm variant |
| [D019](#d019) | 2026-08-16 | Superseded by [D028](#d028) | Fuse QKV and SwiGLU input projections next |
| [D020](#d020) | 2026-08-16 | Accepted; implementation pending | Immutable provenance and honest timing are required |
| [D021](#d021) | 2026-08-16 | Accepted | Preserve and narrowly scope negative results |
| [D022](#d022) | 2026-08-16 | Accepted | Maintain shared human and machine-readable project memory |
| [D023](#d023) | 2026-08-18 | Accepted | Sync-free metric collection and segment-attributed timing |
| [D024](#d024) | 2026-08-18 | Accepted | Microbatch 64 x accumulation 1 is the default batch shape |
| [D025](#d025) | 2026-08-18 | Accepted | Evaluate short SFT across seeds and checkpoint grids |
| [D026](#d026) | 2026-08-18 | Provisional | 5.28B/seed-2031 update 1,000 is the best observed development checkpoint |
| [D027](#d027) | 2026-08-18 | Accepted | Keep 64x1 after post-sync-cleanup throughput validation |
| [D028](#d028) | 2026-08-18 | Superseded by D031 | Fused QKV and gate/up projections, with block-aware Muon |
| [D029](#d029) | 2026-08-18 | Accepted | Muon's bf16 Newton-Schulz makes trajectories kernel-sensitive |
| [D030](#d030) | 2026-08-18 | Superseded by D032 | Chunked vocabulary cross-entropy |
| [D031](#d031) | 2026-08-18 | Rejected | Keep projection fusion off after a null compiled-throughput result |
| [D032](#d032) | 2026-08-18 | Accepted | Keep chunked cross-entropy as a memory-only opt-in |
| [D033](#d033) | 2026-08-18 | Accepted | Expose functional Transformer Engine FP8/NVFP4 modes; keep BF16 default |

<a id="d001"></a>
## D001 — Reframe the repository as an optimization testbed

- **Date:** 2026-08-16
- **Status:** Accepted
- **Scope:** Project mission

**Decision:** The original capacity-matched comparison becomes historical Phase I.
The current mission is to incorporate new architecture, optimizer, data, numerical,
and systems findings and retain only changes that improve the capability-cost
Pareto frontier under controlled measurement.

**Why:** The repository already spans SFT composition, GRPO, Muon, continued
pretraining, scale ladders, schedules, batch shape, low precision, caching, and
normalization variants. Continuing to describe it as a single architecture
comparison obscures the actual work and encourages incompatible evidence standards.

**Consequences:** Living documentation distinguishes current state from historical
results. Candidate status progresses through proposed, implemented, unit-validated,
systems-validated, capability-validated, transferred, and adopted/rejected.

**Evidence:** Repository history from 2026-08-03 through 2026-08-16 and the
experiment documents linked from the README.

<a id="d002"></a>
## D002 — One canonical trajectory; no routine multiple-seed gate

- **Date:** 2026-08-16
- **Status:** Accepted
- **Scope:** Experiment policy

**Decision:** Use seed 2026 and the canonical data order for pretraining screens,
and the established stage seed for checkpoint-forked SFT. Do not require multiple
seeds for routine screening or adoption.

Prefer stronger evidence per unit of compute:

1. paired forks from the same checkpoint where possible;
2. a predeclared minimum useful effect;
3. a gap sustained across several checkpoints or summarized as area under the
   learning curve/time-to-capability;
4. transfer to another size or token budget; and
5. one sealed confirmation after selection.

Additional seeds are optional for publication-grade generalization or a genuinely
borderline, high-impact decision.

**Why:** Full pretraining runs contain enough tokens and optimizer updates that
replicating every weak idea would consume the budget needed to explore or transfer
strong ideas. Large training budgets do not mathematically remove initialization
or ordering effects, so claims must remain scoped to the canonical trajectory.

**Consequences:** Say “validated on the canonical trajectory and hardware,” not
“expected improvement across random runs.” Historical documents that request a
second seed are superseded on experiment policy by this decision.

**Supersedes:** The routine additional-seed recommendations in
[`comparison-protocol.md`](comparison-protocol.md),
[`results-muon.md`](results-muon.md), and
[`siamese-norm-protocol.md`](siamese-norm-protocol.md).

<a id="d003"></a>
## D003 — Separate validation lanes by change semantics

- **Date:** 2026-08-16
- **Status:** Accepted
- **Scope:** Promotion gates

**Decision:** Classify every candidate before implementation:

| Lane | Required evidence |
|---|---|
| Semantics-preserving systems | Output, loss, gradient, optimizer-step, and resume parity; then throughput, memory, energy, prefill, and decode |
| Approximate numerical | Declared tolerance, short-run divergence, convergence and capability trajectory |
| Learning/architecture/data | Fixed compute contract, sustained time/tokens-to-capability, regression suite, sealed confirmation |
| Evaluation | Versioned scorer and all compared baselines rescored |

**Why:** Kernel fusion and a new optimizer cannot share an evidence standard.
Expecting bit parity from a learning change is meaningless; accepting a faster
kernel without parity is unsafe.

**Consequences:** Every protocol and decision must name its lane.

<a id="d004"></a>
## D004 — Split discovery evaluation from sealed confirmation

- **Date:** 2026-08-16
- **Status:** Accepted; implementation pending
- **Scope:** Capability evaluation

**Decision:** Treat the existing 5,024-question numeric suite as a development
benchmark because its errors and examples have repeatedly informed training-data
and probe design. Create a separate sealed confirmation set and use it only after a
candidate is selected under a preregistered gate.

**Why:** The existing workflow legitimately found useful failure modes, but adaptive
inspection means the same suite cannot independently confirm the resulting fixes.
Item-level significance tests do not correct adaptive test-set use.

**Consequences:** Existing scores remain useful development metrics. They are not
generalization estimates. `scripts/probe3.py` is diagnostic only and must not drive
a confirmatory claim.

**Evidence:** The SFT augmentation sequence in the README and benchmark sampling in
`scripts/probe3.py`.

<a id="d005"></a>
## D005 — Dense single-stream Pre-RMSNorm is the canonical family

- **Date:** 2026-08-03
- **Status:** Accepted
- **Scope:** Model topology

**Decision:** The canonical architecture is a decoder-only, dense Transformer with
one residual stream and two sequential pre-normalized sublayers per block:

```text
x <- x + Attention(RMSNorm(x))
x <- x + SwiGLU(RMSNorm(x))
```

**Why:** It is the most validated path in this repository, is simple enough for
controlled ablations, and produced the historical quality/time win at matched
stored parameters.

**Consequences:** New residual topologies are separate architecture arms until they
beat this path on capability-cost evidence. They do not silently become defaults.

**Evidence:** [`comparison-protocol.md`](comparison-protocol.md),
[`results.md`](results.md), and `src/modern_lm/layers.py::Block`.

<a id="d006"></a>
## D006 — RMSNorm residual layout and scaled residual initialization

- **Date:** 2026-08-03
- **Status:** Accepted
- **Scope:** Normalization, precision, initialization

**Decision:** RMSNorm has learned scale only, computes normalization in fp32, and
casts back to the input dtype. Attention and feed-forward are pre-normalized. All
weights initialize at standard deviation 0.02; residual-output projections
`o_proj` and `down_proj` use an additional `1/sqrt(2L)` scale.

**Why:** The dtype cast avoids accidental fp32 propagation under bf16 autocast,
while scaled residual outputs control variance growth with depth.

**Consequences:** Spin-offs must preserve the cast and initialization ordering if
they want checkpoint or trajectory comparability.

**Evidence:** `src/modern_lm/layers.py::RMSNorm` and
`src/modern_lm/model.py::_init_weights`.

<a id="d007"></a>
## D007 — RoPE, QK-norm, causal SDPA, full MHA by default

- **Date:** 2026-08-03
- **Status:** Accepted
- **Scope:** Attention

**Decision:** Use bias-free Q/K/V/O projections; per-head RMSNorm on Q and K;
fp32 RoPE with theta 10,000 applied after QK-norm; and PyTorch causal scaled-dot-
product attention. `n_kv_heads` is configurable, but accepted profiles currently
set it equal to `n_heads` (full MHA).

**Why:** This is the tested stability and positional stack. GQA is retained for
inference-memory experiments but would alter capacity relative to current profiles.

**Consequences:** Preserve the order `project → QK-norm → RoPE → optional K/V
cache/repeat → SDPA → output projection`. Changing it is an architecture change.

**Evidence:** `src/modern_lm/layers.py::Attention`.

<a id="d008"></a>
## D008 — Dense SwiGLU feed-forward

- **Date:** 2026-08-03
- **Status:** Accepted
- **Scope:** Feed-forward network

**Decision:** Use `down(silu(gate(x)) * up(x))` with three bias-free matrices and
profile-specific `ffn_dim` near 3.2–3.4 times model width.

**Why:** It is the validated dense feed-forward path and keeps the block easy to
profile and ablate.

**Consequences:** Gate and up are semantically fusible but are currently separate
linears; see [D019](#d019).

**Evidence:** `src/modern_lm/layers.py::SwiGLU`.

<a id="d009"></a>
## D009 — Bias-free, untied embedding and output head

- **Date:** 2026-08-03
- **Status:** Accepted
- **Scope:** Input/output layers

**Decision:** Use a 16,384-entry token embedding and a separate, bias-free linear
vocabulary head. Weight tying remains configurable but is off in accepted profiles.

**Why:** Untied weights were part of the capacity-matched baseline and every current
champion checkpoint. Changing them alters both stored capacity and optimization.

**Consequences:** Count the output head as compute-bearing: it performs a dense
`[tokens, dim] × [vocab, dim]^T` projection. See [D016](#d016).

**Evidence:** `src/modern_lm/model.py::ModernLM`.

<a id="d010"></a>
## D010 — Optional parity-tested KV cache

- **Date:** 2026-08-11
- **Status:** Accepted
- **Scope:** Inference

**Decision:** Keep greedy generation uncached by default when comparing wall clock
to the historical reference. Enable the per-layer pre-repeat K/V cache when only
model output or practical inference speed matters.

**Why:** Cached and uncached greedy outputs are test-validated as identical, while
the cache makes evaluation roughly four times faster. Keeping it optional preserves
the original wall-clock comparison contract.

**Consequences:** Spin-offs should normally enable caching. GQA caches pre-repeat K/V
to retain its memory benefit. At maximum context, generation falls back to the
uncached sliding-prefix path.

**Evidence:** Commit `4685e13`, `src/modern_lm/model.py::generate`, and cache tests.

<a id="d011"></a>
## D011 — MTP and MoE remain disabled experimental branches

- **Date:** 2026-08-03
- **Status:** Accepted
- **Scope:** Optional architecture branches

**Decision:** `use_mtp=false` and `use_moe=false` in the accepted architecture.

**Why:** MTP costs about 15% throughput in the measured implementation. MoE costs
about 23%, uses a correctness-oriented Python expert loop, and changes the block
state dict. Neither has demonstrated a capability-cost improvement.

**Consequences:** Their code remains for experiments, but neither belongs in a
spin-off baseline or “current best” diagram except as a disabled branch.

**Evidence:** Historical architecture benchmarks and `src/modern_lm/layers.py::MoE`.

<a id="d012"></a>
## D012 — Hybrid Muon/AdamW is the working pretraining optimizer

- **Date:** 2026-08-08
- **Status:** Provisional
- **Scope:** Pretraining optimizer

**Decision:** The size ladder operationally uses Muon at learning rate 0.005 for
hidden two-dimensional matrices and AdamW at 3e-4 for embeddings, the vocabulary
head, norms, and other parameters.

**Why:** Muon improved the 250M loss trajectory and time-to-AdamW-loss, which made it
useful for exploration. It did not improve 250M benchmark capability, its advantage
did not persist cleanly to 2B, and weight decay/update scale remain confounded.

**Consequences:** Describe this as a working recipe, not “Muon is better.” Before a
clean optimizer claim, compare update-RMS-matched recipes with Muon decay separated.
Use the canonical trajectory policy in [D002](#d002), not routine seed replication.

**Evidence:** [`results-muon.md`](results-muon.md) and size-ladder run metadata.

<a id="d013"></a>
## D013 — Cosine remains canonical; 50M WSD is rejected

- **Date:** 2026-08-16
- **Status:** Accepted
- **Scope:** Learning-rate schedule

**Decision:** Continue using warmup plus cosine decay for canonical runs. Do not
promote WSD based on literature alone.

**Why:** At the controlled 50M rung, WSD ended with worse held-out loss (2.311 versus
2.297) and worse post-SFT development capability (459 versus 474).

**Consequences:** The WSD implementation remains available for future regimes, but
its current status is a scoped negative at 50M, Muon, and 20 body-tokens/parameter.

**Evidence:** `runs/size50m-20x-wsd/latest.json`,
`runs/eval-sft-50m-wsd.summary.json`, and
`runs/eval-sft-50m-cosine.summary.json`.

<a id="d014"></a>
## D014 — 300M-body 3.45B checkpoint is capability champion

- **Date:** 2026-08-16
- **Status:** Provisional
- **Scope:** Current checkpoint selection

**Decision:** Record the dense 300M-body profile at 3,450,011,648 pretraining tokens
plus 1,000 standard `sft-math-words` updates as the current capability champion.

**Why:** It scores 718/5,024 (14.291%), the highest recorded comparable development
score in the repository.

**Consequences:** This selects a checkpoint, not a universal recipe. The parent run
is incomplete, changed batch shape during training, and was selected on an
adaptively used development benchmark. Do not call it sealed-test performance.

**Evidence:** `runs/size300m-20x/checkpoint-003450011648.json`,
`runs/sft-300m-3450M/latest.json`, and
`runs/eval-sft-300m-3450M.summary.json`.

<a id="d015"></a>
## D015 — Current SFT recipe and token-matched comparison policy

- **Date:** 2026-08-16
- **Status:** Accepted
- **Scope:** Supervised fine-tuning

**Decision:** Keep concise answer-shaped supervision with arithmetic and number-word
coverage as the working SFT baseline. Use AdamW at 5e-5, 100 warmup updates, and
1,000 planned updates unless a protocol states otherwise. Compare future data arms
at matched supervised-token budgets and report wall clock.

**Why:** Concise targets and targeted coverage produced the strongest 145M gains.
However, later “matched” word-heavy arms held example count approximately constant
while processing up to 3.4 times as many supervised tokens, so those are not clean
composition-only comparisons.

**Consequences:** Example count and update count remain diagnostics, not the compute
contract. Keep SFT as a separate stage in architecture manifests.

**Evidence:** [`results-sft-concise.md`](results-sft-concise.md),
[`results-sft-augmented.md`](results-sft-augmented.md),
[`results-sft-number-words.md`](results-sft-number-words.md), and run metadata.

<a id="d016"></a>
## D016 — Replace body-only efficiency accounting

- **Date:** 2026-08-16
- **Status:** Accepted; implementation pending
- **Scope:** Scaling and efficiency metrics

**Decision:** Report at least total stored parameters, non-embedding compute-bearing
parameters, and estimated forward/backward FLOPs. Retain “body parameters” only as
a historical/profile label, never as the sole compute or tokens-per-parameter axis.

**Why:** `scripts/bench_throughput.py` excludes both embedding and `lm_head`, but the
untied vocabulary head is a large dense matmul. At the 5M-body rung it contains
4,194,304 parameters and materially changes the small-end scale relationship.

**Consequences:** Recalculate size-ladder plots and token ratios before making a
scaling-law claim. Existing raw runs remain valid; their normalized x-axis needs
correction.

**Evidence:** `scripts/bench_throughput.py::body_params` and [D009](#d009).

<a id="d017"></a>
## D017 — Defer low precision until a fused supported path exists

- **Date:** 2026-08-16
- **Status:** Deferred
- **Scope:** Numerical/systems optimization

**Decision:** Do not spend a full training run on the current FP8 or NVFP4 path.
Reconsider low precision only with hardware-supported custom/fused kernels and the
quantization recipe required by that implementation.

**Why:** Current end-to-end FP8 measured 0.96–0.99× forward and roughly 0.76× with
backward; the raw NVFP4 ceiling was also unattractive at tested shapes. Existing
shape scripts additionally model fused QKV/gate-up operations that the current
network does not execute.

**Consequences:** Preserve the negative as “current implementation on GB10,” not a
general rejection of FP8/NVFP4.

**Evidence:** Commits `81764c2` and `e0be6a1`, plus the FP8/NVFP4 benchmark scripts.

<a id="d018"></a>
## D018 — Relabel current Siamese arm as a local HybridNorm variant

- **Date:** 2026-08-16
- **Status:** Accepted
- **Scope:** Architecture fidelity

**Decision:** Preserve the running 50M two-stream result under the name
`siamese-local-hybrid`. Do not present it as a faithful validation of published
SiameseNorm.

**Why:** The local code computes one combined block update, scales it by
`1/sqrt(layer+1)`, and fuses `X + norm(Y)` before the head. The current reference
implementation performs separate attention and feed-forward stream updates, uses
different depth scaling, and has additional input/final normalization.

**Consequences:** The arm can still answer whether this local topology helps at 50M
with Muon. A paper-level claim requires a separate faithful path and equation-level
tests against the reference implementation. Under [D002](#d002), a promising
result transfers to another rung rather than automatically triggering another seed.

**Evidence:** `src/modern_lm/layers.py::Block`, `src/modern_lm/model.py::ModernLM`,
[`siamese-norm-protocol.md`](siamese-norm-protocol.md), and the published reference
linked there.

<a id="d019"></a>
## D019 — Fuse QKV and SwiGLU input projections next

- **Date:** 2026-08-16
- **Status:** Superseded by [D028](#d028), which implements it
- **Original status:** Planned
- **Scope:** Semantics-preserving systems optimization

**Decision:** Make fused QKV and fused gate/up projections the next model-side
efficiency candidates. Include a checkpoint converter so existing models remain
usable.

**Why:** The current implementation launches three attention input linears and two
feed-forward input linears, while recent low-precision shape benchmarks already
assume fused matrices. Fusion should reduce launches and make microbenchmarks match
production.

**Consequences:** This is a [D003](#d003) semantics-preserving change. It cannot be
promoted without eager/compiled forward, loss, gradient, optimizer-step, generation,
cache, and resume/checkpoint parity tests.

**Evidence:** `src/modern_lm/layers.py`, `scripts/bench_fp8_shapes.py`, and
`scripts/bench_nvfp4.py`.

<a id="d020"></a>
## D020 — Immutable provenance and honest timing are required

- **Date:** 2026-08-16
- **Status:** Accepted; implementation pending
- **Scope:** Reproducibility and performance measurement

**Decision:** Each run and evaluation must record an immutable manifest containing
the full command, config, commit and dirty-diff identity, data/tokenizer hashes,
scorer and decode settings, dependency/runtime versions, hardware, and interventions.
Report segment training throughput separately from compile, evaluation, checkpoint,
and end-to-end time.

**Why:** Current checkpoints omit key provenance; mid-run setting changes are not
canonical events. `compute_loss` also converts device tensors to Python floats every
microbatch, adding synchronization that confounds batch-shape timing. Cumulative
throughput mixes work and boundary overhead.

**Consequences:** An intervention creates a new run identity or immutable event.
Performance claims remain provisional until instrumentation overhead is removed.

**Evidence:** `src/modern_lm/train.py`, `src/modern_lm/sft.py`, and current checkpoint
metadata.

<a id="d021"></a>
## D021 — Preserve and narrowly scope negative results

- **Date:** 2026-08-16
- **Status:** Accepted
- **Scope:** Reporting

**Decision:** Keep failed and null experiments when their implementation and protocol
are understood. State exactly what was rejected: implementation, hardware, model
shape, optimizer, data, and token budget.

**Why:** GRPO's zero-gradient diagnosis, CPT's capability collapse despite recovered
loss, WSD's failed transfer, and current low-precision slowdowns prevent repeated
dead ends. Overgeneralizing them would block better implementations of the same
ideas.

**Consequences:** “Rejected” never means “the research area cannot work.” It means
the tested point did not improve the project's capability-cost frontier.

**Evidence:** [`results-grpo.md`](results-grpo.md),
[`cpt-8b-gate-result.md`](cpt-8b-gate-result.md), [D013](#d013), and [D017](#d017).

<a id="d022"></a>
## D022 — Maintain shared human and machine-readable project memory

- **Date:** 2026-08-16
- **Status:** Accepted
- **Scope:** Project continuity and spin-offs

**Decision:** Maintain four synchronized sources: `AGENTS.md` for operating rules,
`PROJECT_MEMORY.md` for current state, this append-only ledger for why choices were
made, and `architecture.md` plus `architecture.json` for the human and structured
architecture contract.

**Why:** The repository's trajectory changes faster than its historical narrative.
New work and spin-off projects need a compact statement of current truth without
reverse-engineering commits, run directories, and dated result notes.

**Consequences:** Architecture changes update both representations and pass
`tests/test_architecture_manifest.py`. Decisions are appended and superseded rather
than overwritten. Current run state is dated and points to authoritative artifacts.

**Evidence:** The shared-memory files introduced with this decision.

<a id="d023"></a>
## D023 — Sync-free metric collection and segment-attributed timing

- **Date:** 2026-08-18
- **Status:** Accepted
- **Scope:** Reproducibility and performance measurement
- **Implements:** [D020](#d020), items 3 and "honest timing"

**Decision:** Loss components stay on the device until a logging boundary reads
them, and pretraining wall clock is attributed to disjoint segments — `setup`,
`compile_and_warmup`, `data`, `step`, `evaluation`, `checkpoint` — carried across
resumes. `training_tokens_per_second` divides tokens measured in the training
segments by the time in those segments, and is the number to quote.
`tokens_per_second` remains end-to-end for comparability with historical logs.
Optional `--profile-every N` emits one synchronized data/forward/backward/optimizer
breakdown per N updates. Utilization is reported as achieved TFLOP/s always, and as
MFU only when the operator declares `--device-peak-tflops`.

**Why:** `compute_loss` converted two to four device scalars to Python floats on
every microbatch. Each conversion drains the queue and blocks the CPU until the GPU
catches up, so the instrumentation was charging its cost to the thing it measured —
and unevenly: at accumulation 4 a 16x4 update paid four times the stalls of a 64x1
update, which is a confound in the very comparison that motivated [D024](#d024).
Separately, the logged rate was `tokens_seen / elapsed_seconds` since run start, so
compilation, every held-out evaluation, and every checkpoint write were counted as
training time. A run that evaluated more often reported lower "throughput" at
identical GPU speed.

**Consequences:** Segment-aware records are not comparable field-for-field with
pre-2026-08-18 logs: an old `tokens_per_second` sits between the new end-to-end and
training-only rates. Compare new runs on `training_tokens_per_second`, and treat
every throughput number recorded before this date as end-to-end. Two things this
does not fix and that remain open: `sft.py` still converts per example and also
syncs on its `isfinite` guard and its supervised-token count, and no `mfu` will
appear in any log until a measured device peak is supplied — deliberately, since a
guessed constant would rescale every derived number invisibly.

**Evidence:** `src/modern_lm/perf.py`, `src/modern_lm/train.py::compute_loss`,
`src/modern_lm/train.py::train`, and `tests/test_perf.py`.

<a id="d024"></a>
## D024 — Microbatch 64 x accumulation 1 is the default batch shape

- **Date:** 2026-08-18
- **Status:** Accepted
- **Scope:** Training systems

**Decision:** The trainer defaults to `microbatch_size=64, gradient_accumulation=1`.
The token budget is unchanged at 32,768 per optimizer update. Runs whose model does
not fit — above roughly 600M body parameters — pass an explicit shape. Scripts that
record completed runs keep the shape those runs used and are not retrofitted.

**Why:** 16x4 entered the repository as a pinned control in the DeepSeek-V4
comparison protocol, where the point was that architecture be the only variable. It
was never a tuning result, and every run since inherited it. The two shapes are
mathematically identical — token-weighted accumulation makes the gradient the same —
but 16x4 executes four passes over 8,192-row GEMMs instead of one over 32,768 rows,
paying four times the fixed per-pass cost. `bench_batch_shape.py` measured 1.09x at
50M and 300M, 1.06x at 100M and 1B, and 1.04x at 600M, compiled. Eager measures
0.96-1.00x: the accumulation loop hides inside per-kernel overhead there, which is
why this went unnoticed. Consistent with Marek et al. (arXiv:2507.07101), gradient
accumulation buys nothing on a single GPU.

**Consequences:** Peak memory rises — 40.2GB at 300M, 58.7GB at 600M, 86.9GB at 1B
of a 121GB pool shared with everything else on the box — so the 1B rung keeps 16x4
or 32x2 and 600M keeps 32x2. The 300M resume changes shape mid-trajectory; that is a
recorded intervention (below), and it adds to the reasons [D014](#d014) holds the
champion at provisional. The measured 1.09x is an UPPER BOUND: it was taken before
[D023](#d023) removed the per-microbatch syncs that penalized 16x4 four times per
update, so some of the gain was stalls rather than GEMM shape. `--profile-every 200`
on the 300M resume is what will separate the two.

To keep mid-run changes visible, a resume now diffs the checkpoint's settings
sidecar against its own flags and writes the difference into `train.jsonl` as the
`run_identity` record's `interventions` field. A run that changes shape, learning
rate, or schedule mid-trajectory therefore carries that fact in its own log rather
than in someone's memory of which flags were typed.

**Evidence:** `scripts/bench_batch_shape.py`, commit `f528c92`,
`src/modern_lm/train.py::TrainSettings`, `scripts/resume_300m_20x.sh`, and
`tests/test_perf.py::test_resume_records_a_changed_batch_shape_as_an_intervention`.

<a id="d025"></a>
## D025 — Evaluate short SFT across seeds and checkpoint grids

- **Date:** 2026-08-18
- **Status:** Accepted
- **Scope:** Evaluation

**Decision:** For the current short SFT stage, evaluate two declared seeds at a
fixed checkpoint grid rather than reading only update 1,000. Report the full grid,
the per-checkpoint mean and spread, and the best observed checkpoint separately.
The best is a discovery/champion statistic, not an estimate of the arm's expected
score; comparisons must use equal seed and checkpoint search budgets. Escalating to
more seeds is a targeted diagnostic when instability blocks interpretation, not a
routine pretraining gate.

**Why:** A single seed at update 1,000 is one draw from a wide distribution. The
completed diagnostic used five SFT seeds on one identical 300M base
(5,280,006,144 pretraining tokens, same corpus and recipe) at updates 600, 800,
and 1,000. The 15 readings span 650–887 of 5,024. At update 1,000 the scores are
697, 791, 818, 801, and 887: mean 798.8, sample standard deviation 68.1. Seed 2027
alone moved 650 → 822 → 697, while all three added seeds peaked at update 1,000.
The apparent late-schedule regression therefore did not replicate; it was an
SFT-seed/checkpoint interaction, not evidence that cosine pretraining lost capacity.

The mechanism is that this benchmark is a large set of near-threshold binary
outcomes. The observed regressions are operand misreads (`46 - 22` written as
`46 - 32`), not arithmetic failures, so hundreds of shallow asdiv/svamp items flip
in bulk on small weight changes. That is also why held-out SFT loss fell
monotonically across arms while accuracy bounced: loss averages 9,803 supervised
tokens and is stable, accuracy does not.

**Consequences:** Held-out SFT loss must not select a checkpoint or stand in for
capability. Differences smaller than roughly 80 questions (~1.6 points) from one
seed/update are not reportable. Best-of-grid scores must be labeled selection-biased.
The size ladder's effects remain much larger; [D002](#d002)'s one-trajectory policy
still controls pretraining. This exception is scoped to the 1,000-update SFT stage
over 23,780 examples, where the observed variation is empirical rather than assumed.

**Evidence:** `runs/eval-sft-300m-5280M-seed-grid.summary.json`,
`runs/eval-sft-300m-5280M-step*.summary.json`, and
`runs/eval-sft-300m-5280M-seed{2028,2029,2030,2031}-step*.summary.json`;
`runs/sft-seed-timing.tsv`. Caveat: the original seed-2027 final report used eval
batch 32 while the checkpoint probes used 8, so cross-setting aggregates remain
diagnostic rather than a clean estimator.

**Supersedes:** D015's single-endpoint reporting rule and D002 only for targeted
replication of this short SFT stage; the SFT recipe and pretraining seed policy are
unchanged.

<a id="d026"></a>
## D026 — 5.28B/seed-2031 update 1,000 is the best observed development checkpoint

- **Date:** 2026-08-18
- **Status:** Provisional
- **Scope:** Current capability champion

**Decision:** Track the 300M-body checkpoint pretrained for 5,280,006,144 tokens
and SFT-trained with seed 2031 for 1,000 updates as the best *observed* development
checkpoint: 887/5,024 (17.655%). Do not describe it as a validated recipe
improvement or an estimate of expected performance.

**Why:** It exceeds the former 3.45B/seed-2027 champion's 718 by 169 questions, but
it was selected from a 15-cell seed/checkpoint grid on the adaptively used
development suite. The five-seed update-1,000 mean is 798.8 with sample standard
deviation 68.1; 887 is only 59 above the independently observed 4.77B score of 828,
below [D025](#d025)'s approximate single-reading effect floor. The checkpoint is
useful as an artifact and spin-off starting point, not proof that 5.28B pretraining
tokens dominate 4.77B.

**Consequences:** Living memory and architecture manifests point to this artifact
while preserving the full grid, selection caveat, incomplete pretraining trajectory,
and lack of sealed confirmation. Future capability promotion compares distributions
under equal search budgets or uses the sealed set once implemented.

**Evidence:** `runs/size300m-20x/checkpoint-005280006144.json`,
`runs/sft-300m-5280M-seed2031/checkpoint-001000.json`, and
`runs/eval-sft-300m-5280M-seed2031-step001000.summary.json`; full selection context
in `runs/eval-sft-300m-5280M-seed-grid.summary.json`.

**Supersedes:** D014 for champion identity only; its provisionality rationale remains.

<a id="d027"></a>
## D027 — Keep 64x1 after post-sync-cleanup throughput validation

- **Date:** 2026-08-18
- **Status:** Accepted
- **Scope:** Semantics-preserving systems validation

**Decision:** Keep `microbatch_size=64, gradient_accumulation=1` as the single-GPU
default through 300M where memory permits. Treat [D023](#d023)'s sync-free metric
path as measurement correctness, not a direct throughput optimization: it produced
no material steady-state speed change in the matched test. Persist terminal
checkpoint segment timing through the JSON sidecar so resumed rates remain honest.

**Why:** On the NVIDIA GB10, compiled matched-token tests after sync cleanup measured
64x1 over 16x4 at +9.55% for 50M (64,925 vs 59,266 tokens/s) and +9.83% for 300M
(19,307 vs 17,579 tokens/s). Comparing optimized code with commit `ba35390` at the
same batch shape ranged from -0.38% to +0.89%, inside run noise; the batch-shape win
is therefore GEMM/launch efficiency rather than avoided scalar-sync stalls. Peak
allocated memory rises from 5.0 to 16.0GB at 50M and 14.0 to 40.2GB at 300M.

Loss/gradient/AdamW-step equivalence passes for combined versus token-weighted
microbatches. A compiled CUDA smoke test's cumulative training rate matched its
synchronized step profile within 0.1%. That smoke test exposed a resume-accounting
bug: a terminal checkpoint reported its write time at completion but serialized
state from before the write. Post-write sidecar timing plus load-time overlay fixes
the issue without serializing weights twice.

**Consequences:** Remove D024's upper-bound caveat at 50M and 300M. Measurements for
600M and 1B were not repeated here and retain their earlier status and explicit
smaller batch shapes. Quote `training_tokens_per_second` for new training speed and
keep historical `tokens_per_second` labeled end-to-end.

**Evidence:** `docs/results-throughput-2026-08-18.md`, `scripts/bench_batch_shape.py`,
`src/modern_lm/train.py::{compute_loss,refresh_checkpoint_timing}`,
`tests/test_perf.py`, and the full repository test suite.

**Supersedes:** D024's pre-sync-cleanup upper-bound caveat; clarifies D023's
performance mechanism without changing its accounting policy.

<a id="d028"></a>
## D028 — Fused QKV and gate/up projections, with block-aware Muon

- **Date:** 2026-08-18
- **Status:** Implemented; parity-validated, throughput unmeasured
- **Scope:** Semantics-preserving systems optimization
- **Implements:** [D019](#d019)

**Decision:** `ModernConfig.fuse_projections` replaces `q_proj`/`k_proj`/`v_proj`
with one `qkv_proj` and `gate_proj`/`up_proj` with one `gate_up_proj`. Muon
orthogonalizes each original sub-matrix separately via a `row_blocks` param-group
option, so the optimizer step is unchanged. `modern_lm.fusion` and
`scripts/convert_projection_fusion.py` convert existing checkpoints in both
directions, model weights and optimizer state together. The flag defaults to
**off** until the throughput win is measured on this hardware.

**Why:** The block ran five input projections where two suffice. The arithmetic is
identical; what changes is that `x` is read once instead of three times for
attention and once instead of twice for the feed-forward, that five kernel
launches per block per pass become two, and that the GEMMs get a better aspect
ratio for the same work. On a box where every win has come from moving fewer bytes
this is the right shape of change — but Inductor may already fuse some of it, so
the source-level gain is a measurement, not a deduction. `scripts/bench_fusion.py`
is the measurement.

**The part that was not obvious:** orthogonalization is not separable. Newton-Schulz
on a stacked `[3*dim, dim]` matrix does not produce the three orthogonal factors it
would produce on the parts, and Muon's aspect-ratio scale would jump from 1 to
sqrt(3) as well. A straightforward fusion therefore changes the optimizer while
looking like a pure systems change. Measured: naive fusion moves the weights
8.6e-4 relative to the separate-matrix update within three steps. `row_blocks`
makes the fused update **bitwise identical** to the separate one given identical
gradients, and slices back to exactly the shapes the unfused model compiled for, so
no new Dynamo specializations appear either.

**What parity does and does not hold:** with the same weights, forward output and
loss are bitwise identical, and gradients agree to 4e-7 relative — one GEMM
reducing in a different order than three. Under AdamW the trajectories stay
together at 4e-8 relative after five steps. Under **Muon** they separate to 1.7e-3
relative, because bf16 Newton-Schulz amplifies float32-epsilon gradient differences
([D029](#d029)). Fusion is therefore semantics-preserving by construction and by
the AdamW evidence, but on a Muon trajectory it must be validated as an approximate
numerical change under [D003](#d003)'s second lane, not waved through as bit-exact.

**Consequences:** Turning this on mid-run is a numerical intervention, not a free
switch; it is recorded as one by [D024](#d024)'s intervention mechanism. A fresh
fused run initializes to bitwise-identical weights from the same seed (verified),
so fused and unfused arms remain comparable from step zero. Adoption waits on
`bench_fusion.py`; if compiled throughput does not improve, the honest outcome is
to keep the flag off and record that Inductor already had it.

**Evidence:** `src/modern_lm/layers.py`, `src/modern_lm/muon.py::_row_spans`,
`src/modern_lm/fusion.py`, `scripts/bench_fusion.py`, and the nineteen parity tests
in `tests/test_fusion.py`.

<a id="d029"></a>
## D029 — Muon's bf16 Newton-Schulz makes trajectories kernel-sensitive

- **Date:** 2026-08-18
- **Status:** Accepted
- **Scope:** Reproducibility, optimizer

**Decision:** Treat any change that alters GEMM reduction order — projection
fusion, a kernel or library upgrade, a different GPU, a changed batch shape at the
microbatch level — as an approximate numerical change when the run uses Muon, even
when it is provably exact in exact arithmetic. Declare a tolerance and check
trajectory divergence rather than asserting bit-exactness.

**Why:** Measured on the same model with the same seed, fused versus separate
projections: gradients differ by 4e-7 relative, which is float32 rounding. After
five AdamW steps the weights differ by 4e-8 relative — noise stays noise. After
five **Muon** steps they differ by 1.7e-3 relative, four orders of magnitude
larger. `zeropower_via_newtonschulz` casts to bf16, which has roughly three decimal
digits, so a difference in the last bits of a float32 gradient can flip a bf16
rounding and change the orthogonal factor at the 1e-3 level; the update is then
applied at full learning rate.

**Consequences:** [D002](#d002)'s "one canonical deterministic trajectory" is
reproducible only against an identical kernel stack when Muon is in use. This does
not indicate instability — Muon's own results stand, and an orthogonalized update
is meant to be robust to the direction's fine detail — but it does mean bit-exact
reproduction is not an available acceptance test for systems work on a Muon run,
and that two runs differing only in kernels will not agree on a loss curve's last
digits. Prefer AdamW when isolating a numerical question, since it leaves rounding
noise at 1e-8 where Muon amplifies it.

**Evidence:** `tests/test_fusion.py::test_adamw_trajectories_stay_together`,
`::test_full_muon_trajectories_stay_within_the_measured_tolerance`, and
`::test_naive_fusion_would_have_changed_the_optimizer`.

<a id="d030"></a>
## D030 — Chunked vocabulary cross-entropy

- **Date:** 2026-08-18
- **Status:** Implemented; parity-validated, throughput unmeasured
- **Scope:** Semantics-preserving systems optimization

**Decision:** `TrainSettings.chunked_cross_entropy` computes the vocabulary loss a
slice of rows at a time, recomputing each slice's logits during the backward pass
instead of storing them. The model gained `return_hidden`, which skips the head so
the projection happens inside the loss. Default **off** until the throughput is
measured by `scripts/bench_cross_entropy.py`.

**Why:** At the batch shape [D024](#d024) settled on — 32,768 targets per update
against a 16,384-token vocabulary — the logit tensor is 1.07 GB in bf16, autograd
saves it for the backward pass, and its gradient is another 1.07 GB. That is the
largest allocation in the step, roughly 2.1 GB of traffic per micro-batch, for a
quantity reduced to one scalar and discarded. The trade is explicit: the head
projection runs about one and a half times per step instead of once, so FLOPs rise
while bytes fall. On a bandwidth-bound box that should win, but "should" has been
wrong here before — FP8's GEMMs were genuinely faster and quantization ate the
gain ([D017](#d017)) — so it stays off until measured.

**What is verified without a GPU:** in float64 the chunked loss and both gradients
match `F.cross_entropy` to 1e-15, so the arithmetic is the same arithmetic. The
chunk size is not a hyperparameter: sizes from 1 to larger-than-the-batch, including
ones that do not divide it, give the same answer. The memory claim is checked
structurally rather than asserted — `saved_tensors_hooks` confirms the standard path
retains two tensors of tokens × vocabulary and the chunked path retains none.

**On accuracy:** in bf16 the two paths' weight gradients differ by 2.3e-3 relative.
That is not error introduced by chunking. Measured against the same problem solved
in float64, the standard path is 2.368e-3 from truth and the chunked path 2.343e-3
— both sit on bf16's noise floor, and per-chunk GEMMs accumulated in fp32 land a
hair closer. Neither is the wrong answer; they are two samples of the same noise.

**Consequences:** 2.3e-3 is a much larger perturbation than projection fusion's
4e-7 ([D028](#d028)), so under [D029](#d029) this will visibly move a Muon
trajectory. Enabling it mid-run is a numerical intervention and is recorded as one.
The peak-memory result may matter more than the wall clock: if memory drops sharply
while throughput merely holds, the change still buys headroom to raise the
microbatch — worth 4-9% on its own by [D024](#d024) — or to fit a rung that does not
currently fit. Not covered: the MTP head still materializes its own logits, and
`sft.py` still uses the standard path.

**Evidence:** `src/modern_lm/losses.py`, `src/modern_lm/model.py::forward`,
`src/modern_lm/train.py::compute_loss`, `scripts/bench_cross_entropy.py`, and
`tests/test_losses.py`.

<a id="d031"></a>
## D031 — Keep projection fusion off after a null compiled-throughput result

- **Date:** 2026-08-18
- **Status:** Rejected
- **Scope:** Semantics-preserving systems optimization

**Decision:** Keep `fuse_projections=false` in the canonical architecture. Retain
the parity-tested implementation, block-aware Muon routing, and bidirectional
checkpoint converter for portability experiments, but do not describe source-level
QKV/gate-up fusion as an efficiency improvement on the NVIDIA GB10 stack tested
here.

**Why:** Two order-balanced compiled measurements at the production 32,768 targets
per update found no speed or memory win. At 50M, median fused throughput was 65,985
tokens/s versus 66,129.5 separate (0.998x); at 300M it was 19,005 versus 19,168
(0.991x). Peak allocated memory was identical at 16.0GB and 40.2GB respectively.
The first ordering measured 1.002x/1.000x and the reverse measured 0.994x/0.983x,
so the small apparent direction changes with order rather than establishing an
effect. The compiler/runtime evidently already removes enough launch or read
overhead that changing the module layout does not improve the end-to-end step.

**Consequences:** Existing and new canonical runs keep separate projections. No
capability run is warranted for a candidate with no systems gain. A future kernel,
compiler, hardware target, or low-precision path may reopen the implementation,
but must remeasure it. Enabling fusion on a Muon run remains an approximate
numerical intervention under [D029](#d029), not a free mid-run switch.

**Evidence:** [`results-fusion-chunked-ce-2026-08-18.md`](results-fusion-chunked-ce-2026-08-18.md),
`scripts/bench_fusion.py`, and the full 191-test reconciled suite including
`tests/test_fusion.py`.

**Supersedes:** [D019](#d019)'s planned adoption and [D028](#d028)'s
throughput-unmeasured disposition. The implementation and parity evidence in D028
remain valid.

<a id="d032"></a>
## D032 — Keep chunked cross-entropy as a memory-only opt-in

- **Date:** 2026-08-18
- **Status:** Accepted
- **Scope:** Systems optimization

**Decision:** Keep `chunked_cross_entropy=false` by default and reject it as a
throughput optimization on the tested GB10 stack. Retain it as an explicit
memory-pressure option when saving roughly 3–5GB makes an otherwise blocked model
or batch shape feasible, with chunk size selected by a local sweep.

**Why:** At 50M, the standard path measured 66,998 tokens/s and 16.0GB peak;
2,048/4,096/8,192-row chunks measured 48,256/49,088/49,434 tokens/s
(0.720x/0.733x/0.738x) and 11.3/11.8/12.7GB. Reversing the selected 8,192-row
comparison reproduced the result at 49,206 versus 66,388 tokens/s (0.741x). At
300M, the standard path measured 19,288 tokens/s and 40.2GB; 4,096 and 8,192 rows
measured 17,388 and 17,371 tokens/s (both 0.901x), at 36.0 and 37.0GB. Reversing the
4,096-row comparison measured 17,376 versus 19,333 tokens/s (0.899x). The memory
reduction is real, but recomputing the vocabulary projection costs 10% at 300M and
26–28% at 50M.

**Consequences:** Throughput-focused runs use standard cross-entropy. A
memory-constrained use must report both the saved peak allocation and the speed
cost, and must record the switch as a numerical intervention: bf16 gradients differ
at roughly 2.3e-3 and Muon can amplify that trajectory difference under
[D029](#d029). The result does not establish value at 600M/1B or when the saved
memory changes the feasible batch shape; that specific end-to-end case remains a
valid future test.

**Evidence:** [`results-fusion-chunked-ce-2026-08-18.md`](results-fusion-chunked-ce-2026-08-18.md),
`scripts/bench_cross_entropy.py`, `src/modern_lm/losses.py`, and the full 191-test
suite including `tests/test_losses.py`.

**Supersedes:** [D030](#d030)'s throughput-unmeasured disposition. Its arithmetic,
gradient, and saved-tensor evidence remain valid.

<a id="d033"></a>
## D033 — Expose functional Transformer Engine FP8/NVFP4 modes; keep BF16 default

- **Date:** 2026-08-18
- **Status:** Accepted
- **Scope:** Numerical and systems optimization

**Decision:** Support `--precision fp8` and `--precision nvfp4` as checkpoint-
portable experimental modes in pretraining and SFT, backed by Transformer Engine
2.18. Keep BF16 autocast as the default and do not present either low-precision
mode as an efficiency or capability improvement yet. Quantize only aligned hidden
projections under `blocks.*`/`mtp.*`; retain fp32 master weights and keep embedding,
norms, routers, attention, the vocabulary head, and loss on the existing path.

Use `Float8CurrentScaling` for FP8. On GB10/sm_121 use `NVFP4BlockScaling` with
2-D scaling and the random Hadamard transform enabled, but stochastic rounding
disabled: Transformer Engine 2.18 compiles its stochastic FP4 conversion only for
sm_100/sm_103, and requesting it on sm_121 triggers a device assertion. Record the
exact recipe, capability, converted/skipped linears, and disabled features in every
run identity. Treat the sm_121 NVFP4 path as a deterministic-rounding experiment,
not as evidence for the full recommended NVFP4 training recipe.

**Why:** Both modes now pass finite forward/backward, hybrid Muon/AdamW updates,
two-microbatch accumulation, `torch.compile`, and BF16↔low-precision model and
optimizer checkpoint loading on the local GB10. The official kernels remove the
correctness limitation of the earlier probes, but not the end-to-end economics.
At 32,768 tokens/update, order-balanced compiled tests produced these mean median
throughput ratios to BF16:

| Layout | 50M FP8 | 50M NVFP4 | 300M FP8 | 300M NVFP4 |
|---|---:|---:|---:|---:|
| Separate projections (default) | 0.732× | 0.647× | 0.803× | 0.706× |
| Fused QKV + gate/up | 0.830× | 0.736× | 0.916× | 0.816× |

At fused 300M, isolated peak allocation fell from 40.18GB BF16 to 37.08GB FP8
and 34.98GB NVFP4. That memory reduction is real, but neither mode clears the
throughput bar, and short repeated-batch losses diverge enough that capability
cannot be inferred from kernel validity.

**Consequences:** A low-precision run is an approximate numerical intervention
and must be a declared checkpoint fork with trajectory-level loss/capability
measurement. Projection fusion is useful for reducing low-precision launch and
quantization overhead, but remains off in the canonical BF16 architecture. Keep
outer `torch.compile`: Transformer Engine linears are graph breaks, yet disabling
compilation cut fused 300M low-precision throughput by more than half. Revisit
promotion with a newer backend, larger GEMMs, a memory-enabled larger batch/model,
or an end-to-end fused Transformer layer—not with raw GEMM peak claims.

**Evidence:** [`low-precision.md`](low-precision.md),
[`results-low-precision-2026-08-18.md`](results-low-precision-2026-08-18.md),
`src/modern_lm/low_precision.py`, `scripts/bench_low_precision.py`,
`tests/test_low_precision.py`, and `tests/test_low_precision_gpu.py`.

**Supersedes:** [D017](#d017). D017's earlier custom-kernel measurements remain
valid for that implementation; they no longer describe the available runtime.

## New-entry template

Append entries; do not insert them above older decisions.

```markdown
<a id="dNNN"></a>
## DNNN — Imperative decision title

- **Date:** YYYY-MM-DD
- **Status:** Accepted | Provisional | Planned | Deferred | Rejected | Superseded
- **Scope:** Architecture | Training | Evaluation | Systems | Project

**Decision:** What is now true.

**Why:** The mechanism and tradeoff, not merely the outcome.

**Consequences:** What future work must do differently.

**Evidence:** Protocols, runs, measurements, code, or external reference.

**Supersedes:** DNNN, when applicable.
```
