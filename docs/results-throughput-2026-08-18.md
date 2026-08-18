# Throughput and accounting validation — 2026-08-18

Status: **accepted for the NVIDIA GB10 and the tested 50M/300M profiles**

Decision: [D027](decisions.md#d027)

## Question

After removing per-microbatch GPU-to-host metric conversions, does the trainer
become faster, and does `64 × 1` still beat historical `16 × 4` when both execute
the same 32,768 supervised tokens per optimizer update?

## Protocol

- Hardware: one NVIDIA GB10 with a shared 121GB unified-memory pool.
- Baseline code: `ba35390`, immediately before the incoming performance patch.
- Candidate code: `3f3e2d1`, containing sync-free metrics and segment accounting.
- Driver: `scripts/bench_batch_shape.py::measure` on the canonical FineMath stream.
- Execution: `torch.compile`, bf16 autocast, hybrid Muon/AdamW, sequence length 512.
- Shapes: `microbatch=16, accumulation=4` and `microbatch=64, accumulation=1`.
- Warmup: four complete optimizer updates before timing each measurement.
- Timed work: 20 updates at 50M and 8 updates at 300M.
- Replication: two measurements per cell, with shape order reversed on the second
  pass to expose order/thermal bias.
- Memory safety: 70GB per-process allocation cap.

This is a systems benchmark, not a learning run. Both shapes consume the same token
stream and update budget; loss is not used as the performance outcome.

## Results

Values are the mean of the two order-balanced repetitions.

| Profile | Code | 16×4 tok/s | 64×1 tok/s | 64×1 gain | Peak allocated, 16×4 → 64×1 |
|---|---|---:|---:|---:|---:|
| 50M | `ba35390` | 59,489 | 65,132 | +9.49% | 5.0 → 16.0GB |
| 50M | sync-free `3f3e2d1` | 59,266 | 64,925 | **+9.55%** | 5.0 → 16.0GB |
| 300M | `ba35390` | 17,425 | 19,257 | +10.52% | 14.0 → 40.2GB |
| 300M | sync-free `3f3e2d1` | 17,579 | 19,307 | **+9.83%** | 14.0 → 40.2GB |

The sync-free patch's same-shape deltas were −0.38% and −0.32% at 50M, and
+0.89% and +0.26% at 300M for 16×4 and 64×1 respectively. This range is not a
material throughput effect. Removing the scalar reads improves measurement
discipline and eliminates a synchronization hazard, but it did not make the
steady-state loop meaningfully faster in this test.

Raw token/s repetitions:

| Profile | Code/shape | Pass 1 | Pass 2 |
|---|---|---:|---:|
| 50M | baseline 16×4 | 59,495 | 59,482 |
| 50M | baseline 64×1 | 65,133 | 65,131 |
| 50M | sync-free 16×4 | 59,103 | 59,429 |
| 50M | sync-free 64×1 | 64,923 | 64,926 |
| 300M | baseline 16×4 | 17,351 | 17,499 |
| 300M | baseline 64×1 | 19,231 | 19,284 |
| 300M | sync-free 16×4 | 17,576 | 17,582 |
| 300M | sync-free 64×1 | 19,275 | 19,339 |

## Functional validation

- The incoming performance/accounting tests and architecture manifest passed
  19/19 before additional coverage was added.
- The complete pre-reconciliation repository suite passed 151/151; after adding
  parity and terminal-checkpoint coverage plus the timing fix, it passed 153/153.
- Added coverage directly compares a combined batch with four token-weighted
  microbatches through loss, gradients, and an AdamW optimizer step.
- A compiled CUDA trainer smoke test exercised the segment ledger,
  `--profile-every`, evaluation, checkpointing, and batch-shape intervention on
  resume. Its cumulative training-only rate (150,535 tok/s) agreed with the
  synchronized update profile (150,623 tok/s) within 0.1%.
- The smoke resume exposed that a terminal checkpoint's own write time was absent
  from its serialized state. The trainer now refreshes timing in the small JSON
  sidecar after weight I/O and overlays only those timing fields on load. A focused
  regression test covers the one-checkpoint terminal case.
- Lazy evaluation-graph compilation is warmed in `compile_and_warmup` before the
  initial held-out measurement; a CUDA check reduced initial evaluation attribution
  from 3.92 seconds including compilation to 0.013 seconds of actual evaluation.

## Interpretation

1. Keep `64 × 1` as the default through 300M when its memory cost fits. The gain
   survives removal of the suspected synchronization confound.
2. Classify sync-free metric collection as an accounting/correctness change, not a
   measured efficiency win.
3. Keep 600M and 1B on explicit smaller shapes until they are remeasured; this test
   did not revisit them.
4. New runs quote `training_tokens_per_second`. Historical
   `tokens_per_second` remains end-to-end and is not field-comparable.

The result is scoped to this hardware, compiler/runtime, model implementation, and
the two tested profiles. Two short repetitions establish a large systems effect,
not a cross-hardware generalization claim.
