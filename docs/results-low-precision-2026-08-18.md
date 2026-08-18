# Transformer Engine FP8/NVFP4 validation — 2026-08-18

## Outcome

FP8 and NVFP4 are now executable end to end in pretraining and SFT, including
finite backward, hybrid Muon/AdamW updates, accumulated microbatches,
`torch.compile`, and portable checkpoints. Neither is an end-to-end throughput
win on the NVIDIA GB10 at 50M or 300M, so BF16 remains the default. The modes are
retained for numerical/capability experiments, future backend versions, larger
shapes, and memory-constrained cases ([D033](decisions.md#d033)).

## Protocol

- Hardware: NVIDIA GB10, compute capability 12.1; GPU utilization was 0% at the
  start of every process. Idle inference services retained roughly 30GB of unified
  memory but did not consume SM time.
- Software: PyTorch `2.14.0.dev20260620+cu130`, Transformer Engine 2.18.0,
  Python 3.12.3.
- Step: synthetic fixed tokens, sequence 512, `64 × 1 = 32,768` targets/update,
  full vocabulary cross-entropy, global gradient clip, production hybrid
  Muon/AdamW optimizer, outer `torch.compile`.
- Sampling: four discarded warmup updates; 10 timed updates per mode at 50M and
  eight at 300M. Each layout was run in forward `BF16→FP8→NVFP4` and reverse
  `NVFP4→FP8→BF16` order. Values below are the mean of the two per-process median
  rates.
- Layouts: the canonical separate Q/K/V and gate/up modules, and the optional
  source-level fused QKV/gate-up topology.

The benchmark intentionally includes projection quantization, attention, LM head,
loss, backward, clipping, Muon's Newton–Schulz work, and the AdamW step. It is not a
raw GEMM benchmark and excludes data loading.

## Throughput

| Profile/layout | BF16 tok/s | FP8 tok/s | FP8/BF16 | NVFP4 tok/s | NVFP4/BF16 |
|---|---:|---:|---:|---:|---:|
| 50M, separate | 65,812 | 48,168 | 0.732× | 42,554 | 0.647× |
| 50M, fused | 65,727 | 54,532 | 0.830× | 48,386 | 0.736× |
| 300M, separate | 19,253 | 15,452 | 0.803× | 13,598 | 0.706× |
| 300M, fused | 18,950 | 17,362 | 0.916× | 15,463 | 0.816× |

Larger GEMMs amortize the quantization overhead better, and combining QKV plus
gate/up reduces the number of Transformer Engine linears at 300M from 140 to 80.
That closes most of FP8's gap, but the best tested low-precision path is still
8.4% slower than BF16. Fully eager fused 300M was much worse—7,806 tok/s FP8 and
7,649 tok/s NVFP4—so the graph boundaries introduced by TE do not justify turning
off the outer compiler.

## Memory

Peak allocation must be measured with each mode first/alone because Transformer
Engine retains process-global workspaces. Isolated fused-300M runs measured:

| Mode | Peak allocated | Change from BF16 |
|---|---:|---:|
| BF16 | 40.18GB | — |
| FP8 | 37.08GB | −3.11GB (−7.7%) |
| NVFP4 | 34.98GB | −5.20GB (−12.9%) |

This is useful headroom but not a default-worthy efficiency result by itself. A
future experiment may justify the mode if the saved memory enables a larger
microbatch or model that BF16 cannot run; that end-to-end configuration must be
benchmarked directly.

## Numerical and hardware findings

- FP8 `Float8CurrentScaling` has finite activation and weight gradients at the
  real 300M projection shape (`M=8192, K=1024, N=3456`). `DelayedScaling` yielded
  zero gradients on this exact stack and is not exposed.
- Full `NVFP4BlockScaling` reaches an sm_121 device assertion in the stochastic
  FP4 conversion. Transformer Engine 2.18's source instantiates that instruction
  only for sm_100/sm_103. On GB10, disabling stochastic rounding while retaining
  2-D scaling and RHT gives finite real-shape forward/backward.
- The first loss from identical initialization and tokens differs by less than
  0.001 between BF16 and either low-precision mode. Subsequent repeated-batch
  optimization diverges, particularly for deterministic NVFP4. That is expected
  for approximate numerics and is why this benchmark cannot stand in for a
  real-data capability trajectory.
- Canonical checkpoints omit TE `_extra_state` quantizer caches. Strict loading
  succeeds BF16→FP8/NVFP4 and back, including optimizer state; unrelated missing
  or unexpected keys still fail.

## Disposition

Implementation: **accepted as an opt-in**. Throughput promotion: **rejected on the
tested stack**. Capability: **unvalidated**. BF16, separate projections, and outer
compilation remain the canonical defaults. The appropriate next test is a paired
real-data fork only if low precision unlocks a concrete memory advantage or a
newer Transformer Engine/CUDA release materially closes the measured speed gap.
