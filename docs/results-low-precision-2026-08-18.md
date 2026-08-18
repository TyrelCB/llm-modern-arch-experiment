# Transformer Engine FP8/NVFP4 validation — 2026-08-18

## Outcome

FP8 and NVFP4 are now executable end to end in pretraining and SFT, including
finite backward, hybrid Muon/AdamW updates, accumulated microbatches,
`torch.compile`, and portable checkpoints. Neither is an end-to-end throughput
win on the NVIDIA GB10 at 50M or 300M. At fixed batch shape FP8 approaches but
does not exceed BF16 at 600M and 1B. A conditional crossover does appear at 1B:
FP8's memory headroom permits `64 × 1` instead of BF16 `32 × 2` at the same 32,768
tokens/update, improving throughput by 1.68%. BF16 remains the project-wide
default; the 1B result is a scoped systems option pending capability validation
([D033](decisions.md#d033), [D034](decisions.md#d034)).

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

## Larger-model and batch follow-up

The follow-up tested the exact project ladder shapes—600M (`D=1280, L=24,
F=4352`) and 1B (`D=1536, L=30, F=5248`)—plus a larger 300M microbatch. All runs
used fused QKV/gate-up projections, the best low-precision layout from the first
screen, and the same compiled full training step. Each precision ran in a fresh
process, and all 23 retained measurements began at 0% GPU utilization.

An 82GB decimal per-process CUDA ceiling protected the GB10's 121GB unified pool
while idle inference services retained about 29.4GB. Four updates were discarded
for compile/autotune warmup and six were timed. The near-parity BF16/FP8 pairs at
600M and 1B were repeated in reverse process order; the table reports the mean of
per-process medians. The clear 300M result and fixed-shape NVFP4 rows use one long
process each.

### Same-shape throughput

| Profile / batch shape | Mode | tok/s | Ratio to BF16 | Peak allocated |
|---|---|---:|---:|---:|
| 300M, `128 × 1` | BF16 | 20,579 | 1.000× | 76.85GB |
|  | FP8 | 18,865 | 0.917× | 70.60GB |
|  | NVFP4 | 16,312 | 0.793× | 66.65GB |
| 600M, `64 × 1` | BF16 | 10,697 | 1.000× | 58.68GB |
|  | FP8 | 10,644 | 0.995× | 53.42GB |
|  | NVFP4 | 9,727 | 0.909× | 50.16GB |
| 1B, `32 × 1` | BF16 | 5,462 | 1.000× | 48.95GB |
|  | FP8 | 5,451 | 0.998× | 44.74GB |
|  | NVFP4 | 5,176 | 0.948× | 41.73GB |

Doubling only the 300M microbatch does not help FP8 relative to BF16. Increasing
projection width does: FP8 is within 0.5% at 600M and 0.2% at 1B while saving
roughly 9% peak allocation at the same shape. That is parity/headroom, not a
fixed-shape speedup. NVFP4 also closes much of its gap with scale but remains
5.2–20.7% slower.

### Memory-enabled batch shape at 1B

The canonical optimizer update contains 32,768 targets. Under the 82GB ceiling,
BF16 `64 × 1` does not fit, so its trajectory-preserving shape is `32 × 2`. FP8
and NVFP4 both fit `64 × 1`, eliminating a second forward/backward pass without
changing tokens per update:

| Mode | Microbatch × accumulation | tok/update | Mean tok/s | Ratio | Peak allocated | Process medians |
|---|---:|---:|---:|---:|---:|---|
| BF16 | `32 × 2` | 32,768 | 6,463 | 1.000× | 53.19GB | 6,465; 6,460 |
| FP8 | `64 × 1` | 32,768 | 6,571 | **1.017×** | 78.47GB | 6,570; 6,572 |
| NVFP4 | `64 × 1` | 32,768 | 6,107 | 0.945× | 73.39GB | 6,105; 6,109 |

Thus FP8 produces a repeatable 1.68% end-to-end throughput improvement only when
its lower same-shape memory use removes BF16 gradient accumulation. The selected
FP8 configuration consumes more total memory than BF16 `32 × 2`; the benefit is
using saved bytes to make larger GEMMs, not reducing memory and time
simultaneously. As a secondary, non-trajectory-matched screen, BF16 `52 × 1`, FP8
`64 × 1`, and NVFP4 `68 × 1` reached 6,303, 6,571, and 6,205 tok/s respectively.
BF16 batches 56/64 and NVFP4 batch 72 exceeded the declared ceiling.

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

Implementation: **accepted as an opt-in**. Fixed-shape throughput promotion:
**rejected**. Memory-enabled 1B FP8: **conditionally accepted for systems
experiments** at fused `64 × 1`, where it is 1.68% faster than BF16 `32 × 2` at a
matched update size. NVFP4 throughput promotion: **rejected**. Capability remains
**unvalidated**, so BF16 and separate projections remain the canonical defaults.
The appropriate next FP8 test is a declared paired real-data 1B trajectory; do not
generalize this small hardware/configuration-specific crossover to smaller models.
