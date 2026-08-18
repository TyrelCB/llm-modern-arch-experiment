# FP8 and NVFP4 training

Status as of **2026-08-18**: both formats are functional Transformer Engine
training modes; both remain experimental and default-off under
[`D033`](decisions.md#d033). BF16 is still the accepted general default. FP8 is a
conditional 1B systems option under [`D034`](decisions.md#d034): its memory
headroom permits `64 × 1` where BF16 uses `32 × 2`, yielding 1.68% more throughput
at the same tokens/update. Neither low-precision mode has completed a capability
trajectory.

## Install

The CUDA 13 backend is an optional dependency. On this GB10/aarch64 machine the
Transformer Engine PyTorch binding must be compiled against the installed nightly
PyTorch and the cuDNN/NCCL headers shipped in its Python packages:

```bash
bash scripts/setup_low_precision.sh
```

The script creates `.venv` with `--system-site-packages`, installs the pinned
Transformer Engine 2.18 packages, supplies the pip CUDA header/library paths to
the binding build, and verifies import plus GPU capability. The runtime also
preloads those pip libraries, so commands do not require a hand-maintained
`LD_LIBRARY_PATH` afterward.

NVIDIA's upstream references are the
[Transformer Engine installation guide](https://github.com/NVIDIA/TransformerEngine),
[FP8 primer](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html),
and [NVFP4 guide](https://docs.nvidia.com/deeplearning/transformer-engine-releases/release-2.15/user-guide/features/low_precision_training/nvfp4/nvfp4.html).

## Run

Activate the optional environment, then add one flag to either training stage:

```bash
source .venv/bin/activate

PYTHONPATH=src python -m modern_lm.train \
  --target-tokens 250000000 --planned-total-tokens 250000000 \
  --run-dir runs/lowp-fp8-50m \
  --dim 576 --n-layers 11 --n-heads 9 --n-kv-heads 9 --ffn-dim 1984 \
  --optimizer muon --muon-learning-rate 0.005 --precision fp8 --device cuda

PYTHONPATH=src python -m modern_lm.sft \
  --checkpoint runs/modern-145m-2b/latest.pt \
  --run-dir runs/modern-145m-2b-sft-nvfp4 \
  --target-updates 100 --planned-total-updates 1000 \
  --precision nvfp4 --device cuda
```

`--fuse-projections` can be paired with low precision for a new pretraining run.
It materially reduces Transformer Engine overhead, but it changes checkpoint
topology and the Muon numerical trajectory; use the existing converter when
forking a separate-projection checkpoint and record the intervention.

## Exact numerical boundary

| Part | BF16 mode | FP8 mode | NVFP4 mode |
|---|---|---|---|
| `blocks.*` and `mtp.*` hidden linears | PyTorch BF16 autocast | TE `Float8CurrentScaling` | TE `NVFP4BlockScaling` |
| Master parameters | fp32 | fp32 | fp32 |
| Embedding, RMSNorm, router, SDPA, LM head, loss | existing BF16/fp32 path | unchanged | unchanged |
| GB10/sm_121 recipe adjustment | — | none | stochastic rounding disabled |
| Checkpoint payload | canonical weights | canonical weights, no TE cache | canonical weights, no TE cache |

Only linears whose input and output dimensions are divisible by 16 are converted.
Skipped names and dimensions are emitted in the run's precision report. Quantized
weight workspaces are runtime caches rather than scientific state, so a checkpoint
saved in any mode strictly loads into BF16, FP8, or NVFP4. Optimizer state keeps
the same parameter order. The trainer also tells Transformer Engine which
accumulation microbatch is first, allowing it to skip an unnecessary accumulation
and reuse quantized weights on later microbatches.

Pretraining may resume under a different precision and records that change as an
intervention. SFT exact resume requires identical settings; to switch precision,
start a declared fork from the checkpoint instead of using `--resume`.

FP8 uses current scaling because it gives finite gradients on this stack and is
Transformer Engine's default FP8 family for compute capability 12.x. Delayed
scaling produced zero gradients locally and is excluded. Full NVFP4 stochastic
conversion is not executable on sm_121 in Transformer Engine 2.18; the source
guards it to sm_100/sm_103. Deterministic rounding, 2-D block scaling, and RHT all
pass real 300M projection forward/backward on GB10. On sm_120, the integration
also disables RHT per NVIDIA's documented shared-memory workaround in
[issue #3062](https://github.com/NVIDIA/TransformerEngine/issues/3062).

Transformer Engine's `Linear.forward` deliberately creates a `torch.compile`
graph boundary. ModernLM keeps the outer compilation because it still optimizes
the surrounding model and more than doubles measured fused-300M throughput versus
fully eager execution. This is functional compile compatibility, not a full-graph
claim.

## Scaling disposition

Fused, fixed-shape FP8 remains slower at 300M (`0.917×` at batch 128), reaches
near parity at 600M (`0.995×` at batch 64) and 1B
(`0.998×` at batch 32). Its useful crossover is memory-enabled rather than a
fixed-shape kernel win. Under an 82GB process ceiling on this GB10:

| 1B mode | Batch shape | tok/s | Ratio to BF16 | Peak allocated |
|---|---:|---:|---:|---:|
| BF16 | `32 × 2` | 6,463 | 1.000× | 53.19GB |
| FP8 | `64 × 1` | 6,571 | 1.017× | 78.47GB |
| NVFP4 | `64 × 1` | 6,107 | 0.945× | 73.39GB |

All shapes process 32,768 targets per optimizer update. Use the FP8 shape only
when the declared memory budget prevents BF16 `64 × 1`; a different machine load
can move that boundary. NVFP4 does not produce a throughput win even after using
its memory savings. See the results document for order-balanced repetitions and
the unmatched maximum-fit screen.

## Validate and benchmark

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/test_low_precision_gpu.py -q

.venv/bin/python scripts/bench_low_precision.py \
  --profile 300m --fuse-projections --order bf16,fp8,nvfp4 \
  --warmup 4 --steps 8 --json /tmp/lowp-forward.json
.venv/bin/python scripts/bench_low_precision.py \
  --profile 300m --fuse-projections --order nvfp4,fp8,bf16 \
  --warmup 4 --steps 8 --json /tmp/lowp-reverse.json

# Peak memory: one mode per process because TE retains global workspaces.
.venv/bin/python scripts/bench_low_precision.py \
  --profile 300m --fuse-projections --order fp8 \
  --warmup 4 --steps 2 --json /tmp/lowp-fp8-memory.json

# 1B matched-update crossover; run each mode in its own process.
.venv/bin/python scripts/bench_low_precision.py \
  --profile 1b --fuse-projections --order bf16 \
  --microbatch 32 --accumulation 2 --memory-budget-gb 82
.venv/bin/python scripts/bench_low_precision.py \
  --profile 1b --fuse-projections --order fp8 \
  --microbatch 64 --accumulation 1 --memory-budget-gb 82
```

See [`results-low-precision-2026-08-18.md`](results-low-precision-2026-08-18.md)
for the measured disposition. Passing the integration tests proves execution,
gradient, optimizer, accumulation, compile, and checkpoint plumbing. It does not
prove learning quality; promotion requires a paired real-data trajectory under the
approximate-numerics lane in [`D003`](decisions.md#d003).
