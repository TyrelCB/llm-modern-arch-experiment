# Projection fusion and chunked cross-entropy: local GPU validation

Date: **2026-08-18**<br>
Candidate commits: **`d4831de`**, **`3bb8856`**<br>
Hardware: **NVIDIA GB10**, compute capability 12.1, driver 580.173.02<br>
Runtime: Python 3.12.3, PyTorch 2.14.0.dev20260620+cu130, CUDA 13.0

## Outcome

- Projection fusion is **not an efficiency win** on this compiled GB10 stack.
  Order-balanced aggregate throughput was 0.998x at 50M and 0.991x at 300M,
  with no peak-allocation change. It remains implemented but default-off
  ([D031](decisions.md#d031)).
- Chunked vocabulary cross-entropy is a **real memory optimization but a clear
  throughput regression**. It saves 3.2-4.8GB while running at 0.72-0.90x the
  standard path. It remains a memory-pressure opt-in, not a default
  ([D032](decisions.md#d032)).

## Contract and method

Both benchmarks used synthetic token IDs to isolate model/optimizer work from the
packed-data loader, seed 2026, bf16 autocast, hybrid Muon/AdamW, 512-token
sequences, and the production 64x1 batch shape (32,768 targets/update). Models were
compiled with `torch.compile`. Each arm received four warmup updates; each reported
rate is the median of 12 synchronized updates at 50M or eight at 300M.

The machine's resident ComfyUI, Unsloth, and llama-server processes held about
30GB of GPU/unified allocation, but utilization was 0% immediately before and after
the tests. No training process was active. Projection fusion was tested once in
separate-to-fused order and once in fused-to-separate order. Chunked loss used a
full sweep followed by a reverse-order confirmation of the selected chunk at each
scale.

Before performance testing:

```text
PYTHONPATH=src python3 -m pytest -q
189 passed, 14 warnings in 12.25s
```

The warnings are existing PyTorch JIT deprecations from `test_fp8_linear.py`.
After merging with the newer local timing and SFT work, the reconciled tree passed
all 191 tests with the same 14 warnings.

## Projection fusion

Forward order:

| Size | Separate tok/s | Fused tok/s | Fused/separate | Peak separate | Peak fused |
|---|---:|---:|---:|---:|---:|
| 50M | 65,381 | 65,509 | 1.002x | 16.0GB | 16.0GB |
| 300M | 18,996 | 18,992 | 1.000x | 40.2GB | 40.2GB |

Reverse order:

| Size | Fused tok/s | Separate tok/s | Fused/separate | Peak fused | Peak separate |
|---|---:|---:|---:|---:|---:|
| 50M | 66,461 | 66,878 | 0.994x | 16.0GB | 16.0GB |
| 300M | 19,018 | 19,340 | 0.983x | 40.2GB | 40.2GB |

Median across the two order positions:

| Size | Fused tok/s | Separate tok/s | Fused/separate |
|---|---:|---:|---:|
| 50M | 65,985 | 66,129.5 | 0.998x |
| 300M | 19,005 | 19,168 | 0.991x |

The apparent effect changes direction with order at 50M and never exceeds baseline
at 300M. With identical peak allocation, there is no capability-cost reason to pay
the checkpoint-layout and Muon-trajectory complexity. The converter and block-aware
Muon implementation remain useful if another compiler, kernel, precision, or
hardware target makes fusion valuable later.

## Chunked vocabulary cross-entropy

Full forward-order sweep:

| Size | Variant | Tok/s | Relative | Peak | Saved |
|---|---|---:|---:|---:|---:|
| 50M | standard | 66,998 | 1.000x | 16.0GB | -- |
| 50M | chunk 2,048 | 48,256 | 0.720x | 11.3GB | 4.8GB |
| 50M | chunk 4,096 | 49,088 | 0.733x | 11.8GB | 4.3GB |
| 50M | chunk 8,192 | 49,434 | 0.738x | 12.7GB | 3.3GB |
| 300M | standard | 19,288 | 1.000x | 40.2GB | -- |
| 300M | chunk 4,096 | 17,388 | 0.901x | 36.0GB | 4.2GB |
| 300M | chunk 8,192 | 17,371 | 0.901x | 37.0GB | 3.2GB |

Reverse-order confirmation:

| Size | Selected chunk | Chunked tok/s | Standard tok/s | Relative | Peak chunked | Peak standard |
|---|---:|---:|---:|---:|---:|---:|
| 50M | 8,192 | 49,206 | 66,388 | 0.741x | 12.7GB | 16.0GB |
| 300M | 4,096 | 17,376 | 19,333 | 0.899x | 36.0GB | 40.2GB |

The reverse run reproduces both the memory saving and slowdown. Chunk size is an
operational memory/launch tradeoff, not a learning hyperparameter, but the bf16
gradient path differs numerically from standard cross-entropy. Under Muon that is a
recorded trajectory intervention ([D029](decisions.md#d029)).

## Scope and next test

These results cover compiled 50M and 300M profiles on this GB10 software stack.
They do not answer whether chunked loss is worthwhile at 600M/1B when the saved
memory changes the feasible microbatch shape. That is the only strong reason to
reopen it: compare the best standard shape with the best feasible chunked shape,
including the recomputation cost, rather than comparing memory in isolation.
