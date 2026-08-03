# Results (in progress)

The 250M-token ModernLM run is executing. This file is completed from
`runs/comparison.json` when it finishes; the numbers below that are already
fixed are the reference side, read from the DeepSeek-V4 experiment's own
committed artifacts.

## Reference baseline (fixed, not recomputed)

DeepSeek-V4 145M, `runs/finemath-145m`, 250,000,000 target tokens:

| Metric | Value |
|---|---:|
| Stored parameters | 144,669,412 |
| Active parameters / token | 45,578,980 |
| Final held-out main loss | 2.55136 |
| Held-out perplexity | 12.8245 |
| Initial held-out loss | 9.87923 |
| Throughput (best measured, 64x1 optimized) | 9,720 tok/s |
| Throughput (as actually run, 16x4) | 8,600 tok/s |

Pretrained-checkpoint benchmarks, greedy, 32 new tokens, no SFT:

| Benchmark | Correct / total | Accuracy |
|---|---:|---:|
| ASDiv | 22 / 2,305 | 0.954% |
| SVAMP | 15 / 1,000 | 1.500% |
| GSM8K | 18 / 1,319 | 1.365% |
| Generated algebra | 1 / 100 | 1.000% |
| Generated arithmetic | 0 / 300 | 0.000% |
| **Overall** | **56 / 5,024** | **1.115%** |

## ModernLM run

| Metric | Value |
|---|---:|
| Parameters | 144,630,912 (-0.027% vs reference stored) |
| Active parameters / token | 144,630,912 (dense) |
| Initial held-out loss | 9.8526 |
| Observed steady-state throughput | ~36,300 tok/s |

Results pending.
