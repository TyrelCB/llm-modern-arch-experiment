#!/usr/bin/env python3
"""Does FP8's break-even move with batch size and context length?

The first FP8 measurement used one shape -- microbatch 16 x seq 512 = 8192 rows
-- and landed at 0.96-0.99x end to end: break-even. That single point does not
say whether the format is fundamentally not worth it here or whether the
configuration was simply too small to amortize quantization.

Both knobs change the GEMM's aspect ratio. Rows = microbatch * seq_len, so
either raising the microbatch or lengthening the context makes the same weight
matrix serve more rows. The quantize cost scales with rows*K (memory-bound, one
pass over the activation) while the GEMM scales with rows*K*N (compute-bound),
so N is what decides whether the GEMM can pay for the quantization -- and the
row count decides whether the GEMM is big enough to hit peak.

Reports the per-GEMM speedup including quantization, which is the number that
matters, not the isolated `_scaled_mm` figure.
"""
from __future__ import annotations

import argparse
import itertools
import time

import torch

E4M3_MAX = 448.0


def quantize_rowwise(x: torch.Tensor):
    scale = (x.abs().amax(dim=-1, keepdim=True) / E4M3_MAX).clamp(min=1e-12)
    return (x / scale).to(torch.float8_e4m3fn), scale.float()


def bench(fn, n: int = 30, warmup: int = 12) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / n


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dim", type=int, default=1024, help="300M's dim")
    parser.add_argument("--ffn", type=int, default=3456)
    args = parser.parse_args()

    quantize = torch.compile(quantize_rowwise, dynamic=False)
    torch._dynamo.config.cache_size_limit = 256

    # The four GEMMs in a block, as (label, K, N).
    gemms = [
        ("qkv", args.dim, 3 * args.dim),
        ("attn_out", args.dim, args.dim),
        ("ffn_up", args.dim, 2 * args.ffn),
        ("ffn_down", args.ffn, args.dim),
    ]
    microbatches = [4, 8, 16, 32, 64]
    seq_lens = [512, 1024, 2048, 4096]

    print(f"dim={args.dim} ffn={args.ffn}   speedup includes quantization\n")
    header = "rows".rjust(8) + "".join(f"{label:>11}" for label, _, _ in gemms)
    print(f"{'mb':>4} {'seq':>6} {header}  {'block':>8}")

    for mb, seq in itertools.product(microbatches, seq_lens):
        rows = mb * seq
        if rows > 262144:  # keep the sweep inside memory
            continue
        cells, total_bf16, total_fp8 = [], 0.0, 0.0
        for _, k, n in gemms:
            a = torch.randn(rows, k, device="cuda", dtype=torch.bfloat16)
            w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
            a8, sa = quantize(a)
            w8, sw = quantize(w)

            t_bf16 = bench(lambda: a @ w.t())

            def fp8_path():
                qa, s = quantize(a)
                return torch._scaled_mm(qa, w8.t(), scale_a=s, scale_b=sw.t(),
                                        out_dtype=torch.bfloat16)

            t_fp8 = bench(fp8_path)
            total_bf16 += t_bf16
            total_fp8 += t_fp8
            cells.append(f"{t_bf16 / t_fp8:>10.2f}x")
            del a, w, a8, w8
            torch.cuda.empty_cache()

        print(f"{mb:>4} {seq:>6} {rows:>8,}" + "".join(cells)
              + f"  {total_bf16 / total_fp8:>7.2f}x")


if __name__ == "__main__":
    main()
