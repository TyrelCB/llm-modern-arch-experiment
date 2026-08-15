#!/usr/bin/env python3
"""Is NVFP4 worth it at the shapes ModernLM actually trains at?

The headline 3.6x for NVFP4 comes from square 4096 GEMMs. This repo's models
are much smaller, and a training step is not only GEMMs. This script measures
the GEMM speedup at the real per-layer shapes before any training path gets
written.

Shapes come from the size ladder: tokens = microbatch * seq_len = 16 * 512 =
8192 rows, against each block's qkv / attn-out / ffn-up / ffn-down weights.

NVFP4 here is the raw `torch._scaled_mm` with per-16-element E4M3 block scales.
Quantization cost is NOT included -- this is the optimistic ceiling. If the
ceiling is not worth having, the real implementation certainly is not.
"""
from __future__ import annotations

import time

import torch

# name, dim, ffn -- from the verified body-param ladder
LADDER = [
    ("50m", 576, 1984),
    ("100m", 704, 2368),
    ("300m", 1024, 3456),
    ("600m", 1536, 5120),
]
TOKENS = 16 * 512  # microbatch * sequence_length


def bench(fn, n: int = 30, warmup: int = 10) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / n


def _pad(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def make_fp4(rows: int, cols: int):
    """Packed e2m1 pair-per-byte, plus per-16-element e4m3 block scales.

    The kernel wants the scales in a swizzled layout, not a plain [rows,
    cols/16] matrix: rows pad to 128 and the block count pads to 4. Passing the
    unpadded shape fails with "Invalid scaling configuration" even though the
    element count looks right for aligned sizes -- N=1728 needs 64512 scales
    (padded to 1792 rows), not the 62208 the naive shape gives.
    """
    packed = torch.randint(0, 255, (rows, cols // 2), device="cuda",
                           dtype=torch.uint8).view(torch.float4_e2m1fn_x2)
    scales = torch.ones(_pad(rows, 128) * _pad(cols // 16, 4), device="cuda",
                        dtype=torch.float8_e4m3fn)
    return packed, scales


def compare(m: int, k: int, n: int) -> tuple[float, float, float]:
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    t_bf16 = bench(lambda: a @ b.t())

    a4, sa = make_fp4(m, k)
    b4, sb = make_fp4(n, k)
    t_fp4 = bench(lambda: torch._scaled_mm(
        a4, b4.t(), scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16))

    flops = 2 * m * k * n
    return t_bf16, t_fp4, flops


def main() -> None:
    print(f"tokens per microbatch: {TOKENS}")
    print(f"device: {torch.cuda.get_device_properties(0).name}\n")
    print(f"{'size':>6} {'gemm':>10} {'M':>6} {'K':>6} {'N':>6} "
          f"{'bf16 TF':>9} {'fp4 TF':>9} {'speedup':>8}")

    for name, dim, ffn in LADDER:
        # The four GEMMs in a block, as (label, K, N) against TOKENS rows.
        gemms = [
            ("qkv", dim, 3 * dim),
            ("attn_out", dim, dim),
            ("ffn_up", dim, 2 * ffn),   # gated: gate+up in one
            ("ffn_down", ffn, dim),
        ]
        total_bf16 = total_fp4 = 0.0
        for label, k, n in gemms:
            t_bf16, t_fp4, flops = compare(TOKENS, k, n)
            total_bf16 += t_bf16
            total_fp4 += t_fp4
            print(f"{name:>6} {label:>10} {TOKENS:>6} {k:>6} {n:>6} "
                  f"{flops/t_bf16/1e12:>9.1f} {flops/t_fp4/1e12:>9.1f} "
                  f"{t_bf16/t_fp4:>7.2f}x")
        print(f"{name:>6} {'BLOCK':>10} {'':>6} {'':>6} {'':>6} "
              f"{'':>9} {'':>9} {total_bf16/total_fp4:>7.2f}x\n")


if __name__ == "__main__":
    main()
