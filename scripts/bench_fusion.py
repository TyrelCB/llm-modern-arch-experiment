#!/usr/bin/env python3
"""What does fusing Q/K/V and SwiGLU gate/up actually buy?

The block runs five input projections where it could run two. Fusing them does
not change a single arithmetic operation -- the same matmuls happen against the
same numbers -- so this cannot show up as fewer FLOPs. What it changes is:

  - reads of x: three passes over the activation for q/k/v become one, two for
    gate/up become one. On a box whose wins have all come from moving fewer
    bytes, this is the reason to expect anything at all.
  - kernel launches: five per block per pass become two.
  - GEMM shape: [dim, dim] and [dim, ffn] become [dim, 3*dim] and [dim, 2*ffn],
    which is a better aspect ratio for the same total work.

Parity is established in tests/test_fusion.py; this only answers whether the
change is worth adopting ([D028](../docs/decisions.md#d028)). Compiled GB10
screening found no win ([D031](../docs/decisions.md#d031)); this script exists to
remeasure other stacks. It alternates arm order because a sub-percent result is
otherwise indistinguishable from warmup or thermal order bias.

Reports tokens/sec and peak memory for both, at the batch shape D024 settled on.
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from dataclasses import replace
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# One process walks several models through Dynamo; the Newton-Schulz kernel is
# compiled per shape with dynamic=False and would otherwise blow the default
# cache limit partway through the sweep.
torch._dynamo.config.recompile_limit = 256

from modern_lm.config import ModernConfig  # noqa: E402
from modern_lm.model import ModernLM  # noqa: E402
from modern_lm.muon import build_optimizer  # noqa: E402
from modern_lm.train import TrainSettings, compute_loss  # noqa: E402

# Same rungs and aspect rules as bench_throughput.py's ladder.
SIZES = {
    "50m": (576, 11, 9, 1984),
    "100m": (704, 14, 11, 2368),
    "300m": (1024, 20, 16, 3456),
    "600m": (1280, 24, 20, 4352),
}


def measure(config: ModernConfig, *, microbatch: int, accumulation: int, steps: int,
            warmup: int, compile_model: bool, device: torch.device) -> dict:
    torch.manual_seed(2026)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    settings = TrainSettings(microbatch_size=microbatch,
                             gradient_accumulation=accumulation,
                             optimizer="muon", muon_learning_rate=0.005)
    model = ModernLM(config).to(device)
    if compile_model:
        model = torch.compile(model)
    optimizer = build_optimizer(model, learning_rate=3e-4, muon_learning_rate=0.005,
                                weight_decay=0.1, muon_weight_decay=0.1)

    # Synthetic tokens: every token costs the same FLOPs, and this isolates the
    # model from PackedTokenStream's Python gather, which fusion cannot affect.
    tokens = torch.randint(0, config.vocab_size,
                           (microbatch, settings.sequence_length + 1), device=device)

    def update() -> None:
        optimizer.zero_grad(set_to_none=True)
        for _ in range(accumulation):
            with torch.autocast(device.type, dtype=torch.bfloat16):
                loss, _, _ = compute_loss(model, tokens, settings)
            (loss / accumulation).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    for _ in range(warmup):
        update()

    tokens_per_update = microbatch * accumulation * settings.sequence_length
    rates = []
    for _ in range(steps):
        torch.cuda.synchronize()
        started = time.perf_counter()
        update()
        torch.cuda.synchronize()
        rates.append(tokens_per_update / (time.perf_counter() - started))

    peak = torch.cuda.max_memory_allocated() / 1e9
    del model, optimizer, tokens
    torch.cuda.empty_cache()
    return {"tok_s": statistics.median(rates), "peak_gb": peak}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sizes", nargs="*", default=["50m", "100m", "300m"])
    parser.add_argument("--microbatch", type=int, default=64)
    parser.add_argument("--accumulation", type=int, default=1)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--repetitions", type=int, default=2,
                        help="repeat each arm and alternate order; must be positive")
    parser.add_argument("--eager", action="store_true",
                        help="also report uncompiled, where launch overhead dominates "
                             "and the fusion should look larger than it is in production")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device; these numbers are meaningless on CPU")
    if args.repetitions <= 0:
        raise SystemExit("--repetitions must be positive")

    device = torch.device("cuda")
    modes = [("compiled", True)] + ([("eager", False)] if args.eager else [])
    print(f"{'size':>6} {'mode':>9} {'separate':>12} {'fused':>12} {'ratio':>7} "
          f"{'peak sep':>9} {'peak fus':>9}")
    for name in args.sizes:
        dim, layers, heads, ffn = SIZES[name]
        base = replace(ModernConfig(), dim=dim, n_layers=layers, n_heads=heads,
                       n_kv_heads=heads, ffn_dim=ffn)
        for mode, compiled in modes:
            observations = {"separate": [], "fused": []}
            for repetition in range(args.repetitions):
                order = (("separate", False), ("fused", True))
                if repetition % 2:
                    order = tuple(reversed(order))
                for label, fused in order:
                    observations[label].append(measure(
                        replace(base, fuse_projections=fused),
                        microbatch=args.microbatch, accumulation=args.accumulation,
                        steps=args.steps, warmup=args.warmup, compile_model=compiled,
                        device=device))
            results = {
                label: {
                    "tok_s": statistics.median(item["tok_s"] for item in samples),
                    "peak_gb": statistics.median(item["peak_gb"] for item in samples),
                }
                for label, samples in observations.items()
            }
            ratio = results["fused"]["tok_s"] / results["separate"]["tok_s"]
            print(f"{name:>6} {mode:>9} {results['separate']['tok_s']:>12,.0f} "
                  f"{results['fused']['tok_s']:>12,.0f} {ratio:>6.3f}x "
                  f"{results['separate']['peak_gb']:>8.1f}G "
                  f"{results['fused']['peak_gb']:>8.1f}G")


if __name__ == "__main__":
    main()
