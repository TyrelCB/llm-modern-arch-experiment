#!/usr/bin/env python3
"""Does computing the vocabulary loss in chunks pay for the extra matmul?

The trade is explicit. Chunked cross-entropy recomputes each slice's logits
during the backward pass instead of storing them, so it runs the head projection
roughly one and a half times per step where the standard path runs it once. In
exchange it never allocates the [tokens, 16,384] logit tensor, nor the gradient
with respect to it -- about 2.1 GB of bf16 traffic per micro-batch at the
production shape, and the single largest allocation in the step.

FLOPs up, bytes down. On a box that is bandwidth-bound rather than FLOP-bound
that should be a win, but "should" has already been wrong here: FP8's GEMMs were
genuinely faster and quantization ate the gain. So this measures both numbers
that matter -- wall clock and peak memory -- and reports them side by side
([D027](../docs/decisions.md#d027)).

Peak memory is the more reliable payoff. If it drops sharply while throughput
only holds level, the change still buys something real: headroom to raise the
microbatch, which D024 measured as worth 4-9% on its own, or to fit a rung that
does not currently fit.
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

torch._dynamo.config.recompile_limit = 256

from modern_lm.config import ModernConfig  # noqa: E402
from modern_lm.model import ModernLM  # noqa: E402
from modern_lm.muon import build_optimizer  # noqa: E402
from modern_lm.train import TrainSettings, compute_loss  # noqa: E402

SIZES = {
    "50m": (576, 11, 9, 1984),
    "100m": (704, 14, 11, 2368),
    "300m": (1024, 20, 16, 3456),
    "600m": (1280, 24, 20, 4352),
}


def measure(config: ModernConfig, settings: TrainSettings, *, steps: int, warmup: int,
            compile_model: bool, device: torch.device) -> dict:
    torch.manual_seed(2026)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    model = ModernLM(config).to(device)
    if compile_model:
        model = torch.compile(model)
    optimizer = build_optimizer(model, learning_rate=3e-4, muon_learning_rate=0.005,
                                weight_decay=0.1, muon_weight_decay=0.1)
    tokens = torch.randint(0, config.vocab_size,
                           (settings.microbatch_size, settings.sequence_length + 1),
                           device=device)

    def update() -> None:
        optimizer.zero_grad(set_to_none=True)
        for _ in range(settings.gradient_accumulation):
            with torch.autocast(device.type, dtype=torch.bfloat16):
                loss, _, _ = compute_loss(model, tokens, settings)
            (loss / settings.gradient_accumulation).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    for _ in range(warmup):
        update()

    per_update = (settings.microbatch_size * settings.gradient_accumulation
                  * settings.sequence_length)
    rates = []
    for _ in range(steps):
        torch.cuda.synchronize()
        started = time.perf_counter()
        update()
        torch.cuda.synchronize()
        rates.append(per_update / (time.perf_counter() - started))

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
    parser.add_argument("--chunks", nargs="*", type=int, default=[2048, 4096, 8192],
                        help="row slices to sweep; the best is a memory/launch tradeoff")
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--eager", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device; these numbers are meaningless on CPU")

    device = torch.device("cuda")
    compile_model = not args.eager
    base_settings = TrainSettings(microbatch_size=args.microbatch,
                                  gradient_accumulation=args.accumulation,
                                  optimizer="muon", muon_learning_rate=0.005)
    tokens_per_update = args.microbatch * args.accumulation * base_settings.sequence_length

    print(f"{tokens_per_update:,} tokens per update, vocabulary "
          f"{ModernConfig().vocab_size:,}: a full logit tensor is "
          f"{tokens_per_update * ModernConfig().vocab_size * 2 / 1e9:.2f} GB in bf16, "
          f"and its gradient the same again\n")
    print(f"{'size':>6} {'variant':>14} {'tok/s':>12} {'ratio':>7} {'peak':>8} {'saved':>8}")

    for name in args.sizes:
        dim, layers, heads, ffn = SIZES[name]
        config = replace(ModernConfig(), dim=dim, n_layers=layers, n_heads=heads,
                         n_kv_heads=heads, ffn_dim=ffn)
        baseline = measure(config, base_settings, steps=args.steps, warmup=args.warmup,
                           compile_model=compile_model, device=device)
        print(f"{name:>6} {'standard':>14} {baseline['tok_s']:>12,.0f} "
              f"{1.0:>6.3f}x {baseline['peak_gb']:>7.1f}G {'--':>8}")

        for chunk in args.chunks:
            settings = replace(base_settings, chunked_cross_entropy=True,
                               cross_entropy_chunk=chunk)
            result = measure(config, settings, steps=args.steps, warmup=args.warmup,
                             compile_model=compile_model, device=device)
            ratio = result["tok_s"] / baseline["tok_s"]
            saved = baseline["peak_gb"] - result["peak_gb"]
            print(f"{name:>6} {'chunk ' + str(chunk):>14} {result['tok_s']:>12,.0f} "
                  f"{ratio:>6.3f}x {result['peak_gb']:>7.1f}G {saved:>7.1f}G")


if __name__ == "__main__":
    main()
