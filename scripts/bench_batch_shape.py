#!/usr/bin/env python3
"""Is microbatch 16 x accumulation 4 costing us wall clock?

32,768 tokens per optimizer update is inherited from the DeepSeek-V4 comparison
protocol, where it was a CONTROL -- pinned so architecture was the only
variable, never tuned for this hardware. Every run since has carried it.

The same 32,768 tokens can come from mb 16 x ga 4 or mb 64 x ga 1. They are
mathematically identical: token-weighted accumulation means the gradient is the
same either way. What differs is execution -- 4 forward/backward passes with
8,192-row GEMMs versus 1 pass with 32,768-row GEMMs, and 4x the kernel launches
and Python loop overhead.

Marek et al. (arXiv:2507.07101) recommend against gradient accumulation unless
training multiple replicas across devices. This is one GPU, so ga>1 buys nothing
in principle. This measures what it costs in practice, and what it costs in
memory -- the reason to keep accumulation is if mb 64 does not fit.

Reports steady-state tokens/sec and peak memory for each configuration.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from modern_lm.config import ModernConfig  # noqa: E402
from modern_lm.data import PackedTokenStream  # noqa: E402
from modern_lm.model import ModernLM  # noqa: E402
from modern_lm.muon import build_optimizer  # noqa: E402
from modern_lm.train import TrainSettings, compute_loss  # noqa: E402

TRAIN_BIN = Path("/home/tyrel/projects/llm-deepseek-v4-experiment/"
                 "data/finemath-6b/train.bin")

# name, dim, layers, heads, ffn -- from the verified body-param ladder
SIZES = {
    "50m": (576, 11, 9, 1984),
    "100m": (704, 14, 11, 2368),
    "300m": (1024, 20, 16, 3456),
    # 600m is the real runs/muon-600m-8b shape; 1b is a candidate on the same
    # aspect rules (head_dim 64, ffn ~3.4*dim) at body 1,008,696,576.
    "600m": (1280, 24, 20, 4352),
    "1b": (1536, 30, 24, 5248),
}


def measure(config: ModernConfig, microbatch: int, accumulation: int,
            steps: int, compile_model: bool, budget_gb: float = 0.0) -> dict:
    """Time one configuration. `budget_gb` caps the process's share of the pool.

    This box has UNIFIED memory: the GPU and CPU share one 121GB pool, so an
    over-large microbatch does not fail cleanly with a CUDA OOM -- it evicts
    page cache and starts pressuring everything else on the machine. The
    fraction cap makes the allocator raise OutOfMemoryError at a chosen ceiling
    instead, which the caller catches and reports.
    """
    torch.manual_seed(2026)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    device = torch.device("cuda")
    if budget_gb:
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        torch.cuda.set_per_process_memory_fraction(min(0.95, budget_gb / total))

    settings = TrainSettings(microbatch_size=microbatch,
                             gradient_accumulation=accumulation,
                             optimizer="muon", muon_learning_rate=0.005)
    model = ModernLM(config).to(device)
    if compile_model:
        model = torch.compile(model)
    optimizer = build_optimizer(model, learning_rate=3e-4,
                                muon_learning_rate=0.005, weight_decay=0.1,
                                muon_weight_decay=0.1)
    stream = PackedTokenStream(TRAIN_BIN, settings.sequence_length, settings.seed)

    micro_step = 0

    def one_update() -> None:
        nonlocal micro_step
        optimizer.zero_grad(set_to_none=True)
        for _ in range(accumulation):
            tokens = stream.batch(micro_step, microbatch, device)
            micro_step += microbatch
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, _, _ = compute_loss(model, tokens, settings)
            (loss / accumulation).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    for _ in range(4):          # warmup: allocator, autotune, compile
        one_update()
    torch.cuda.synchronize()

    started = time.perf_counter()
    for _ in range(steps):
        one_update()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    tokens_per_update = microbatch * accumulation * settings.sequence_length
    result = {
        "tok_s": steps * tokens_per_update / elapsed,
        "ms_per_update": elapsed / steps * 1000,
        "peak_gb": torch.cuda.max_memory_allocated() / 1e9,
    }
    del model, optimizer
    torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="*", default=["50m", "100m", "300m"])
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--compile", action="store_true",
                        help="match real runs, which train compiled")
    parser.add_argument("--budget-gb", type=float, default=70.0,
                        help="abort a config above this rather than squeeze "
                             "the rest of the box (unified memory)")
    args = parser.parse_args()

    # Every pairing yields 32,768 tokens per update.
    configs = [(16, 4), (32, 2), (64, 1), (128, 1), (256, 1)]

    print("NOTE: mb 16/32/64 all give 32,768 tokens per update; mb 128 and "
          "256\n      give 65,536 and 131,072 -- a different optimizer "
          "trajectory, not\n      just a faster one. Compare tok/s, not loss.\n")
    print(f"{args.steps} timed updates"
          f"{', compiled' if args.compile else ', eager'}\n")
    print(f"{'size':>6} {'mb x ga':>9} {'tok/s':>10} {'ms/update':>11} "
          f"{'peak GB':>9} {'vs 16x4':>9}")

    for name in args.sizes:
        dim, layers, heads, ffn = SIZES[name]
        config = ModernConfig(dim=dim, n_layers=layers, n_heads=heads,
                              n_kv_heads=heads, ffn_dim=ffn)
        baseline = None
        for microbatch, accumulation in configs:
            try:
                r = measure(config, microbatch, accumulation, args.steps,
                            args.compile, args.budget_gb)
            except torch.OutOfMemoryError:
                print(f"{name:>6} {f'{microbatch}x{accumulation}':>9} "
                      f"{'over budget':>10}")
                torch.cuda.empty_cache()
                continue
            baseline = baseline or r["tok_s"]
            print(f"{name:>6} {f'{microbatch}x{accumulation}':>9} "
                  f"{r['tok_s']:>10,.0f} {r['ms_per_update']:>10.1f}ms "
                  f"{r['peak_gb']:>8.1f}G {r['tok_s'] / baseline:>8.2f}x")
        print()


if __name__ == "__main__":
    main()
