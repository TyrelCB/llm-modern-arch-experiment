#!/usr/bin/env python3
"""End-to-end NVFP4 vs BF16 on a real ModernLM, small enough to finish fast.

The GEMM microbenchmark says 1.8-3.0x at this repo's shapes, but that is the
GEMM alone with no quantization cost. This trains the actual model both ways,
same seed and same data, and reports wall clock and loss together -- speed is
only interesting if the loss curve survives.

NVFP4 here replaces the forward GEMM in every block Linear. Embeddings,
lm_head, norms, and attention stay bf16, and the backward is a straight-through
bf16 matmul. That is the cheapest thing that could show a real speedup; a
publishable recipe needs stochastic rounding and Hadamard-transformed
gradients, which are deliberately out of scope until the wall clock justifies
writing them.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from modern_lm.config import ModernConfig  # noqa: E402
from modern_lm.data import PackedTokenStream, default_paths  # noqa: E402
from modern_lm.model import ModernLM  # noqa: E402
from modern_lm.train import TrainSettings, compute_loss  # noqa: E402
from fp8_linear import FP8Linear  # noqa: E402


def convert_to_fp8(model: torch.nn.Module, min_expansion: float = 0.0) -> int:
    """Swap block Linears for FP8 ones, preserving weights. Returns count.

    `min_expansion` filters by N/K, which is what decides whether FP8 pays.
    Quantizing costs one memory-bound pass over the activation (rows*K) while
    the GEMM is rows*K*N, so only a wide output amortizes it. Measured per-GEMM
    at 300M's shape: ffn_up (N/K=6.75) 1.50x, qkv (3.0) 1.10x, ffn_down (0.30)
    0.84x, attn_out (1.0) 0.78x. Passing 2.0 keeps the first two and leaves the
    losers in bf16.
    """
    swapped = 0
    for block in model.blocks:
        for parent in (block.attn, block.feed_forward):
            for name, child in list(parent.named_children()):
                if not isinstance(child, torch.nn.Linear):
                    continue
                # K multiple of 16 keeps the kernel happy.
                if child.in_features % 16 or child.out_features % 16:
                    continue
                if child.out_features / child.in_features < min_expansion:
                    continue
                replacement = FP8Linear(child.in_features, child.out_features,
                                          bias=child.bias is not None)
                replacement.weight.data.copy_(child.weight.data)
                if child.bias is not None:
                    replacement.bias.data.copy_(child.bias.data)
                setattr(parent, name, replacement.to(child.weight.device))
                swapped += 1
    return swapped


def run(use_fp4: bool, steps: int, config: ModernConfig,
        settings: TrainSettings, train_path: Path,
        min_expansion: float = 0.0) -> dict:
    torch.manual_seed(2026)
    device = torch.device("cuda")
    model = ModernLM(config).to(device)

    swapped = convert_to_fp8(model, min_expansion) if use_fp4 else 0
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    stream = PackedTokenStream(train_path, settings.sequence_length, settings.seed)

    model.train()
    losses = []
    micro_step = 0
    # Warm up outside the timed region: allocator and any autotuning.
    for warm in range(3):
        tokens = stream.batch(micro_step, settings.microbatch_size, device)
        micro_step += settings.microbatch_size
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, _, _ = compute_loss(model, tokens, settings)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    torch.cuda.synchronize()
    started = time.perf_counter()
    for step in range(steps):
        for _ in range(settings.gradient_accumulation):
            tokens = stream.batch(micro_step, settings.microbatch_size, device)
            micro_step += settings.microbatch_size
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, _, _ = compute_loss(model, tokens, settings)
            (loss / settings.gradient_accumulation).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        losses.append(loss.item())
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    tokens_per_step = (settings.microbatch_size * settings.gradient_accumulation
                       * settings.sequence_length)
    return {
        "elapsed": elapsed,
        "tok_s": steps * tokens_per_step / elapsed,
        "first_loss": losses[0],
        "final_loss": sum(losses[-10:]) / len(losses[-10:]),
        "swapped": swapped,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--dim", type=int, default=576)
    parser.add_argument("--layers", type=int, default=11)
    parser.add_argument("--heads", type=int, default=9)
    parser.add_argument("--ffn", type=int, default=1984)
    parser.add_argument("--min-expansion", type=float, default=0.0,
                        help="only convert Linears with out/in >= this")
    args = parser.parse_args()

    config = ModernConfig(dim=args.dim, n_layers=args.layers,
                          n_heads=args.heads, n_kv_heads=args.heads,
                          ffn_dim=args.ffn)
    settings = TrainSettings(optimizer="adamw")
    train_path = Path("/home/tyrel/projects/llm-deepseek-v4-experiment/"
                      "data/finemath-6b/train.bin")

    print(f"shape dim={args.dim} layers={args.layers} ffn={args.ffn}, "
          f"{args.steps} steps\n")
    results = {}
    for label, use_fp4 in (("bf16", False), ("fp8", True)):
        r = run(use_fp4, args.steps, config, settings, train_path,
                args.min_expansion)
        results[label] = r
        extra = f", {r['swapped']} linears swapped" if use_fp4 else ""
        print(f"{label:>6}: {r['tok_s']:>9,.0f} tok/s  {r['elapsed']:>6.1f}s  "
              f"loss {r['first_loss']:.3f} -> {r['final_loss']:.3f}{extra}")

    speedup = results["fp8"]["tok_s"] / results["bf16"]["tok_s"]
    delta = results["fp8"]["final_loss"] - results["bf16"]["final_loss"]
    print(f"\nspeedup {speedup:.2f}x   loss delta {delta:+.4f}")


if __name__ == "__main__":
    main()
