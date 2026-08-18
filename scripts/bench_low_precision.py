#!/usr/bin/env python3
"""Compare BF16, Transformer Engine FP8, and NVFP4 full training steps.

This prices the production objective and hybrid Muon/AdamW optimizer at the
real 50M or 300M model shape.  Hidden projections change precision; embedding,
normalization, attention, vocabulary projection, loss, and master weights keep
their accepted path.  Run both a forward and reverse ``--order`` to detect
order/thermal effects before treating small differences as real.
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from dataclasses import replace
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from modern_lm.config import ModernConfig  # noqa: E402
from modern_lm.low_precision import configure_low_precision  # noqa: E402
from modern_lm.model import ModernLM  # noqa: E402
from modern_lm.muon import build_optimizer  # noqa: E402
from modern_lm.train import TrainSettings, compute_loss, seed_everything  # noqa: E402


# name -> (dim, layers, heads, ffn, expected total parameters)
PROFILES = {
    "50m": (576, 11, 9, 1984, 71_199_040),
    "300m": (1024, 20, 16, 3456, 329_821_696),
}


def gpu_utilization() -> int | None:
    import subprocess
    try:
        output = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=False).stdout
        return int(output.strip().splitlines()[0])
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None


def percentile(samples: list[float], fraction: float) -> float:
    ordered = sorted(samples)
    return ordered[round(fraction * (len(ordered) - 1))]


def measure(precision: str, *, profile: str, microbatch: int, accumulation: int,
            sequence_length: int, warmup: int, steps: int, compile_model: bool,
            fuse_projections: bool, device: torch.device, seed: int) -> dict:
    dim, layers, heads, ffn, expected_parameters = PROFILES[profile]
    seed_everything(seed)
    config = replace(
        ModernConfig(), dim=dim, n_layers=layers, n_heads=heads,
        n_kv_heads=heads, ffn_dim=ffn, max_seq_len=sequence_length,
        fuse_projections=fuse_projections)
    model = ModernLM(config).to(device)
    if model.num_params() != expected_parameters:
        raise RuntimeError(
            f"{profile} profile has {model.num_params():,} parameters; "
            f"expected {expected_parameters:,}")

    precision_report = configure_low_precision(model, precision, device)
    settings = TrainSettings(
        optimizer="muon", precision=precision, sequence_length=sequence_length,
        microbatch_size=microbatch, gradient_accumulation=accumulation)

    if compile_model:
        model = torch.compile(model)
    optimizer = build_optimizer(
        model,
        learning_rate=settings.learning_rate,
        muon_learning_rate=settings.muon_learning_rate,
        weight_decay=settings.weight_decay,
        muon_weight_decay=settings.effective_muon_weight_decay())

    # Generate on CPU so resetting the seed produces identical tokens even if a
    # numerical backend allocates or initializes CUDA-side workspaces.
    generator = torch.Generator(device="cpu").manual_seed(seed + 1)
    tokens = torch.randint(
        0, config.vocab_size, (microbatch, sequence_length + 1),
        generator=generator).to(device)
    amp = device.type == "cuda" and torch.cuda.is_bf16_supported()

    def step() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        total = None
        for microbatch_index in range(accumulation):
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                loss, _, _ = compute_loss(
                    model, tokens, settings,
                    is_first_microbatch=(microbatch_index == 0))
            scaled = loss / accumulation
            scaled.backward()
            total = scaled.detach() if total is None else total + scaled.detach()
        torch.nn.utils.clip_grad_norm_(model.parameters(), settings.grad_clip)
        optimizer.step()
        return total

    torch.cuda.reset_peak_memory_stats(device)
    warmup_losses = []
    for _ in range(warmup):
        warmup_losses.append(float(step()))
    torch.cuda.synchronize(device)

    rates = []
    losses = []
    tokens_per_step = microbatch * accumulation * sequence_length
    for _ in range(steps):
        start = time.perf_counter()
        loss = step()
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
        rates.append(tokens_per_step / elapsed)
        losses.append(float(loss))

    peak_gb = torch.cuda.max_memory_allocated(device) / 1e9
    result = {
        "precision": precision,
        "precision_report": precision_report.to_dict(),
        "median_tokens_per_second": statistics.median(rates),
        "p10_tokens_per_second": percentile(rates, 0.10),
        "p90_tokens_per_second": percentile(rates, 0.90),
        "peak_allocated_gb": peak_gb,
        "first_warmup_loss": warmup_losses[0],
        "last_warmup_loss": warmup_losses[-1],
        "last_timed_loss": losses[-1],
        "rate_samples": rates,
        "timed_losses": losses,
    }

    del optimizer, model, tokens, loss
    torch._dynamo.reset()
    torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=PROFILES, default="50m")
    parser.add_argument("--order", default="bf16,fp8,nvfp4",
                        help="comma-separated measurement order")
    parser.add_argument("--microbatch", type=int, default=64)
    parser.add_argument("--accumulation", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--fuse-projections", action="store_true")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    order = tuple(item.strip().lower() for item in args.order.split(",") if item.strip())
    if not order or any(item not in ("bf16", "fp8", "nvfp4") for item in order):
        parser.error("--order must contain only bf16,fp8,nvfp4")
    if args.warmup < 1 or args.steps < 1:
        parser.error("--warmup and --steps must be positive")

    device = torch.device(args.device)
    if device.type != "cuda":
        parser.error("this benchmark requires CUDA")
    start_utilization = gpu_utilization()
    metadata = {
        "profile": args.profile,
        "order": order,
        "microbatch": args.microbatch,
        "accumulation": args.accumulation,
        "sequence_length": args.sequence_length,
        "tokens_per_step": args.microbatch * args.accumulation * args.sequence_length,
        "warmup": args.warmup,
        "steps": args.steps,
        "compiled": not args.no_compile,
        "fuse_projections": args.fuse_projections,
        "seed": args.seed,
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "gpu": torch.cuda.get_device_name(device),
        "compute_capability": ".".join(map(str, torch.cuda.get_device_capability(device))),
        "gpu_utilization_at_start": start_utilization,
        "memory_measurement": (
            "isolated"
            if len(order) == 1
            else "order-contaminated: run one precision per process for peak-memory claims"
        ),
    }
    print(json.dumps({"event": "benchmark_identity", **metadata}), flush=True)
    if len(order) > 1:
        print("WARNING: Transformer Engine retains process-global workspaces; "
              "throughput is order-balanced, but peak memory is valid only for "
              "the first mode. Use --order MODE in separate processes for memory.",
              flush=True)

    results = []
    for precision in order:
        print(json.dumps({"event": "measurement_start", "precision": precision}),
              flush=True)
        result = measure(
            precision, profile=args.profile, microbatch=args.microbatch,
            accumulation=args.accumulation, sequence_length=args.sequence_length,
            warmup=args.warmup, steps=args.steps,
            compile_model=not args.no_compile,
            fuse_projections=args.fuse_projections, device=device, seed=args.seed)
        results.append(result)
        print(json.dumps({"event": "measurement_complete", **result}), flush=True)

    baseline = next((item for item in results if item["precision"] == "bf16"), None)
    if baseline is not None:
        for item in results:
            item["throughput_ratio_to_bf16"] = (
                item["median_tokens_per_second"]
                / baseline["median_tokens_per_second"])

    payload = {
        **metadata,
        "gpu_utilization_at_end": gpu_utilization(),
        "results": results,
    }
    print(json.dumps({"event": "summary", **payload}, indent=2), flush=True)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
