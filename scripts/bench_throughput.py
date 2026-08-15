#!/usr/bin/env python3
"""Training throughput across model sizes, for picking a size that iterates fast.

Sizes are named by *body* parameters -- transformer blocks plus the final norm,
with the embedding table and lm_head excluded. With the untied 16,384-token
vocab a "5M" model would otherwise be 13.1M parameters of which 8.4M is a lookup
table that contributes no matmul FLOPs. Body params are the compute-bearing
capacity, so they are what throughput should be plotted against.

WHAT THIS MEASURES, AND WHAT IT DOES NOT
----------------------------------------
Batches are synthetic. That bypasses `PackedTokenStream.batch` (data.py), which
gathers rows in a Python loop at a cost independent of model size and is the
likely bottleneck below ~50M. So these numbers are a GPU-side ceiling and will
OVERSTATE what the small end actually trains at.

The run is also expected to share the GPU with a live training job, which
depresses every number by an unknown and possibly size-dependent amount.

Both caveats mean: use these for the RELATIVE SHAPE of throughput vs size, not
as absolute tok/s. They are not comparable to historical figures from real runs.
Pass --include-reference to measure the 145M and 600M shapes in the same pass;
their ratio to each other calibrates how much distortion is present.

The training step itself is the real one -- `build_optimizer` (Muon on hidden
matrices, AdamW elsewhere) and `compute_loss` are imported from the production
code rather than reimplemented, so Newton-Schulz orthogonalization and the bf16
cross-entropy path are priced in exactly as they are in training.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from dataclasses import replace
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# `zeropower_via_newtonschulz` is compiled with dynamic=False, so every distinct
# matrix shape gets its own specialized kernel. A training run has one model and
# stays under Dynamo's default limit of 8; this script walks nine models through
# one process and blows past it around the third size. Raising the limit keeps
# each shape specialized exactly as it is in training -- the alternative,
# recompiling dynamically, would measure a kernel production never uses.
torch._dynamo.config.recompile_limit = 256

from modern_lm.config import ModernConfig  # noqa: E402
from modern_lm.model import ModernLM  # noqa: E402
from modern_lm.muon import build_optimizer  # noqa: E402
from modern_lm.train import TrainSettings, compute_loss  # noqa: E402

# name, dim, n_layers, n_heads, ffn_dim, expected body params.
#
# head_dim is 64 and ffn_dim ~= 3.4*dim throughout, following the design rule in
# run_muon_600m_8b.sh; the dim/n_layers aspect stays near 52. Every rung is
# within 5% of its round-number target. Expected body counts were measured from
# the real ModernLM, and are asserted at build time -- a silent shape error
# would invalidate the whole comparison.
LADDER = [
    ("5m", 256, 5, 4, 896, 4_754_816),
    ("10m", 320, 7, 5, 1088, 10_184_256),
    ("20m", 448, 7, 7, 1536, 20_078_016),
    ("50m", 576, 11, 9, 1984, 52_324_672),
    ("100m", 704, 14, 11, 2368, 97_793_728),
    ("200m", 896, 17, 14, 3072, 195_003_136),
    ("300m", 1024, 20, 16, 3456, 296_267_264),
]

# The two shapes with real training runs behind them, on the same body axis.
REFERENCE = [
    ("145m-ref", 768, 15, 12, 2432, 119_465_088),
    ("600m-ref", 1280, 24, 20, 4352, 558_432_512),
]


def gpu_is_busy() -> bool:
    """True if another job is actively computing, which makes numbers relative-only.

    Keys on utilization, not memory. An idle llama-server can sit on tens of GB
    without consuming SMs; that costs headroom (and may OOM the largest rungs)
    but does not slow the rungs that do fit, so it must not be reported as
    contention.
    """
    import subprocess
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10).stdout
        return int(out.strip().splitlines()[0]) > 10
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return True  # can't tell -- assume the worse case


def body_params(model: ModernLM) -> int:
    """Transformer body only: embeddings and lm_head excluded."""
    excluded = sum(p.numel() for name, p in model.named_parameters()
                   if "embed" in name.lower() or "lm_head" in name.lower())
    return sum(p.numel() for p in model.parameters()) - excluded


def build_config(dim: int, layers: int, heads: int, ffn: int) -> ModernConfig:
    """Same override path the trainer uses (train.py builds these with `replace`)."""
    return replace(ModernConfig(), dim=dim, n_layers=layers, n_heads=heads,
                   n_kv_heads=heads, ffn_dim=ffn)


def measure(name: str, dim: int, layers: int, heads: int, ffn: int, expected: int,
            *, microbatch: int, accum: int, seq_len: int, warmup: int, steps: int,
            device: torch.device, compile_model: bool) -> dict:
    """Time `steps` full optimizer steps on synthetic batches. Returns tok/s.

    The timed unit is one optimizer step including all `accum` micro-batches,
    which is what production does. Timing a single micro-batch instead would
    charge every Newton-Schulz orthogonalization to 1/accum as many tokens and
    understate real throughput by roughly the accumulation factor.
    """
    config = build_config(dim, layers, heads, ffn)
    model = ModernLM(config).to(device)

    measured = body_params(model)
    if measured != expected:
        raise SystemExit(
            f"{name}: body params {measured:,} != expected {expected:,}. "
            "The ladder and the model disagree; fix the table before trusting any number.")

    # Count before compiling: torch.compile prefixes parameter names with
    # "_orig_mod.", and body_params matches on names.
    if compile_model:
        model = torch.compile(model)

    settings = TrainSettings(optimizer="muon", sequence_length=seq_len)
    optimizer = build_optimizer(
        model, learning_rate=settings.learning_rate,
        muon_learning_rate=settings.muon_learning_rate,
        weight_decay=settings.weight_decay,
        muon_weight_decay=settings.effective_muon_weight_decay())

    # One synthetic batch, reused. Content is irrelevant to timing -- every token
    # costs the same FLOPs -- and reusing it keeps allocator behaviour steady.
    tokens = torch.randint(0, config.vocab_size, (microbatch, seq_len + 1), device=device)
    amp = device.type == "cuda" and torch.cuda.is_bf16_supported()

    def step() -> None:
        optimizer.zero_grad(set_to_none=True)
        for _ in range(accum):
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                loss, _, _ = compute_loss(model, tokens, settings)
            # Production scales each micro-batch by its token share; with equal
            # micro-batches that is a constant 1/accum, and the division costs
            # the same either way.
            (loss / accum).backward()
        optimizer.step()

    for _ in range(warmup):
        step()

    tokens_per_step = microbatch * accum * seq_len
    rates = []
    for _ in range(steps):
        torch.cuda.synchronize()
        start = time.perf_counter()
        step()
        torch.cuda.synchronize()
        rates.append(tokens_per_step / (time.perf_counter() - start))

    peak_gb = torch.cuda.max_memory_allocated(device) / 1e9
    torch.cuda.reset_peak_memory_stats(device)
    del model, optimizer, tokens
    torch.cuda.empty_cache()

    ordered = sorted(rates)
    return {
        "name": name,
        "body_params": measured,
        "median": statistics.median(ordered),
        "p10": ordered[int(0.10 * (len(ordered) - 1))],
        "p90": ordered[int(0.90 * (len(ordered) - 1))],
        "peak_gb": peak_gb,
        "samples": rates,
    }


def dry_run(rungs: list[tuple]) -> None:
    """Build every config on CPU and check body params. No GPU touched."""
    print(f"{'name':>9} {'dim':>5} {'L':>3} {'heads':>6} {'ffn':>5} "
          f"{'body params':>13} {'expected':>13}  ok")
    ok = True
    for name, dim, layers, heads, ffn, expected in rungs:
        model = ModernLM(build_config(dim, layers, heads, ffn))
        measured = body_params(model)
        match = measured == expected
        ok = ok and match
        print(f"{name:>9} {dim:>5} {layers:>3} {heads:>6} {ffn:>5} "
              f"{measured:>13,} {expected:>13,}  {'yes' if match else 'NO'}")
        del model
    if not ok:
        raise SystemExit("body-param mismatch: fix LADDER before running on GPU")
    print("\nall shapes valid")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--microbatch", type=int, default=16,
                        help="sequences per micro-batch; fixed across sizes so the "
                             "comparison isolates model size (default: 16)")
    parser.add_argument("--accum", type=int, default=4,
                        help="micro-batches per optimizer step. Default 4 matches "
                             "production's 16 x 4 x 512 = 32,768 tokens/update")
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=12,
                        help="discarded steps; must cover torch.compile's first-step "
                             "cost or compilation leaks into the timings (default: 12)")
    parser.add_argument("--steps", type=int, default=30,
                        help="timed steps per size (default: 30)")
    parser.add_argument("--no-compile", action="store_true",
                        help="skip torch.compile. Production compiles, so this "
                             "measures a path real training does not use")
    parser.add_argument("--include-reference", action="store_true",
                        help="also measure the 145M and 600M shapes, to calibrate "
                             "against runs whose real throughput is known")
    parser.add_argument("--dry-run", action="store_true",
                        help="validate shapes on CPU and exit")
    parser.add_argument("--json", type=Path, help="write raw per-step samples here")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    rungs = list(LADDER) + (REFERENCE if args.include_reference else [])
    # Ascending body params, so a mistake surfaces on a cheap rung first.
    rungs.sort(key=lambda row: row[5])

    if args.dry_run:
        dry_run(rungs)
        return

    device = torch.device(args.device)
    # Whether the GPU was contended decides if these numbers are absolute or
    # only relative, so record it rather than assuming either way.
    busy = gpu_is_busy()

    print(__doc__.split("WHAT THIS MEASURES")[0].strip())
    print()
    print("!! Synthetic batches: no dataloader cost, so the small end is optimistic.")
    print(f"   {args.microbatch} x {args.accum} x {args.seq_len} = "
          f"{args.microbatch * args.accum * args.seq_len:,} tokens/optimizer step, "
          f"{args.warmup} warmup + {args.steps} timed steps per size, Muon, "
          f"{'eager' if args.no_compile else 'torch.compile'}.")
    if busy:
        print("!! GPU is shared right now -- treat these as relative shape, not absolute.")
    else:
        print("   GPU idle at start: these are absolute tok/s.")
    print()

    results = []
    for row in rungs:
        name = row[0]
        print(f"  measuring {name} ...", end="", flush=True)
        try:
            result = measure(name, *row[1:], microbatch=args.microbatch,
                             accum=args.accum, seq_len=args.seq_len,
                             warmup=args.warmup, steps=args.steps, device=device,
                             compile_model=not args.no_compile)
        except torch.cuda.OutOfMemoryError:
            # Expected at the top of the ladder while another job holds memory.
            # Skip rather than abort: the smaller rungs are still worth having.
            torch.cuda.empty_cache()
            print(" OOM (skipped)")
            results.append({"name": name, "body_params": row[5], "oom": True})
            continue
        print(f" {result['median']:,.0f} tok/s")
        results.append(result)

    print()
    print(f"{'size':>9} {'body params':>13} {'tok/s':>10} {'p10':>10} {'p90':>10} "
          f"{'peak GB':>8} {'exponent':>9}")
    previous = None
    for result in results:
        if result.get("oom"):
            print(f"{result['name']:>9} {result['body_params']:>13,} "
                  f"{'OOM':>10} {'':>10} {'':>10} {'':>8} {'':>9}")
            continue
        exponent = ""
        if previous is not None:
            # d log(tok/s) / d log(body params).
            exponent = (f"{(math.log(result['median']) - math.log(previous['median'])) / (math.log(result['body_params']) - math.log(previous['body_params'])):9.2f}")
        print(f"{result['name']:>9} {result['body_params']:>13,} "
              f"{result['median']:>10,.0f} {result['p10']:>10,.0f} "
              f"{result['p90']:>10,.0f} {result['peak_gb']:>8.2f} {exponent:>9}")
        previous = result

    print()
    print("exponent = d log(tok/s) / d log(body params), between adjacent sizes.")
    print("  near -1  compute-bound: throughput falls in proportion to size")
    print("  near  0  something else is the ceiling -- making the model smaller buys nothing")

    if args.json:
        args.json.write_text(json.dumps({
            "microbatch": args.microbatch, "seq_len": args.seq_len,
            "warmup": args.warmup, "steps": args.steps,
            "contended": busy, "synthetic_batches": True,
            "results": results,
        }, indent=2))
        print(f"\nraw samples -> {args.json}")


if __name__ == "__main__":
    main()
