#!/usr/bin/env python3
"""A few real benchmark questions against a checkpoint, for eyeballing progress.

The full 5,024-question benchmark takes ~20 minutes with the KV cache and costs
throughput if training is live. This samples 1-3 questions from each of the five
subsets -- arithmetic, algebra, asdiv, svamp, gsm8k -- so the probe is a
miniature of the real eval rather than a separate test.

Sampling from the benchmark matters. The original three probes were
hand-written and covered only arithmetic and algebra, 8% of the questions,
missing asdiv/svamp/gsm8k entirely at 92%. Several SFT variants scored an
identical 2/3 while their real accuracy ranged 10.85% to 13.54%, because every
difference lived in the subsets the probe could not see.

Answers are scored with the same answer_segment + numeric_equal path the real
benchmark uses, so a tick here means the same thing it means there. Five to
fifteen questions still prove nothing statistically -- read the completions,
not the score.

Usage:
  python3 scripts/probe3.py                       # newest checkpoint, 5 probes
  python3 scripts/probe3.py --per-benchmark 3     # 15 probes
  python3 scripts/probe3.py runs/.../ckpt.pt ...  # specific ones
  python3 scripts/probe3.py --all                 # every checkpoint, oldest first
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import types
from pathlib import Path

import torch
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from modern_lm.config import ModernConfig  # noqa: E402
from modern_lm.data import DEEPSEEK_REPO, default_paths  # noqa: E402
from modern_lm.evaluate_benchmarks import answer_segment  # noqa: E402
from modern_lm.model import ModernLM  # noqa: E402

sys.path.insert(0, str(DEEPSEEK_REPO / "src"))
if "pyarrow.parquet" not in sys.modules:
    stub = types.ModuleType("pyarrow.parquet")
    stub.read_table = None
    sys.modules.setdefault("pyarrow", types.ModuleType("pyarrow"))
    sys.modules["pyarrow.parquet"] = stub
from deepseek_v4.evaluation import extract_number, numeric_equal  # noqa: E402

BENCHMARKS = ("arithmetic", "algebra", "asdiv", "svamp", "gsm8k")


def load_probes(per_benchmark: int, seed: int = 2026) -> list[tuple[str, str, str]]:
    """Sample `per_benchmark` real questions from each of the five subsets.

    Sampled from the benchmark itself rather than hand-written, for two
    reasons. The old hand-written set covered only arithmetic and algebra --
    8% of the 5,024 questions -- and missed asdiv/svamp/gsm8k entirely, which
    are 92% of it. That made the probe blind to exactly where the SFT-corpus
    experiments differed: several arms scored an identical 2/3 here while their
    real accuracy ranged 10.85% to 13.54%.

    Selection is deterministic: sort by a hash of the identifier and take the
    first N. Same questions every invocation, and independent of file order.
    """
    path = default_paths()["evaluation"]
    pools: dict[str, list[dict]] = {name: [] for name in BENCHMARKS}
    with path.open() as handle:
        for line in handle:
            record = json.loads(line)
            if record["benchmark"] in pools:
                pools[record["benchmark"]].append(record)

    probes = []
    for name in BENCHMARKS:
        ordered = sorted(
            pools[name],
            key=lambda r: hashlib.sha1(
                f"{seed}:{r['identifier']}".encode()).hexdigest())
        for record in ordered[:per_benchmark]:
            probes.append((name, record["question"], str(record["answer"])))
    return probes


def print_questions(probes: list[tuple[str, str, str]]) -> None:
    """The probe set, once per invocation.

    Printed here rather than per checkpoint so a --all sweep stays readable:
    the questions are fixed, only the completions below them change.
    """
    print(f"Probes ({len(probes)} real benchmark questions, "
          f"{len(probes) // len(BENCHMARKS)} per subset):")
    for kind, question, gold in probes:
        print(f"  {kind:10s} want {gold:>6s}  {question[:86]}")


def run(checkpoint: Path, tokenizer: Tokenizer, device: torch.device,
        max_new_tokens: int, probes: list[tuple[str, str, str]]) -> None:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    tokens_seen = payload.get("state", {}).get("tokens_seen")
    model = ModernLM(ModernConfig(**payload["config"]))
    model.load_state_dict(payload["model"])
    del payload
    gc.collect()
    model.to(device).eval()

    eos = tokenizer.token_to_id("<eos>")
    label = f"{tokens_seen/1e6:,.0f}M tokens" if tokens_seen else checkpoint.name
    print(f"\n=== {label} ===")

    score = 0
    amp = device.type == "cuda" and torch.cuda.is_bf16_supported()
    for kind, question, gold in probes:
        ids = tokenizer.encode(f"Question: {question}\nAnswer:",
                               add_special_tokens=False).ids
        tensor = torch.tensor([ids], dtype=torch.long, device=device)
        with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            out = model.generate(tensor, max_new_tokens=max_new_tokens, eos_token_id=eos)
        completion = tokenizer.decode(out[0, tensor.shape[1]:].tolist())
        segment = answer_segment(completion)
        correct = numeric_equal(extract_number(segment), gold)
        score += correct
        mark = "OK  " if correct else "MISS"
        print(f"  [{mark}] {kind:10s} {segment.strip()[:96]!r}")

    print(f"  {score}/{len(probes)}", flush=True)
    # Free before the next checkpoint loads. Without moving the weights back to
    # CPU first the allocator holds the device blocks and --all OOMs partway
    # through while training has the rest of the box.
    model.to("cpu")
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", nargs="*", type=Path)
    parser.add_argument("--run-dir", type=Path, default=Path("runs/muon-600m-8b"))
    parser.add_argument("--all", action="store_true",
                        help="every checkpoint in --run-dir, oldest first")
    parser.add_argument("--per-benchmark", type=int, default=1,
                        choices=(1, 2, 3),
                        help="questions sampled from each of the five "
                             "benchmarks (5, 10, or 15 probes total)")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()

    if args.checkpoints:
        targets = args.checkpoints
    else:
        found = sorted(args.run_dir.glob("checkpoint-*.pt"))
        if not found:
            print(f"no checkpoints in {args.run_dir}")
            return
        targets = found if args.all else found[-1:]

    probes = load_probes(args.per_benchmark)
    print_questions(probes)

    tokenizer = Tokenizer.from_file(str(default_paths()["tokenizer"]))
    device = torch.device(args.device)
    for checkpoint in targets:
        run(checkpoint, tokenizer, device, args.max_new_tokens, probes)


if __name__ == "__main__":
    main()
