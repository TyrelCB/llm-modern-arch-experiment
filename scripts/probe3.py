#!/usr/bin/env python3
"""Three questions against a checkpoint, for eyeballing progress in seconds.

The full 5024-question benchmark takes ~80 minutes and costs 15-20% of training
throughput while it runs. These three cover the same three capabilities and are
one forward pass each:

  arithmetic  Calculate 48 + 22.            -- carrying
  algebra     Solve for x: 9x + 11 = 146.   -- two-step isolate-then-divide
  word        Sandra ... six cups ...       -- comprehension plus arithmetic

Answers are scored with the same answer_segment + numeric_equal path the real
benchmark uses, so a tick here means the same thing it means there. Three
questions prove nothing statistically -- read the completions, not the score.

Usage:
  python3 scripts/probe3.py                       # newest checkpoint
  python3 scripts/probe3.py runs/.../ckpt.pt ...  # specific ones
  python3 scripts/probe3.py --all                 # every checkpoint, oldest first
"""
from __future__ import annotations

import argparse
import gc
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

PROBES = [
    ("arithmetic", "Calculate 48 + 22.", "70"),
    ("algebra", "Solve for x: 9x + 11 = 146.", "15"),
    ("word", "Sandra took six cups of coffee and Marcie took two cups of coffee. "
             "How many cups of coffee did they take in total?", "8"),
]


def run(checkpoint: Path, tokenizer: Tokenizer, device: torch.device,
        max_new_tokens: int) -> None:
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
    for kind, question, gold in PROBES:
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
        print(f"  [{mark}] {kind:10s} want {gold:>4s}  ->  "
              f"{segment.strip()[:88]!r}")

    print(f"  {score}/3", flush=True)
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
    parser.add_argument("--max-new-tokens", type=int, default=48)
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

    tokenizer = Tokenizer.from_file(str(default_paths()["tokenizer"]))
    device = torch.device(args.device)
    for checkpoint in targets:
        run(checkpoint, tokenizer, device, args.max_new_tokens)


if __name__ == "__main__":
    main()
