"""Score a ModernLM checkpoint on the identical benchmark suite as the reference.

`extract_number` and `numeric_equal` are imported from the DeepSeek experiment
rather than reimplemented. A separately-maintained copy of a scorer produces
numbers that are not legitimately comparable, and this comparison's entire
value rests on both models being scored by the same code.

Defaults mirror the reference's *controlled* evaluation: greedy decoding,
max_new_tokens=32, `Question: ...\\nAnswer:` prompt format.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import torch
from tokenizers import Tokenizer

from .config import ModernConfig
from .data import DEEPSEEK_REPO, default_paths
from .model import ModernLM

# Reuse the reference scorer directly.
sys.path.insert(0, str(DEEPSEEK_REPO / "src"))
from deepseek_v4.evaluation import extract_number, numeric_equal  # noqa: E402


@torch.no_grad()
def evaluate_checkpoint(checkpoint_path: Path, tokenizer_path: Path,
                        examples_path: Path, output_path: Path,
                        device: torch.device, max_new_tokens: int = 32,
                        batch_size: int = 16) -> dict:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = ModernLM(ModernConfig(**payload["config"]))
    model.load_state_dict(payload["model"])
    del payload
    gc.collect()
    model.to(device).eval()

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    eos_id = tokenizer.token_to_id("<eos>")

    prepared = []
    with examples_path.open() as handle:
        for index, line in enumerate(handle):
            example = json.loads(line)
            prompt = f"Question: {example['question']}\nAnswer:"
            ids = tokenizer.encode(prompt, add_special_tokens=False).ids
            available = model.config.max_seq_len - max_new_tokens
            ids = ids[-available:]
            prepared.append((index, example, ids))

    # Group equal-length prompts so each batch needs no padding, matching the
    # reference's batching strategy.
    by_length: dict[int, list] = {}
    for item in prepared:
        by_length.setdefault(len(item[2]), []).append(item)

    records: list[dict | None] = [None] * len(prepared)
    totals: dict[str, int] = {}
    correct: dict[str, int] = {}
    processed = 0
    amp = device.type == "cuda" and torch.cuda.is_bf16_supported()

    for items in by_length.values():
        for start in range(0, len(items), batch_size):
            batch = items[start:start + batch_size]
            input_ids = torch.tensor([item[2] for item in batch],
                                     dtype=torch.long, device=device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                generated = model.generate(input_ids, max_new_tokens=max_new_tokens,
                                           eos_token_id=eos_id)
            prompt_length = input_ids.shape[1]
            for row, (index, example, _) in enumerate(batch):
                completion = tokenizer.decode(generated[row, prompt_length:].tolist())
                prediction = extract_number(completion)
                matched = numeric_equal(prediction, example["answer"])
                benchmark = example["benchmark"]
                totals[benchmark] = totals.get(benchmark, 0) + 1
                correct[benchmark] = correct.get(benchmark, 0) + int(matched)
                records[index] = {**example, "completion": completion,
                                  "prediction": prediction, "correct": matched}
            processed += len(batch)
            print(json.dumps({"event": "evaluation_progress",
                              "processed": processed, "total": len(prepared)}), flush=True)

    summary = {name: {"correct": correct.get(name, 0), "total": total,
                      "accuracy": correct.get(name, 0) / total}
               for name, total in totals.items()}
    overall_correct = sum(correct.values())
    overall_total = sum(totals.values())
    summary["overall"] = {"correct": overall_correct, "total": overall_total,
                          "accuracy": overall_correct / max(1, overall_total)}
    numeric = sum(1 for r in records if r and r["prediction"] is not None)
    summary["numeric_completion_rate"] = numeric / max(1, len(records))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n")
    output_path.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    paths = default_paths()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, default=paths["tokenizer"])
    parser.add_argument("--examples", type=Path, default=paths["evaluation"])
    parser.add_argument("--output", type=Path, default=Path("runs/modern-145m/evaluation.jsonl"))
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    summary = evaluate_checkpoint(args.checkpoint, args.tokenizer, args.examples,
                                  args.output, torch.device(args.device),
                                  args.max_new_tokens, args.batch_size)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
