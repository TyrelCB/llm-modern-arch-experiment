#!/usr/bin/env python3
"""Replay the FINAL checkpoint's correct answers across an SFT checkpoint ladder.

The 5,024-question benchmark says where a run ended. It does not say when each
answer arrived, or whether the answers the final model gets right were learned
early and held, or acquired late, or bounced. This takes the questions the final
checkpoint answered correctly and re-asks exactly those of every earlier
checkpoint, so each capability gets a settling curve rather than a single number.

Why only the final-correct set: it makes "settled" well defined. Every question
in the set ends at correct, so the curve for each one is a history of how it got
there -- learned-and-held, late-acquired, or unstable. Questions the final model
gets wrong have no endpoint to settle into and would just add noise.

Read the churn column, not only the accuracy column. A run whose accuracy climbs
smoothly while 30% of its answers flip state at every checkpoint has not settled;
it is trading one set of right answers for another, which is what "an edge amid
heavy churn" looked like on the Muon 2B SFT arm.

Scoring uses the benchmark's own answer_segment + numeric path, so a tick here
means what it means there.

ONE CAVEAT, MEASURED NOT ASSUMED. Re-scoring the final checkpoint on its own
correct set does not return 100%: it returns ~98%. Greedy decoding is
deterministic (5 repeats of one prompt give one output, in bf16 and fp32
alike), and the ~2% is batch composition -- the recorded eval grouped all 5,024
questions by prompt length, this replays a 506-question subset, so the length
groups differ and bf16 matmuls are not batch-invariant. It is stable across
batch sizes 16-128, so it is a fixed offset rather than sampling noise.

That means the LAST row is the tool's own zero point, not 100%. Read the curve
relatively: differences between checkpoints are meaningful, the absolute
ceiling is ~2% low, and a per-question flip of a single checkpoint is within
this floor and should not be over-read.

    scripts/probe_settling.py runs/eval-sft-50m-siamese.jsonl runs/sft-50m-siamese
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from modern_lm.config import ModernConfig  # noqa: E402
from modern_lm.model import ModernLM  # noqa: E402
# answer_segment and the reference scorer, imported from the benchmark module so
# a tick here is produced by the same code that produced the recorded eval.
from modern_lm.evaluate_benchmarks import (answer_segment, extract_number,  # noqa: E402
                                           numeric_equal)


def load_model(path: Path, device: torch.device):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = ModernLM(ModernConfig(**payload["config"]))
    model.load_state_dict(payload["model"])
    return model.to(device).eval()


@torch.no_grad()
def score(model, tokenizer, items, device, max_new_tokens, batch_size, eos_id):
    """Greedy-decode each question and score it exactly as the benchmark does.

    Prompt construction, length-grouped batching (which avoids padding entirely,
    so no row sees a pad token), autocast, and scoring all mirror
    evaluate_checkpoint. Anything else would measure a different model than the
    recorded eval did.
    """
    prepared = []
    for item in items:
        prompt = f"Question: {item['question']}\nAnswer:"
        ids = tokenizer.encode(prompt, add_special_tokens=False).ids
        ids = ids[-(model.config.max_seq_len - max_new_tokens):]
        prepared.append((item, ids))

    by_length: dict[int, list] = {}
    for entry in prepared:
        by_length.setdefault(len(entry[1]), []).append(entry)

    amp = device.type == "cuda" and torch.cuda.is_bf16_supported()
    results = {}
    for group in by_length.values():
        for start in range(0, len(group), batch_size):
            batch = group[start:start + batch_size]
            input_ids = torch.tensor([ids for _, ids in batch],
                                     dtype=torch.long, device=device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                generated = model.generate(input_ids, max_new_tokens=max_new_tokens,
                                           eos_token_id=eos_id, use_cache=True)
            prompt_length = input_ids.shape[1]
            for row, (item, _) in enumerate(batch):
                completion = tokenizer.decode(generated[row, prompt_length:].tolist())
                prediction = extract_number(answer_segment(completion))
                results[item["identifier"]] = bool(
                    numeric_equal(prediction, item["answer"]))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("eval_jsonl", type=Path, help="final checkpoint's per-example eval")
    parser.add_argument("run_dir", type=Path, help="SFT run dir holding checkpoint-*.pt")
    parser.add_argument("--max-new-tokens", type=int, default=32,
                        help="must match the budget the eval used (default 32)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tokenizer", type=Path,
                        default=Path("/home/tyrel/projects/llm-deepseek-v4-experiment/"
                                     "data/finemath-6b/tokenizer.json"))
    parser.add_argument("--json", type=Path, help="write per-question histories here")
    args = parser.parse_args()

    records = [json.loads(line) for line in args.eval_jsonl.open()]
    target = [r for r in records if r["correct"]]
    print(f"final checkpoint correct: {len(target)} / {len(records)}")

    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    eos_id = tokenizer.token_to_id("<eos>")
    device = torch.device(args.device)

    checkpoints = sorted(args.run_dir.glob("checkpoint-*.pt"))
    if not checkpoints:
        raise SystemExit(f"no checkpoints in {args.run_dir}")

    by_benchmark = {}
    for item in target:
        by_benchmark.setdefault(item["benchmark"], []).append(item)

    history = {}
    print(f"\n{'checkpoint':>12}{'correct':>9}{'of':>7}{'acc':>8}{'gained':>8}{'lost':>7}"
          + "".join(f"{b[:8]:>10}" for b in sorted(by_benchmark)))
    previous = None
    for path in checkpoints:
        model = load_model(path, device)
        results = score(model, tokenizer, target, device,
                        args.max_new_tokens, args.batch_size, eos_id)
        history[path.stem] = results
        correct = sum(results.values())
        gained = lost = 0
        if previous is not None:
            gained = sum(1 for k in results if results[k] and not previous[k])
            lost = sum(1 for k in results if not results[k] and previous[k])
        per_bench = []
        for bench in sorted(by_benchmark):
            ids = [i["identifier"] for i in by_benchmark[bench]]
            hit = sum(results[i] for i in ids)
            per_bench.append(f"{hit}/{len(ids)}")
        step = path.stem.split("-")[-1]
        print(f"{step:>12}{correct:>9}{len(target):>7}{correct/len(target)*100:>7.1f}%"
              f"{gained:>8}{lost:>7}" + "".join(f"{v:>10}" for v in per_bench))
        previous = results
        del model
        torch.cuda.empty_cache()

    # When did each question first become correct and stay correct?
    steps = list(history)
    settled = {}
    for item in target:
        ident = item["identifier"]
        first = None
        for index, step in enumerate(steps):
            if all(history[s][ident] for s in steps[index:]):
                first = step
                break
        settled[ident] = first
    print("\nfirst checkpoint after which the answer stayed correct:")
    counts = {}
    for ident, step in settled.items():
        counts[step] = counts.get(step, 0) + 1
    cumulative = 0
    for step in steps:
        n = counts.get(step, 0)
        cumulative += n
        bar = "#" * round(n / max(1, len(target)) * 60)
        print(f"  {step.split('-')[-1]:>6} {n:>5} ({cumulative/len(target)*100:>5.1f}% cum) {bar}")
    never = counts.get(None, 0)
    if never:
        print(f"  {'never':>6} {never:>5}  (correct at the end but flipped along the way)")

    if args.json:
        args.json.write_text(json.dumps(
            {"target": [i["identifier"] for i in target],
             "history": history, "settled": settled}, indent=2) + "\n")
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
