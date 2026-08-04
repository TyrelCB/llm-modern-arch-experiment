"""Pack a ~2B-token FineMath-4+ corpus reusing the reference's exact tokenizer.

Critical: the tokenizer is **loaded**, never retrained. A freshly trained BPE on
a larger sample would produce different token ids and different sequence
lengths, making held-out loss incomparable to the completed 250M baseline and to
the DeepSeek-V4 reference. The split seed, holdout permille, and 13-gram
decontamination are likewise reused verbatim, so a document held out at 260M is
still held out at 2B and no benchmark prompt leaks into training.

Source: HuggingFaceTB/finemath, `finemath-4plus` (ODC-By 1.0).
Benchmarks used only for decontamination: GSM8K, SVAMP, ASDiv.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from tokenizers import Tokenizer

DEEPSEEK_REPO = Path("/home/tyrel/projects/llm-deepseek-v4-experiment")
sys.path.insert(0, str(DEEPSEEK_REPO / "src"))

from deepseek_v4.data import (  # noqa: E402
    Decontaminator, iter_parquet_documents, load_contamination_examples,
    pack_documents,
)

REFERENCE_DATA = DEEPSEEK_REPO / "data" / "finemath-4plus"
SPLIT_SEED = "deepseek-v4-finemath-v1"
HOLDOUT_PERMILLE = 10


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet-dir", type=Path,
                        default=DEEPSEEK_REPO / "data" / "raw" / "finemath-4plus")
    parser.add_argument("--output", type=Path,
                        default=DEEPSEEK_REPO / "data" / "finemath-2b")
    parser.add_argument("--tokenizer", type=Path, default=REFERENCE_DATA / "tokenizer.json")
    parser.add_argument("--train-tokens", type=int, default=2_050_000_000)
    parser.add_argument("--heldout-tokens", type=int, default=5_000_000)
    parser.add_argument("--contamination", type=Path, nargs="*",
                        default=[DEEPSEEK_REPO / "data" / "benchmarks" / "decontamination.jsonl"])
    args = parser.parse_args()

    shards = sorted(args.parquet_dir.glob("train-*.parquet"))
    if not shards:
        raise SystemExit(f"no parquet shards under {args.parquet_dir}")
    print(json.dumps({"event": "start", "shards": len(shards),
                      "tokenizer": str(args.tokenizer)}), flush=True)

    # Load, never retrain -- see module docstring.
    tokenizer = Tokenizer.from_file(str(args.tokenizer))

    examples = load_contamination_examples(args.contamination)
    metadata = pack_documents(
        iter_parquet_documents(shards), tokenizer, args.output,
        source_files=shards, split_seed=SPLIT_SEED,
        holdout_permille=HOLDOUT_PERMILLE,
        max_train_tokens=args.train_tokens,
        max_heldout_tokens=args.heldout_tokens,
        decontaminator=Decontaminator(examples),
    )
    # Keep the tokenizer beside the bins so downstream paths resolve locally,
    # but copy the reference's file rather than emitting a new one.
    (args.output / "tokenizer.json").write_bytes(args.tokenizer.read_bytes())
    print(json.dumps(asdict(metadata), indent=2), flush=True)


if __name__ == "__main__":
    main()
