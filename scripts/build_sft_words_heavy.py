#!/usr/bin/env python3
"""A word-problem-heavy SFT corpus, to test whether composition matters.

`data/sft-math-words` is 82.4% bare synthetic arithmetic (`Calculate 805 - 53`)
and 17.6% word problems, while the 5,024-question benchmark is 92% word
problems (asdiv 45.9%, gsm8k 26.3%, svamp 19.9%) and 8% arithmetic/algebra.

Measured on the 50M cosine model, echo-corrected, the corpus share does not
predict the gain -- if anything it anti-predicts:

    subset      SFT share   base    after SFT   multiple
    arithmetic  82.4%       2.00%   7.00%        3.5x
    gsm8k       17.6%       0.23%   3.26%       14.3x
    algebra     0%          1.00%   23.00%      23.0x
    asdiv       0%          0.82%   10.80%      13.1x
    svamp       0%          0.20%   12.00%      60.0x

The subset that dominates the corpus improved least; the two with zero
representation improved most. That suggests SFT's value here is teaching the
model to answer and stop rather than teaching arithmetic, and that the corpus
is badly matched to the eval.

This builds the inverted corpus: every available gsm8k-train record (7,073 in
the upstream pool, ~1.7x what sft-math-words uses) plus a small arithmetic
floor, so word problems dominate. Same record schema, same prompt format, same
decontamination -- the upstream pool already dropped 13 evaluation overlaps and
deduplicated 1,903 records, and this only subsets it.

The matched invocation uses `--repeat-words 3 --arithmetic-floor 2600`, yielding
23,819 records versus sft-math-words' 23,780. The default one-copy mode is a
smaller exploratory corpus and must not be called composition-only.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

UPSTREAM = Path("/home/tyrel/projects/llm-deepseek-v4-experiment/data/sft-math")


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path,
                        default=Path("data/sft-math-wordheavy"))
    parser.add_argument("--arithmetic-floor", type=int, default=3000,
                        help="synthetic records to keep, so the format is not "
                             "lost entirely")
    parser.add_argument("--repeat-words", type=int, default=1,
                        help="duplicate the word-problem pool N times. The "
                             "gsm8k pool caps at 7,073, so matching "
                             "sft-math-words' 23,780 total needs repetition; "
                             "without it, size and composition are confounded.")
    parser.add_argument("--balanced", action="store_true",
                        help="proportion the corpus to the BENCHMARK instead: "
                             "66%% short one-step word problems (the augmented "
                             "source, closest proxy for asdiv/svamp), 26%% "
                             "gsm8k multi-step, 8%% bare arithmetic")
    parser.add_argument("--total", type=int, default=24000,
                        help="balanced mode only: target corpus size")
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    train = load(UPSTREAM / "train.jsonl")
    heldout = load(UPSTREAM / "heldout.jsonl")

    words = [r for r in train if r["source"].startswith("gsm8k")]
    synthetic = [r for r in train if not r["source"].startswith("gsm8k")]

    if args.balanced:
        # The benchmark is 45.9% asdiv + 19.9% svamp (short, one-step),
        # 26.3% gsm8k (multi-step), 8% arithmetic/algebra. sft-math-words has
        # no asdiv/svamp source; the augmented records are word-framed
        # one-step problems, which is the closest available proxy.
        local = load(Path("data/sft-math-words/train.jsonl"))
        short = [r for r in local if r["source"].endswith("augmented")]
        bare = [r for r in local if r["source"] == "synthetic-math-v1"]
        rng.shuffle(short); rng.shuffle(bare); rng.shuffle(words)
        want_short = int(0.66 * args.total)
        want_multi = int(0.26 * args.total)
        want_bare = args.total - want_short - want_multi

        def take(pool, n):
            if not pool:
                return []
            return [pool[i % len(pool)] for i in range(n)]

        out_train = (take(short, want_short) + take(words, want_multi)
                     + take(bare, want_bare))
    else:
        rng.shuffle(synthetic)
        kept = synthetic[:args.arithmetic_floor]
        out_train = words * args.repeat_words + kept
    rng.shuffle(out_train)

    # Heldout keeps the same composition ratio so its loss is comparable.
    if args.balanced:
        local_ho = load(Path("data/sft-math-words/heldout.jsonl"))
        ho_short = [r for r in local_ho if r["source"].endswith("augmented")]
        ho_bare = [r for r in local_ho if r["source"] == "synthetic-math-v1"]
        ho_words = [r for r in heldout if r["source"].startswith("gsm8k")]
        rng.shuffle(ho_short); rng.shuffle(ho_bare); rng.shuffle(ho_words)
        heldout_total = min(600, len(ho_short) + len(ho_bare) + len(ho_words))
        out_heldout = (ho_short[:int(0.66 * heldout_total)]
                       + ho_words[:int(0.26 * heldout_total)])
        out_heldout += ho_bare[:heldout_total - len(out_heldout)]
        rng.shuffle(out_heldout)
        _write(args, out_train, out_heldout)
        return

    ho_words = [r for r in heldout if r["source"].startswith("gsm8k")]
    ho_syn = [r for r in heldout if not r["source"].startswith("gsm8k")]
    rng.shuffle(ho_syn)
    ratio = args.arithmetic_floor / max(1, len(words) * args.repeat_words)
    out_heldout = ho_words + ho_syn[:int(len(ho_words) * ratio)]
    rng.shuffle(out_heldout)

    _write(args, out_train, out_heldout)


def _write(args, out_train, out_heldout) -> None:
    args.out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", out_train), ("heldout", out_heldout)):
        with (args.out / f"{name}.jsonl").open("w") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")

    counts = Counter(r["source"] for r in out_train)
    total = len(out_train)
    word_share = sum(v for k, v in counts.items()
                     if k.startswith("gsm8k") or k.endswith("augmented"))
    metadata = {
        "format_version": 1,
        "seed": args.seed,
        "mode": "balanced" if args.balanced else "word-heavy",
        "built_from": str(UPSTREAM),
        "arithmetic_floor": args.arithmetic_floor,
        "repeat_words": args.repeat_words,
        "counts": {"train": total, "heldout": len(out_heldout)},
        "sources": dict(counts),
        "word_problem_share": round(word_share / total, 4),
        "note": ("Matches benchmark composition with short/multi-step/bare "
                 "proxies." if args.balanced else
                 "Inverts sft-math-words (82.4% synthetic) to test whether "
                 "corpus composition predicts per-subset benchmark gains."),
    }
    (args.out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    print(f"wrote {args.out}")
    print(f"  train {total:,}  heldout {len(out_heldout):,}")
    for source, count in counts.most_common():
        print(f"    {source:32s} {count:>6,}  {100 * count / total:5.1f}%")
    print(f"  word-problem share: {100 * word_share / total:.1f}% "
          f"(sft-math-words: 17.6%)")


if __name__ == "__main__":
    main()
