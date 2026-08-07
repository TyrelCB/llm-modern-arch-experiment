"""Build a concise-response SFT corpus from the shared math/instruction data.

Motivation (measured, not assumed). On the 2B+SFT checkpoint scored at a
256-token budget -- where 95% of completions terminate properly -- accuracy
falls monotonically with the number of reasoning lines the model emits:

    1 line: 22.00% (n=650)    2 lines: 6.57% (n=2207)
    3 lines:  3.87% (n=1112)  4 lines: 1.78% (n=393)   5+: ~2-3%

Each additional generated line is another opportunity for a 145M model to
corrupt a result it had already computed correctly. The same checkpoint's
"oracle" rate -- some number *somewhere* in the completion matches the gold
answer -- is 17.9% at 32 tokens and 21.4% at 256, against 8.20% and 7.13%
actually scored. The model produces the right number two to three times more
often than it gets credit for; what it lacks is the discipline to stop once it
has one.

The existing SFT corpus teaches the opposite. Its gsm8k-train half is
multi-step chain-of-thought (2-8 reasoning lines), which for a model of this
capacity is a liability rather than an aid: it is trained to keep going.

This script rewrites that supervision target to "compute, state the answer,
stop", while leaving the *questions* and the train/heldout split untouched so
the arm stays comparable to every prior one.

Selection rule, applied per record:

  - Responses already at one reasoning line (all 9,606 synthetic-math-v1
    records) are kept unchanged. They are already the target format.
  - Multi-step responses are kept only when their last reasoning line is
    *grounded*: it contains an `=`, names no bare algebraic unknown, and every
    number it references appears in the question, in the answer, or earlier in
    that same response. Such a record is replaced by that line plus its
    unchanged `Final answer: N`.
  - Every other multi-step record is **dropped**.

Allowing the last line to reference intermediates derived earlier in its own
response (4,174 of 7,073 gsm8k-train records, against 146 under the stricter
question-only rule) is what keeps natural-language word problems in the
corpus. The supervision stays truthful -- the line is verbatim source text
ending in the gold answer -- and the demand it places on the model is exactly
the target behaviour: do the intermediate work implicitly, emit one step, stop.

Dropping rather than truncating is the load-bearing choice, and it was made
after inspecting what truncation produced. Collapsing a multi-step response to
a bare `Final answer: 22` turns "Dorothy spent $53 on ingredients ..." into
supervision for stating an unjustified number -- training the model to guess
confidently, which is a different and worse failure than rambling. Quoting a
non-self-contained last line is equally unsound: lines like `x=22 quarters`
reference a variable the concise response never introduces. This arm changes
*how much* the model is taught to say, so it must not also fabricate content;
every response emitted here is verbatim source text.

The cost is corpus size: the gsm8k-train half is largely multi-step, so the
concise corpus is smaller than the original. That is a real trade -- fewer
examples against a supervision target that matches what the model can actually
execute -- and the evaluation is what settles it.

Both the prompt format and the `Final answer:` sentinel are preserved exactly,
because the evaluation harness matches on them and this arm changes only the
training target, never the scorer.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

DEEPSEEK_REPO = Path("/home/tyrel/projects/llm-deepseek-v4-experiment")
SFT_DATA_DIR = DEEPSEEK_REPO / "data" / "sft-math"

NUMBER = re.compile(r"-?\d[\d,]*\.?\d*")


def numbers_in(text: str) -> set[str]:
    """Numeric tokens in `text`, normalised so 1,000 and 1000 compare equal."""
    found = set()
    for raw in NUMBER.findall(text):
        cleaned = raw.replace(",", "").rstrip(".")
        if cleaned in ("", "-"):
            continue
        try:
            value = float(cleaned)
        except ValueError:
            continue
        # Normalise 18 and 18.0 to one key so a restated integer matches.
        found.add(str(int(value)) if value == int(value) else str(value))
    return found


def reasoning_lines(response: str) -> list[str]:
    body = response.split("Final answer:")[0]
    return [line.strip() for line in body.strip().split("\n") if line.strip()]


# A bare algebraic unknown ("x=22 quarters", "3x=66"). Such a line reads as a
# derivation but names a variable the concise response never introduces, so it
# is rejected even when its numbers check out.
VARIABLE = re.compile(r"(?<![A-Za-z])[a-z](?![A-Za-z])")


def is_grounded(line: str, question: str, answer: str, earlier: str) -> bool:
    """True when `line` is a real computation the model could be asked to emit.

    Every number it mentions must trace back to the question, the answer, or an
    intermediate the source response itself derived; a line resting on numbers
    from nowhere would be teaching the model to invent quantities.
    """
    if "=" not in line:
        return False
    if VARIABLE.search(line):
        return False
    available = numbers_in(question) | numbers_in(answer) | numbers_in(earlier)
    return numbers_in(line).issubset(available)


def concise_response(record: dict, keep_lines: int = 1) -> tuple[str | None, str]:
    """Return (response, kind); a None response means drop the record.

    `keep_lines` is how many trailing reasoning lines to retain. It exists so
    the two-step control arm is built by this same code path rather than a
    parallel script -- the arms differ in one integer, which is what makes
    "two steps scores worse than one" a statement about step count.
    """
    response = record["response"]
    answer = str(record["answer"])
    lines = reasoning_lines(response)

    if len(lines) <= keep_lines:
        # Already short enough; keep the record byte-identical to the source.
        return response, "unchanged"

    last = lines[-1]
    if is_grounded(last, record["question"], answer, " ".join(lines[:-1])):
        kept = "\n".join(lines[-keep_lines:])
        return f"{kept}\nFinal answer: {answer}", "quoted_final_step"
    return None, "dropped"


def convert(source: Path, destination: Path, keep_lines: int = 1) -> dict:
    counts: dict[str, int] = {}
    kept = 0
    seen = 0
    with source.open() as handle, destination.open("w") as out:
        for line in handle:
            record = json.loads(line)
            response, kind = concise_response(record, keep_lines)
            counts[kind] = counts.get(kind, 0) + 1
            seen += 1
            if response is None:
                continue
            kept += 1
            out.write(json.dumps({**record, "response": response},
                                 ensure_ascii=False) + "\n")
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    return {"path": str(destination), "source_records": seen, "records": kept,
            "outcomes": counts, "sha256": digest,
            "bytes": destination.stat().st_size}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=SFT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("data/sft-math-concise"))
    parser.add_argument("--keep-lines", type=int, default=1,
                        help="trailing reasoning lines to retain "
                             "(1 = the concise arm, 2 = the two-step control)")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {"derived_from": str(args.source_dir),
                "keep_lines": args.keep_lines, "splits": {}}
    for split in ("train", "heldout"):
        metadata["splits"][split] = convert(args.source_dir / f"{split}.jsonl",
                                            args.output_dir / f"{split}.jsonl",
                                            args.keep_lines)
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
