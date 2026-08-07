"""Extend the concise SFT corpus's arithmetic coverage where it is thinnest.

Motivation (measured on the concise arm's own eval output). Accuracy falls
sharply with the magnitude of the gold answer:

    <10: 14.05% (n=1253)   10-99: 10.46% (n=2410)
    100-999: 5.12% (n=860)  1000+: **0.28%** (n=355)

and the corpus explains why. Of 9,606 synthetic-math-v1 records, answer
magnitudes run 314 / 3,367 / 5,814 / **111** across those same buckets: the
model has seen almost no arithmetic resolving above 1,000. Multiplication is
similarly thin (892 records against 3,776 additions).

Classifying the concise arm's 4,551 errors by whether the equation it wrote is
internally consistent: 1,746 state an equation whose arithmetic is simply wrong
(right operation, miscomputed), against 1,607 that compute correctly from a
wrong setup. The miscomputation half is what coverage can address; problem
setup is not a data-volume problem.

This generates additional records from the **six templates already present in
synthetic-math-v1**, reproducing their response strings exactly, sampled to
fill the magnitude and operation gaps. It invents no new task format, no new
phrasing, and no reasoning: the responses are arithmetic ground truth in the
corpus's existing wording, and every one is checked against Python's own
arithmetic before being written.

Decontamination is mandatory and mirrors the reference's: any generated
question whose normalised text collides with one of the 5,023 evaluation
questions is discarded. Generated items are also deduplicated against each
other and against the existing corpus.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from modern_lm.data import default_paths  # noqa: E402

SOURCE = "synthetic-math-v1-augmented"


def normalise(question: str) -> str:
    """Whitespace/case-insensitive key for overlap checks."""
    return re.sub(r"\s+", " ", question.strip().lower())


def add(a: int, b: int) -> tuple[str, str, str]:
    return (f"Calculate {a} + {b}.",
            f"Add the two numbers: {a} + {b} = {a + b}.\nFinal answer: {a + b}",
            str(a + b))


def subtract(a: int, b: int) -> tuple[str, str, str]:
    return (f"Calculate {a} - {b}.",
            f"Subtract: {a} - {b} = {a - b}.\nFinal answer: {a - b}",
            str(a - b))


def multiply(a: int, b: int) -> tuple[str, str, str]:
    return (f"Calculate {a} × {b}.",
            f"Multiply: {a} × {b} = {a * b}.\nFinal answer: {a * b}",
            str(a * b))


def collection(a: int, b: int) -> tuple[str, str, str]:
    return (f"A collection starts with {a} objects and receives {b} more. "
            f"How many objects are there now?",
            f"Add the starting amount and the new objects: {a} + {b} = {a + b}."
            f"\nFinal answer: {a + b}",
            str(a + b))


# Spelled-out numbers, the form ASDiv and SVAMP overwhelmingly use. The corpus
# barely contains them (7.3% of train questions), and the model scores 8.38% on
# the 1,098 evaluation questions that have them against 10.32% on digit-only
# questions -- it reads "Sandra took six cups" and writes "12 + 6".
NUMBER_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven",
                "eight", "nine", "ten", "eleven", "twelve", "thirteen",
                "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
                "nineteen", "twenty"]

WORD_ITEMS = ["apples", "marbles", "balloons", "pencils", "books", "stickers",
              "coins", "shells", "cards", "flowers"]


def word_sum(a: int, b: int, item: str) -> tuple[str, str, str]:
    """Spelled-out addends, digits in the response.

    The response deliberately restates the operands as digits: the skill being
    taught is reading a number word and computing with it, so the supervision
    has to show the translation rather than echo the words back.
    """
    return (f"{NUMBER_WORDS[a].capitalize()} {item} and {NUMBER_WORDS[b]} more "
            f"{item} are in the basket. How many {item} are in the basket?",
            f"Add the two amounts: {a} + {b} = {a + b}.\nFinal answer: {a + b}",
            str(a + b))


def word_difference(a: int, b: int, item: str) -> tuple[str, str, str]:
    return (f"There were {NUMBER_WORDS[a]} {item} in the basket. "
            f"{NUMBER_WORDS[b].capitalize()} {item} were taken out. "
            f"How many {item} are in the basket now?",
            f"Subtract what was taken: {a} - {b} = {a - b}.\n"
            f"Final answer: {a - b}",
            str(a - b))


def word_more_than(a: int, b: int, item: str) -> tuple[str, str, str]:
    """The "N more than" comparison, which the model reliably misreads."""
    return (f"Ana has {NUMBER_WORDS[b]} more {item} than Ben. "
            f"Ben has {NUMBER_WORDS[a]} {item}. How many {item} does Ana have?",
            f"Add the difference to Ben's amount: {a} + {b} = {a + b}.\n"
            f"Final answer: {a + b}",
            str(a + b))


# Tens words and the multiplicative relation words. Measured on the
# number-words arm: the 559 evaluation questions using "twice", "half", or a
# tens word score 3.76% against 11.31% overall -- "twice" (193) and "half"
# (172) are relations the corpus never expresses in that vocabulary at all.
# These templates measured as a null result; see docs/results-sft-number-words.md.
TENS_WORDS = {20: "twenty", 30: "thirty", 40: "forty", 50: "fifty",
              60: "sixty", 70: "seventy", 80: "eighty", 90: "ninety"}


def spell(value: int) -> str:
    """Spell an integer in the range these templates draw from (1-99)."""
    if value <= 20:
        return NUMBER_WORDS[value]
    tens, ones = divmod(value, 10)
    if ones == 0:
        return TENS_WORDS[tens * 10]
    return f"{TENS_WORDS[tens * 10]}-{NUMBER_WORDS[ones]}"


def word_twice(a: int, item: str) -> tuple[str, str, str]:
    return (f"Ben has {spell(a)} {item}. Ana has twice as many {item} as Ben. "
            f"How many {item} does Ana have?",
            f"Multiply by two: {a} × 2 = {a * 2}.\nFinal answer: {a * 2}",
            str(a * 2))


def word_half(a: int, item: str) -> tuple[str, str, str]:
    """`a` must be even; the caller guarantees it."""
    return (f"There were {spell(a)} {item} in the basket. "
            f"Half of the {item} were taken out. "
            f"How many {item} were taken out?",
            f"Divide by two: {a} / 2 = {a // 2}.\nFinal answer: {a // 2}",
            str(a // 2))


def word_tens_sum(a: int, b: int, item: str) -> tuple[str, str, str]:
    return (f"{spell(a).capitalize()} {item} and {spell(b)} more {item} are in "
            f"the basket. How many {item} are in the basket?",
            f"Add the two amounts: {a} + {b} = {a + b}.\nFinal answer: {a + b}",
            str(a + b))


def split_groups(total: int, groups: int) -> tuple[str, str, str]:
    quotient = total // groups
    return (f"A total of {total} items are split equally into {groups} groups. "
            f"How many items are in each group?",
            f"Divide the total by the number of groups: {total} / {groups} = "
            f"{quotient}.\nFinal answer: {quotient}",
            str(quotient))


def solve_linear(coefficient: int, constant: int, root: int) -> tuple[str, str, str]:
    total = coefficient * root + constant
    remainder = total - constant
    return (f"Solve for x: {coefficient}x + {constant} = {total}.",
            f"Subtract {constant}: {coefficient}x = {remainder}. Then divide by "
            f"{coefficient}: x = {root}.\nFinal answer: {root}",
            str(root))


LINEAR = re.compile(r"Solve for x: (\d+)x \+ (\d+) = (\d+)\.")


def verify(question: str, response: str, answer: str) -> None:
    """Re-derive the stated arithmetic and confirm the corpus teaches no lie.

    Cheap relative to the cost of silently training on wrong arithmetic, and it
    has already earned its place: it caught the linear template, whose response
    states `69x = 21114` rather than a plain binary operation.
    """
    if not response.endswith(f"Final answer: {answer}"):
        raise ValueError(f"answer/response mismatch in {response!r}")

    linear = LINEAR.match(question)
    if linear is not None:
        coefficient, constant, total = (int(linear.group(1)), int(linear.group(2)),
                                        int(linear.group(3)))
        if coefficient * int(answer) + constant != total:
            raise ValueError(f"bad linear solution: {question!r} -> {answer}")
        if f"{coefficient}x = {total - constant}" not in response:
            raise ValueError(f"linear intermediate wrong in {response!r}")
        return

    match = re.search(r"(-?\d+)\s*([+\-×/])\s*(-?\d+)\s*=\s*(-?\d+)", response)
    if match is None:
        raise ValueError(f"no verifiable equation in {response!r}")
    left, operator, right, stated = (int(match.group(1)), match.group(2),
                                     int(match.group(3)), int(match.group(4)))
    expected = {"+": left + right, "-": left - right,
                "×": left * right, "/": left // right}[operator]
    if expected != stated:
        raise ValueError(f"bad arithmetic: {left}{operator}{right}={stated}")
    if str(expected) != answer:
        raise ValueError(f"equation does not yield the answer in {response!r}")


def generate(count: int, rng: random.Random, number_words: bool = False,
             extended_words: bool = False) -> list[tuple[str, str, str]]:
    """Sample records skewed to the magnitudes and operations the corpus lacks.

    `number_words` switches to the spelled-out-operand templates instead; it is
    a separate arm rather than a mixture so the two interventions stay
    independently attributable.
    """
    items: list[tuple[str, str, str]] = []
    if number_words:
        # The three base templates are the 568 arm. `extended_words` adds the
        # twice/half/tens forms, which measured as a null result (568 -> 575,
        # p = 0.78); it stays opt-in so the recommended recipe above keeps
        # reproducing byte-identically.
        kinds = ["sum", "difference", "more_than"]
        if extended_words:
            kinds += ["twice", "half", "tens_sum"]
        while len(items) < count:
            kind = rng.choice(kinds)
            item = rng.choice(WORD_ITEMS)
            if kind == "twice":
                items.append(word_twice(rng.randint(2, 50), item))
            elif kind == "half":
                items.append(word_half(rng.randrange(2, 60, 2), item))
            elif kind == "tens_sum":
                items.append(word_tens_sum(rng.randint(21, 99),
                                           rng.randint(2, 20), item))
            else:
                a, b = rng.randint(1, 20), rng.randint(1, 20)
                if kind == "difference":
                    a, b = max(a, b), min(a, b)
                builder = {"sum": word_sum, "difference": word_difference,
                           "more_than": word_more_than}[kind]
                items.append(builder(a, b, item))
        return items
    while len(items) < count:
        kind = rng.choice(["add", "sub", "mul", "mul", "collection",
                           "split", "linear"])
        # Deliberately biased high: the >=1000 bucket is the empty one.
        big = lambda: rng.randint(1000, 99999)  # noqa: E731
        mid = lambda: rng.randint(100, 9999)    # noqa: E731
        if kind == "add":
            items.append(add(big(), mid()))
        elif kind == "sub":
            a, b = big(), mid()
            items.append(subtract(max(a, b), min(a, b)))
        elif kind == "mul":
            items.append(multiply(rng.randint(11, 999), rng.randint(11, 99)))
        elif kind == "collection":
            items.append(collection(big(), mid()))
        elif kind == "split":
            groups = rng.randint(2, 99)
            items.append(split_groups(groups * rng.randint(11, 999), groups))
        else:
            items.append(solve_linear(rng.randint(2, 99), rng.randint(2, 999),
                                      rng.randint(11, 999)))
    return items


def main() -> None:
    paths = default_paths()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path,
                        default=Path("data/sft-math-concise/train.jsonl"))
    parser.add_argument("--evaluation", type=Path, default=paths["evaluation"])
    parser.add_argument("--output", type=Path,
                        default=Path("data/sft-math-concise-aug/train.jsonl"))
    parser.add_argument("--count", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=2029)
    parser.add_argument("--exclude", type=Path, action="append", default=[],
                        help="JSONL whose questions must not be generated; use "
                             "when building a heldout split against a train "
                             "split, since a different seed alone does not "
                             "prevent the templates from colliding")
    parser.add_argument("--number-words", action="store_true",
                        help="generate spelled-out-operand records instead of "
                             "the large-magnitude ones")
    parser.add_argument("--extended-words", action="store_true",
                        help="with --number-words, also emit the twice/half/"
                             "tens templates (measured null: 568 -> 575, "
                             "p = 0.78; off by default)")
    args = parser.parse_args()

    # Iterate the handle rather than str.splitlines(): some source questions
    # contain U+2028/U+0085, which splitlines() treats as line breaks and JSONL
    # readers (including the trainer's) do not. Splitting on them corrupts the
    # very records this script is meant to preserve.
    with args.base.open() as handle:
        base = [json.loads(line) for line in handle]
    with args.evaluation.open() as handle:
        blocked = {normalise(json.loads(line)["question"]) for line in handle}
    for extra in args.exclude:
        with extra.open() as handle:
            blocked |= {normalise(json.loads(line)["question"]) for line in handle}
    existing = {normalise(record["question"]) for record in base}

    rng = random.Random(args.seed)
    kept: list[dict] = []
    skipped_eval = 0
    skipped_duplicate = 0
    seen = set(existing)

    while len(kept) < args.count:
        for question, response, answer in generate(args.count - len(kept), rng,
                                                   args.number_words,
                                                   args.extended_words):
            verify(question, response, answer)
            key = normalise(question)
            if key in blocked:
                skipped_eval += 1
                continue
            if key in seen:
                skipped_duplicate += 1
                continue
            seen.add(key)
            kept.append({"source": SOURCE,
                         "identifier": hashlib.sha256(key.encode()).hexdigest()[:16],
                         "question": question, "response": response,
                         "answer": answer})
            if len(kept) >= args.count:
                break

    args.output.parent.mkdir(parents=True, exist_ok=True)
    records = base + kept
    args.output.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n")
    report = {"base_records": len(base), "generated": len(kept),
              "total": len(records), "skipped_evaluation_overlap": skipped_eval,
              "skipped_duplicate": skipped_duplicate, "seed": args.seed,
              "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest()}
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
