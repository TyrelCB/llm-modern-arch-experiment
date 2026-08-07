"""Tests for the concise-response SFT rewrite.

The rewrite decides what the eighth arm is *trained to say*, and its failure
mode is silent: a rule that emits a plausible-looking but ungrounded response
still trains, still converges, and simply teaches the model to state numbers it
cannot derive. The grounding predicate is therefore pinned here against the
specific cases that were rejected during development, so a later loosening of
it has to be deliberate.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from prepare_concise_sft import concise_response, convert, is_grounded, numbers_in


def test_numbers_normalises_separators_and_trailing_decimals():
    assert numbers_in("1,000 and 1000 and 18.0") == {"1000", "18"}


def test_single_line_response_passes_through_byte_identical():
    record = {"question": "Calculate 701 + 92.", "answer": "793",
              "response": "Add the two numbers: 701 + 92 = 793.\nFinal answer: 793"}
    response, kind = concise_response(record)
    assert kind == "unchanged"
    assert response == record["response"]


def test_last_line_referencing_own_intermediates_is_kept():
    # 12 is derived in line 1, not present in the question; the line is still a
    # real computation ending in the gold answer, so it survives.
    record = {
        "question": "Out of 30 clients, 7 need vegan and 8 need kosher meals, "
                    "and 3 need both. How many need neither?",
        "answer": "18",
        "response": ("7 + 8 = 15 meals\n15 - 3 = 12 meals\n"
                     "30 meals - 12 meals = 18 meals\nFinal answer: 18"),
    }
    response, kind = concise_response(record)
    assert kind == "quoted_final_step"
    assert response == "30 meals - 12 meals = 18 meals\nFinal answer: 18"


def test_bare_algebraic_unknown_is_rejected():
    # "x=22 quarters" reads as a derivation but names a variable the concise
    # response never introduces.
    assert not is_grounded("x=22 quarters", "John has quarters and dimes.", "22", "")


def test_line_without_an_equation_is_rejected():
    assert not is_grounded("So he has more apples than before", "q", "5", "")


def test_ungrounded_number_is_rejected():
    # 99 comes from nowhere: not the question, not the answer, not earlier.
    assert not is_grounded("99 - 81 = 18", "Megan has 30 clients.", "18", "")


def test_ungroundable_multistep_record_is_dropped_not_flattened():
    # The historic bug: collapsing this to "Final answer: 22" would supervise a
    # number the response never derives.
    record = {
        "question": "Dorothy spent $53 on ingredients. She made 25 doughnuts "
                    "and sells each for $3. What was her profit?",
        "answer": "22",
        "response": ("She earned 25 x $3 = $75.\n"
                     "Let p be profit, so p = $75 - $53 = $22.\nFinal answer: 22"),
    }
    response, kind = concise_response(record)
    assert kind == "dropped"
    assert response is None


def test_keep_lines_two_retains_the_last_two_steps():
    """The control arm must differ from the concise arm in exactly one integer."""
    record = {
        "question": "Out of 30 clients, 7 need vegan and 8 need kosher meals, "
                    "and 3 need both. How many need neither?",
        "answer": "18",
        "response": ("7 + 8 = 15 meals\n15 - 3 = 12 meals\n"
                     "30 meals - 12 meals = 18 meals\nFinal answer: 18"),
    }
    response, kind = concise_response(record, keep_lines=2)
    assert kind == "quoted_final_step"
    assert response == ("15 - 3 = 12 meals\n30 meals - 12 meals = 18 meals\n"
                        "Final answer: 18")


def test_keep_lines_two_leaves_a_two_line_response_unchanged():
    record = {"question": "Dorothy spent $53, made 25 doughnuts at $3.",
              "answer": "22",
              "response": ("She earned 25 x $3 = $75.\n"
                           "Her profit is $75 - $53 = $22.\nFinal answer: 22")}
    response, kind = concise_response(record, keep_lines=2)
    assert kind == "unchanged"
    assert response == record["response"]


def test_convert_drops_records_and_reports_counts(tmp_path):
    rows = [
        {"question": "Calculate 5 + 5.", "answer": "10",
         "response": "Add: 5 + 5 = 10.\nFinal answer: 10"},
        {"question": "A has 40, B has 15. 30 and 7 have holes. How many without?",
         "answer": "18",
         "response": ("40-30 = 10 balls\n15-7 = 8 balls\n"
                      "A total of 10+8 = 18 balls\nFinal answer: 18")},
        {"question": "Unrecoverable.", "answer": "3",
         "response": "Reason vaguely about it\nTherefore it is so\nFinal answer: 3"},
    ]
    source = tmp_path / "train.jsonl"
    source.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    destination = tmp_path / "out.jsonl"

    report = convert(source, destination)

    assert report["source_records"] == 3
    assert report["records"] == 2
    assert report["outcomes"]["dropped"] == 1
    written = [json.loads(line) for line in destination.read_text().splitlines()]
    assert [r["response"] for r in written] == [
        "Add: 5 + 5 = 10.\nFinal answer: 10",
        "A total of 10+8 = 18 balls\nFinal answer: 18",
    ]


def test_generated_arithmetic_is_verified_and_word_operands_use_digits():
    """Both augmentation modes must survive their own arithmetic checker.

    `verify` is the only thing standing between a generated corpus and training
    the model on wrong arithmetic, so it is exercised over a real sample rather
    than trusted.
    """
    import random

    sys.path.insert(0, str(REPO / "scripts"))
    from augment_arithmetic_sft import (
        NUMBER_WORDS, TENS_WORDS, generate, verify,
    )

    spelled = set(NUMBER_WORDS) | set(TENS_WORDS.values())

    rng = random.Random(7)
    for question, response, answer in generate(300, rng):
        verify(question, response, answer)

    rng = random.Random(7)
    word_records = generate(200, rng, number_words=True)
    for question, response, answer in word_records:
        verify(question, response, answer)
        # The question spells its operands; the response must restate them as
        # digits, since translating is the skill being taught. All four
        # operators appear -- the twice/half templates use × and /.
        assert any(w in question.lower() for w in spelled)
        assert re.search(r"\d+\s*[+\-×/]\s*\d+\s*=\s*\d+", response)
        assert response.endswith(f"Final answer: {answer}")


def test_every_emitted_response_is_verbatim_source_text(tmp_path):
    """No emitted reasoning line may be text the source response did not contain.

    This is the property that separates this arm from data augmentation.
    """
    rows = [
        {"question": "Calculate 9 + 6.", "answer": "15",
         "response": "Add: 9 + 6 = 15.\nFinal answer: 15"},
        {"question": "Annie bought 5 TVs at $50 with $260 total, 10 figurines.",
         "answer": "1",
         "response": ("Annie spent 5 * $50 = $250 on televisions.\n"
                      "She has $260 - $250 = $10 left.\n"
                      "A single figurine costs $10/10=$1.\nFinal answer: 1")},
    ]
    source = tmp_path / "train.jsonl"
    source.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    destination = tmp_path / "out.jsonl"
    convert(source, destination)

    originals = {r["response"] for r in rows}
    for line in destination.read_text().splitlines():
        emitted = json.loads(line)["response"]
        reasoning = emitted.split("Final answer:")[0].strip()
        assert any(reasoning in original for original in originals)
