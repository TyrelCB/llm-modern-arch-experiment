"""The answer ends where the model starts inventing its own follow-up questions."""
from modern_lm.evaluate_benchmarks import answer_segment


def test_stops_at_invented_question():
    completion = (" 13x + 1 = 703\n13x = 702\nx = 54\n\n"
                  "Question: Solve for x: 5x + 1 = 703.\nAnswer: x = 142")
    assert answer_segment(completion).strip().endswith("x = 54")


def test_stops_at_abbreviated_question():
    assert answer_segment(" The answer is 7.\nQ: What is 2+2?\nA: 4").strip() == "The answer is 7."


def test_stops_at_numbered_exercise():
    assert answer_segment(" total 12 apples\n2. Mary has 3 pears.").strip() == "total 12 apples"


def test_keeps_explanation_of_the_same_answer():
    # "Explanation:" continues the current answer; it does not start a new problem.
    completion = " 3 x 60 = 180 meters\n\nExplanation:\n\n3 x 60 = 180"
    assert answer_segment(completion) == completion


def test_leaves_a_clean_answer_untouched():
    assert answer_segment(" x = 45") == " x = 45"


def test_scoring_reads_the_answer_not_the_ramble():
    # Regression: extract_number takes the last number in the text, so a correct
    # derivation followed by an invented question used to score on that question.
    from modern_lm.evaluate_benchmarks import extract_number
    completion = " x = 52\n\nQuestion: Solve for x: 3x + 1 = 10.\nAnswer: x = 3"
    assert extract_number(completion) == "3"
    assert extract_number(answer_segment(completion)) == "52"
