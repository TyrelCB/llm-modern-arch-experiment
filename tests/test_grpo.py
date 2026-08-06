"""Tests for the GRPO stage.

The contamination test is the important one. GRPO optimizes directly against
the scorer, so if a benchmark question were ever in the prompt pool the headline
number would be measuring memorization of its own reward signal. Every other
arm in this project is protected by the corpus split; this arm needs the check
made explicit because it is the first one that trains on model-generated text.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from modern_lm.config import ModernConfig
from modern_lm.data import default_paths
from modern_lm.grpo import (GRPOSettings, completion_logprobs, group_advantages,
                            load_prompts, rollout, score_completion, SFT_DATA_DIR)
from modern_lm.model import ModernLM

PROMPTS = SFT_DATA_DIR / "train.jsonl"
EVAL_EXAMPLES = default_paths()["evaluation"]

requires_sft_data = pytest.mark.skipif(not PROMPTS.exists(),
                                       reason="SFT corpus not present")
requires_eval_data = pytest.mark.skipif(not EVAL_EXAMPLES.exists(),
                                        reason="benchmark examples not present")


def _normalize(text: str) -> str:
    return " ".join(text.split()).strip().lower()


def _digest(text: str) -> str:
    return hashlib.sha256(_normalize(text).encode("utf-8")).hexdigest()


@requires_sft_data
@requires_eval_data
def test_prompt_pool_does_not_contain_benchmark_questions():
    """Registered in docs/grpo-protocol.md: zero overlap, by hashed question."""
    pool = {_digest(record.question) for record in load_prompts(PROMPTS)}
    benchmark = set()
    with EVAL_EXAMPLES.open() as handle:
        for line in handle:
            benchmark.add(_digest(json.loads(line)["question"]))
    overlap = pool & benchmark
    assert not overlap, f"{len(overlap)} benchmark questions found in the GRPO prompt pool"


@requires_sft_data
def test_prompt_pool_answers_are_all_numeric():
    """A non-numeric gold answer would score 0.0 forever and silently poison
    its group's advantage with a reward the policy cannot ever earn."""
    from modern_lm.grpo import extract_number
    for record in load_prompts(PROMPTS):
        assert extract_number(record.answer) is not None, record.identifier


def test_score_completion_uses_reference_scorer():
    assert score_completion("The answer is 18", "18") == 1.0
    assert score_completion("Final answer: 18", "18") == 1.0
    assert score_completion("Final answer: 19", "18") == 0.0
    assert score_completion("no number here", "18") == 0.0


def test_group_advantages_are_zero_when_group_is_unanimous():
    """All-correct and all-wrong groups carry no preference signal, so they must
    contribute exactly zero gradient rather than a divide-by-noise value."""
    all_right = torch.ones(8)
    all_wrong = torch.zeros(8)
    assert torch.allclose(group_advantages(all_right, 8), torch.zeros(8))
    assert torch.allclose(group_advantages(all_wrong, 8), torch.zeros(8))


def test_group_advantages_are_centered_and_signed():
    rewards = torch.tensor([1.0, 1.0, 0.0, 0.0])
    advantages = group_advantages(rewards, 4)
    assert pytest.approx(float(advantages.mean()), abs=1e-5) == 0.0
    assert advantages[0] > 0 and advantages[2] < 0


def test_group_advantages_are_computed_per_group_not_globally():
    """Two groups with different difficulty must not leak into each other: a
    correct answer in a hard group and one in an easy group get different
    advantages, which is the whole point of the group-relative baseline."""
    rewards = torch.tensor([1.0, 0.0, 0.0, 0.0,
                            1.0, 1.0, 1.0, 0.0])
    advantages = group_advantages(rewards, 4)
    assert advantages[0] > advantages[4]


def _tiny_model() -> ModernLM:
    config = ModernConfig(vocab_size=64, n_layers=2, n_heads=2, n_kv_heads=2,
                          dim=32, ffn_dim=64, max_seq_len=64)
    torch.manual_seed(0)
    return ModernLM(config)


def test_completion_logprobs_align_with_manual_forward():
    """Off-by-one here would train on the wrong token and still look plausible,
    so the alignment is pinned against an explicit reference computation."""
    model = _tiny_model().eval()
    sequences = torch.randint(0, 64, (2, 12))
    width = 5
    device = torch.device("cpu")

    got = completion_logprobs(model, sequences, width, amp=False, device=device)

    with torch.no_grad():
        logits = model(sequences).logits
    expected = []
    for row in range(sequences.shape[0]):
        row_values = []
        for position in range(width, sequences.shape[1]):
            distribution = torch.log_softmax(logits[row, position - 1].float(), dim=-1)
            row_values.append(distribution[sequences[row, position]])
        expected.append(torch.stack(row_values))
    expected = torch.stack(expected)

    assert got.shape == (2, sequences.shape[1] - width)
    assert torch.allclose(got, expected, atol=1e-5)


def test_rollout_masks_tokens_after_eos():
    """Padding past <eos> must not be trained on or counted in the KL."""
    model = _tiny_model().eval()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(0)
    sequences, mask, lengths = rollout(
        model, [[1, 2, 3]], group_size=4, max_new_tokens=8, temperature=1.0,
        eos_id=3, pad_id=0, device=torch.device("cpu"), amp=False,
        microbatch=4, generator=generator)

    assert sequences.shape[0] == 4
    assert mask.shape == (4, 8)
    for row in range(4):
        active = mask[row].tolist()
        # Mask must be a prefix: once False it never returns to True.
        assert active == sorted(active, reverse=True)


def test_rollout_left_pads_to_a_shared_completion_offset():
    model = _tiny_model().eval()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(0)
    sequences, mask, lengths = rollout(
        model, [[1, 2, 3], [4, 5]], group_size=2, max_new_tokens=4,
        temperature=1.0, eos_id=3, pad_id=0, device=torch.device("cpu"),
        amp=False, microbatch=4, generator=generator)
    assert int(lengths.min()) == int(lengths.max()) == 3
    assert sequences.shape == (4, 3 + 4)


def test_settings_match_registered_protocol():
    """These values are pre-registered in docs/grpo-protocol.md. Changing them
    after seeing results is the failure mode the protocol exists to prevent."""
    settings = GRPOSettings()
    assert settings.group_size == 8
    assert settings.temperature == 1.0
    assert settings.kl_coefficient == 0.04
    assert settings.learning_rate == 1e-6
    assert settings.max_new_tokens == 64
    assert settings.planned_total_updates == 300
    assert settings.seed == 2028
