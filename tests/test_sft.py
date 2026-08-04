"""Tests for the SFT stage.

The prompt-masking and token-weighting tests are the load-bearing ones: a bug in
either is invisible in the loss curve but silently trains the model on the wrong
thing, which is exactly the class of error that costs GPU hours.
"""
from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from modern_lm.config import ModernConfig
from modern_lm.data import default_paths
from modern_lm.model import ModernLM
from modern_lm.sft import (SFT_DATA_DIR, SFTSettings, collate_sft, encode_sft_record,
                           learning_rate_for_update, save_sft_checkpoint)

from tokenizers import Tokenizer  # noqa: E402

SFT_AVAILABLE = (SFT_DATA_DIR / "train.jsonl").exists()
TOKENIZER_AVAILABLE = default_paths()["tokenizer"].exists()
requires_sft = pytest.mark.skipif(
    not (SFT_AVAILABLE and TOKENIZER_AVAILABLE), reason="SFT corpus not present")

# Recorded in the reference's data/sft-math/metadata.json. Pinning these makes a
# silent regeneration of the corpus -- which would change the holdout split and
# break comparability with the reference's SFT baseline -- fail loudly.
EXPECTED = {
    "train.jsonl": ("8fcac3ce762d125c6dcbf7e2204436b6e0ad5098e05a62afae8d2d4be0594f6a", 16679),
    "heldout.jsonl": ("195147408d917221d15215347b4513c005f5a34d7722ba0105f5d23b98ab1cf3", 878),
}


@requires_sft
def test_sft_corpus_matches_recorded_fingerprints():
    for name, (digest, count) in EXPECTED.items():
        path = SFT_DATA_DIR / name
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        assert actual == digest, f"{name} changed; holdout split may differ"
        assert sum(1 for line in path.open() if line.strip()) == count


@requires_sft
def test_sft_train_and_heldout_are_disjoint():
    def keys(name):
        return {" ".join(json.loads(l)["question"].casefold().split())
                for l in (SFT_DATA_DIR / name).open() if l.strip()}
    assert not (keys("train.jsonl") & keys("heldout.jsonl"))


@pytest.fixture
def tokenizer():
    if not TOKENIZER_AVAILABLE:
        pytest.skip("tokenizer not present")
    return Tokenizer.from_file(str(default_paths()["tokenizer"]))


def _record(question="What is 2 + 2?", response="2 + 2 = 4\nFinal answer: 4"):
    from modern_lm.sft import SFTRecord
    return SFTRecord(source="test", identifier="t0", question=question,
                     response=response, answer="4")


@requires_sft
def test_prompt_is_masked_and_response_is_supervised(tokenizer):
    """Prompt tokens stay in context but must contribute no gradient."""
    record = _record()
    inputs, labels = encode_sft_record(record, tokenizer, 512)

    prompt = f"Question: {record.question}\nAnswer:"
    n_prompt = len(tokenizer.encode(prompt, add_special_tokens=False).ids)

    assert inputs.shape == labels.shape
    # Every prompt-interior label is masked...
    assert bool((labels[:n_prompt - 1] == -100).all())
    # ...and the first response token IS supervised (the exact boundary).
    assert int(labels[n_prompt - 1]) != -100
    assert bool((labels != -100).any())
    # The prompt itself remains visible to the model.
    assert torch.equal(inputs[:n_prompt],
                       torch.tensor(tokenizer.encode(prompt, add_special_tokens=False).ids))


@requires_sft
def test_eos_is_the_final_supervised_label(tokenizer):
    """SFT teaching <eos> is the mechanism expected to fix run-on generation."""
    inputs, labels = encode_sft_record(_record(), tokenizer, 512)
    eos_id = tokenizer.token_to_id("<eos>")
    assert int(labels[-1]) == eos_id
    supervised = labels[labels != -100]
    assert int(supervised[-1]) == eos_id


@requires_sft
def test_collate_pads_without_creating_supervised_tokens(tokenizer):
    short = encode_sft_record(_record(response="4"), tokenizer, 512)
    long = encode_sft_record(
        _record(response="Add them: 2 + 2 = 4.\nFinal answer: 4"), tokenizer, 512)
    pad_id = tokenizer.token_to_id("<pad>")
    inputs, labels = collate_sft([short, long], pad_id)

    width = max(short[0].numel(), long[0].numel())
    assert inputs.shape == (2, width) and labels.shape == (2, width)
    # Padding must never be supervised, or the model learns to emit <pad>.
    assert int((labels != -100).sum()) == int((short[1] != -100).sum()) + int((long[1] != -100).sum())
    assert bool((labels[0, short[0].numel():] == -100).all())
    assert bool((inputs[0, short[0].numel():] == pad_id).all())


def test_token_weighted_accumulation_equals_one_combined_batch():
    """The correctness property: splitting a batch must not change the gradient.

    A plain 1/accum average would over-weight the microbatch with fewer
    supervised tokens; scaling by each microbatch's token share is what makes
    accumulation exact for variable-length responses.
    """
    torch.manual_seed(0)
    config = ModernConfig.tiny()
    model = ModernLM(config)
    pad_id = 0

    # Two examples with deliberately different supervised-token counts.
    a_in = torch.randint(4, config.vocab_size, (7,))
    a_lb = a_in.clone(); a_lb[:5] = -100          # 2 supervised
    b_in = torch.randint(4, config.vocab_size, (11,))
    b_lb = b_in.clone(); b_lb[:3] = -100          # 8 supervised

    def grads(loss):
        model.zero_grad(set_to_none=True)
        loss.backward()
        return [p.grad.detach().clone() for p in model.parameters() if p.grad is not None]

    # Combined single batch.
    inputs, labels = collate_sft([(a_in, a_lb), (b_in, b_lb)], pad_id)
    logits = model(inputs).logits
    combined = grads(F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                     labels.reshape(-1), ignore_index=-100))

    # Two microbatches, token-weighted.
    total = int((a_lb != -100).sum()) + int((b_lb != -100).sum())
    model.zero_grad(set_to_none=True)
    for one in [(a_in, a_lb), (b_in, b_lb)]:
        mb_in, mb_lb = collate_sft([one], pad_id)
        count = int((mb_lb != -100).sum())
        lg = model(mb_in).logits
        loss = F.cross_entropy(lg.reshape(-1, lg.shape[-1]), mb_lb.reshape(-1),
                               ignore_index=-100)
        (loss * count / total).backward()
    split = [p.grad.detach().clone() for p in model.parameters() if p.grad is not None]

    assert len(combined) == len(split)
    for c, s in zip(combined, split):
        assert torch.allclose(c, s, atol=1e-5), "token weighting is not exact"


def test_lr_schedule_is_unchanged_by_a_shorter_target():
    """A 100-update gate then resume-to-1000 must trace the 1000-update path."""
    settings = SFTSettings(warmup_updates=100, learning_rate=5e-5,
                           planned_total_updates=1000)
    assert learning_rate_for_update(100, settings) == pytest.approx(5e-5)
    assert learning_rate_for_update(50, settings) == pytest.approx(2.5e-5)
    end = learning_rate_for_update(1000, settings)
    assert end == pytest.approx(5e-6, rel=1e-6)
    values = [learning_rate_for_update(u, settings) for u in range(100, 1001, 50)]
    assert all(b <= a + 1e-12 for a, b in zip(values, values[1:]))


def test_sft_checkpoint_round_trip_restores_rng_and_optimizer(tmp_path):
    config = ModernConfig.tiny()
    settings = SFTSettings(microbatch_size=2, gradient_accumulation=1)
    random.seed(5)
    torch.manual_seed(5)
    model = ModernLM(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    inputs = torch.randint(0, config.vocab_size, (2, 9))
    labels = inputs.clone(); labels[:, :4] = -100
    logits = model(inputs).logits
    F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1),
                    ignore_index=-100).backward()
    optimizer.step()

    state = {"optimizer_step": 1, "examples_seen": 2, "supervised_tokens_seen": 10,
             "epoch": 0, "batch_in_epoch": 1, "elapsed_seconds": 0.5}
    path = tmp_path / "sft.pt"
    save_sft_checkpoint(path, model, optimizer, config, settings, state,
                        {"loss": 1.0, "perplexity": 2.718})

    expected = random.random()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["stage"] == "sft"
    random.setstate(payload["rng"]["python"])
    assert random.random() == expected

    fresh = ModernLM(config)
    fresh.load_state_dict(payload["model"])
    for a, b in zip(model.parameters(), fresh.parameters()):
        assert torch.equal(a, b)
    assert payload["optimizer"]["state"], "optimizer moments were not saved"
    assert path.with_suffix(".json").exists()
