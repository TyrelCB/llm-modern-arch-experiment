"""The playground must not become a second, divergent decode path.

Its whole justification is that it inspects the *same* weights the harness
scores. So the property that matters most here is that greedy streaming agrees
token-for-token with `ModernLM.generate`; the rest guards the parts that face
the network -- checkpoint path containment and the sampling filter.
"""
from __future__ import annotations

import json

import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers

from modern_lm.config import ModernConfig
from modern_lm.model import ModernLM
from modern_lm.serve import (
    ModelRegistry, REPO_ROOT, _sort_key, _top_p_filter, discover_checkpoints,
    resolve_checkpoint,
)


def _tokenizer_file(tmp_path):
    """A byte-level tokenizer whose ids are stable and small."""
    vocabulary = {chr(index): index for index in range(256)}
    vocabulary["<eos>"] = 256
    tokenizer = Tokenizer(models.WordLevel(vocabulary, unk_token=chr(0)))
    tokenizer.pre_tokenizer = pre_tokenizers.Split("", "isolated")
    path = tmp_path / "tokenizer.json"
    tokenizer.save(str(path))
    return path


@pytest.fixture(autouse=True)
def _isolate_rng():
    """Restore global RNG on exit.

    These tests seed torch to build a deterministic throwaway model. Leaking
    that seed changes the draw sequence for whatever test runs next, which this
    suite is legitimately sensitive to -- `test_checkpoint_round_trip` asserts
    on RNG restoration and fails when a neighbour reseeds the global generator.
    """
    state = torch.get_rng_state()
    try:
        yield
    finally:
        torch.set_rng_state(state)


def _checkpoint(tmp_path, name="latest.pt", **overrides):
    torch.manual_seed(0)
    config = ModernConfig(**{"vocab_size": 257, "max_seq_len": 64, "dim": 64,
                             "n_layers": 2, "n_heads": 4, "n_kv_heads": 4,
                             "ffn_dim": 128, **overrides})
    model = ModernLM(config)
    path = tmp_path / name
    torch.save({"format_version": 1, "stage": "sft", "config": config.to_dict(),
                "state": {"optimizer_step": 42, "supervised_tokens_seen": 1234},
                "evaluation": {"loss": 0.5, "perplexity": 1.65},
                "model": model.state_dict()}, path)
    return path, model


def _registry(tmp_path):
    return ModelRegistry(_tokenizer_file(tmp_path))


def test_greedy_stream_matches_generate(tmp_path):
    """Temperature 0 must reproduce the reference decode exactly."""
    path, model = _checkpoint(tmp_path)
    registry = _registry(tmp_path)
    registry.load(path, "cpu")

    prompt = "abc"
    events = list(registry.stream(prompt, 12, 0.0, 1.0, None, stop_on_eos=False))
    streamed = events[-1]["text"]

    ids = registry.tokenizer.encode(prompt, add_special_tokens=False).ids
    reference = registry.current.model.generate(
        torch.tensor([ids]), max_new_tokens=12, eos_token_id=None, use_cache=True)
    expected = registry.tokenizer.decode(reference[0, len(ids):].tolist())

    assert streamed == expected
    assert events[-1]["generated_tokens"] == 12
    assert "".join(e["text"] for e in events if e["type"] == "token") == streamed


def test_stream_emits_incrementally_then_done(tmp_path):
    path, _ = _checkpoint(tmp_path)
    registry = _registry(tmp_path)
    registry.load(path, "cpu")
    events = list(registry.stream("ab", 5, 0.0, 1.0, None, stop_on_eos=False))
    assert [e["type"] for e in events[:-1]] == ["token"] * (len(events) - 1)
    done = events[-1]
    assert done["type"] == "done"
    assert done["finish_reason"] == "length"
    assert done["prompt_tokens"] == 2 and not done["prompt_truncated"]


def test_seeded_sampling_is_reproducible(tmp_path):
    path, _ = _checkpoint(tmp_path)
    registry = _registry(tmp_path)
    registry.load(path, "cpu")
    run = lambda seed: list(registry.stream("ab", 10, 1.0, 0.95, seed, False))[-1]["text"]
    assert run(7) == run(7)


def test_prompt_longer_than_context_is_truncated(tmp_path):
    path, _ = _checkpoint(tmp_path)
    registry = _registry(tmp_path)
    registry.load(path, "cpu")
    done = list(registry.stream("a" * 200, 8, 0.0, 1.0, None, False))[-1]
    assert done["prompt_truncated"]
    assert done["prompt_tokens"] == 64 - 8


def test_stop_on_eos_ends_generation(tmp_path):
    """A forced-eos model stops well short of the budget, and never emits <eos>."""
    path, _ = _checkpoint(tmp_path)
    registry = _registry(tmp_path)
    registry.load(path, "cpu")
    with torch.no_grad():  # drive <eos> to the top of the distribution
        registry.current.model.lm_head.weight[registry.eos_id] += 500.0
    events = list(registry.stream("ab", 16, 0.0, 1.0, None, stop_on_eos=True))
    done = events[-1]
    assert done["finish_reason"] == "eos"
    assert done["generated_tokens"] < 16
    # The stop token terminates the stream rather than being shown to the user.
    assert "<eos>" not in done["text"]


def test_stop_on_eos_disabled_runs_to_the_budget(tmp_path):
    path, _ = _checkpoint(tmp_path)
    registry = _registry(tmp_path)
    registry.load(path, "cpu")
    with torch.no_grad():
        registry.current.model.lm_head.weight[registry.eos_id] += 500.0
    done = list(registry.stream("ab", 6, 0.0, 1.0, None, stop_on_eos=False))[-1]
    assert done["finish_reason"] == "length"
    assert done["generated_tokens"] == 6


def test_unload_clears_model(tmp_path):
    path, _ = _checkpoint(tmp_path)
    registry = _registry(tmp_path)
    registry.load(path, "cpu")
    registry.unload()
    assert registry.current is None
    with pytest.raises(RuntimeError):
        list(registry.stream("ab", 4, 0.0, 1.0, None, True))


def test_describe_reports_run_metadata(tmp_path):
    path, _ = _checkpoint(tmp_path)
    info = _registry(tmp_path).load(path, "cpu")
    assert info["stage"] == "sft"
    assert info["optimizer_step"] == 42
    assert info["tokens_seen"] == 1234
    assert info["eval_loss"] == 0.5
    assert info["non_embedding_params"] < info["total_params"]


def test_resolve_checkpoint_refuses_paths_outside_the_repo():
    for candidate in ("../../etc/passwd", "/etc/passwd", "/etc/hosts"):
        with pytest.raises(ValueError):
            resolve_checkpoint(candidate)


def test_resolve_checkpoint_refuses_non_checkpoints(tmp_path):
    stray = REPO_ROOT / "pyproject.toml"
    if stray.exists():
        with pytest.raises(ValueError):
            resolve_checkpoint("pyproject.toml")


def test_checkpoints_sort_numerically_with_latest_last():
    names = ["checkpoint-000600014848.pt", "checkpoint-000400031744.pt",
             "latest.pt", "checkpoint-001750007808.pt"]
    ordered = sorted((__import__("pathlib").Path(n) for n in names), key=_sort_key)
    assert [p.name for p in ordered] == [
        "checkpoint-000400031744.pt", "checkpoint-000600014848.pt",
        "checkpoint-001750007808.pt", "latest.pt"]


def test_discover_checkpoints_groups_by_run(tmp_path):
    (tmp_path / "run-a").mkdir()
    (tmp_path / "run-b").mkdir()
    (tmp_path / "empty").mkdir()
    for directory in ("run-a", "run-b"):
        (tmp_path / directory / "checkpoint-000100.pt").write_bytes(b"x")
    runs = discover_checkpoints(tmp_path)
    assert {r["run"] for r in runs} == {"run-a", "run-b"}  # "empty" omitted
    assert all(len(r["checkpoints"]) == 1 for r in runs)


def test_top_p_filter_keeps_threshold_token_and_renormalizes():
    probabilities = torch.tensor([[0.5, 0.3, 0.15, 0.05]])
    filtered = _top_p_filter(probabilities.clone(), 0.9)
    assert filtered[0, 3] == 0.0            # tail dropped
    assert filtered[0, 2] > 0.0             # token crossing the threshold kept
    assert filtered.sum().item() == pytest.approx(1.0)


def test_top_p_filter_always_keeps_one_token():
    probabilities = torch.tensor([[0.5, 0.3, 0.15, 0.05]])
    filtered = _top_p_filter(probabilities.clone(), 0.0)
    assert filtered[0, 0] == pytest.approx(1.0)
    assert filtered.sum().item() == pytest.approx(1.0)
