"""Segment accounting, FLOP estimation, and the sync-free metric path.

These cover the three claims [D023](../docs/decisions.md#d023) makes: that time
is attributed to exactly one segment, that the training rate excludes evaluation
and checkpoint and compile time, and that the training loop no longer converts
device tensors to Python floats inside the microbatch loop.
"""
from __future__ import annotations

import copy
import json
import math
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from modern_lm.config import ModernConfig
from modern_lm.model import ModernLM
from modern_lm.perf import (SegmentClock, estimate_flops_per_token,
                            parameter_breakdown, summarize)
from modern_lm.train import (TrainSettings, compute_loss, settings_drift, train)


def test_segments_are_disjoint_and_sum_to_wall_clock():
    clock = SegmentClock()
    clock.enter("step")
    time.sleep(0.02)
    with clock.section("evaluation"):
        time.sleep(0.02)
    time.sleep(0.02)
    clock.close()

    totals = clock.totals
    assert set(totals) == {"step", "evaluation"}
    # The section returned to `step`, so step holds two of the three sleeps.
    assert totals["step"] > totals["evaluation"]
    assert abs(sum(totals.values()) - 0.06) < 0.05


def test_snapshot_includes_the_open_bucket():
    clock = SegmentClock()
    clock.enter("step")
    time.sleep(0.01)
    assert clock.snapshot()["step"] > 0.0, "an open bucket must still be visible"
    assert "step" not in clock.totals, "snapshot must not close the open bucket"


def test_resumed_totals_carry_forward():
    clock = SegmentClock({"step": 100.0, "unattributed": 5.0})
    clock.enter("step")
    clock.close()
    assert clock.totals["step"] >= 100.0
    assert clock.totals["unattributed"] == 5.0


def test_training_rate_excludes_evaluation_and_checkpoint_time():
    clock = SegmentClock({"data": 1.0, "step": 9.0, "evaluation": 5.0,
                          "checkpoint": 3.0, "compile_and_warmup": 2.0})
    record = summarize(clock, tokens=1000)

    assert record["training_tokens_per_second"] == 100.0   # 1000 / (1 + 9)
    assert record["tokens_per_second"] == 50.0             # 1000 / 20
    assert record["seconds_evaluation"] == 5.0


def test_the_warmup_updates_tokens_are_excluded_with_its_time():
    """Counting them against a duration that excludes them invents throughput."""
    clock = SegmentClock({"data": 0.0, "step": 1.0, "compile_and_warmup": 9.0})
    record = summarize(clock, tokens=1000, training_tokens=100)

    assert record["training_tokens_per_second"] == 100.0
    assert record["tokens_per_second"] == 100.0


def test_no_measured_training_reports_none_not_zero():
    clock = SegmentClock({"compile_and_warmup": 9.0})
    record = summarize(clock, tokens=1000, training_tokens=0, flops_per_token=1e9)

    assert record["training_tokens_per_second"] is None
    assert "training_tflops" not in record
    assert record["seconds_step"] == 0.0, "a zero training bucket is still a fact"


def test_mfu_appears_only_with_a_declared_device_peak():
    clock = SegmentClock({"data": 0.0, "step": 1.0})
    without = summarize(clock, tokens=10, flops_per_token=1e9)
    assert "training_tflops" in without and "mfu" not in without

    with_peak = summarize(clock, tokens=10, flops_per_token=1e9,
                          device_peak_tflops=100.0)
    assert with_peak["mfu"] == (1e9 * 10 / 1e12) / 100.0


def test_parameter_breakdown_separates_head_from_embedding():
    config = ModernConfig.tiny()
    model = ModernLM(config)
    params = parameter_breakdown(model)

    assert params["stored"] == sum(p.numel() for p in model.parameters())
    assert params["body"] == params["stored"] - params["embedding"] - params["lm_head"]
    # The untied head is compute-bearing: it must be inside `non_embedding`,
    # which is the correction D016 asks for.
    assert params["non_embedding"] == params["body"] + params["lm_head"]
    assert params["lm_head"] > 0


def test_flops_per_token_counts_the_head_and_scales_with_context():
    config = ModernConfig.tiny()
    model = ModernLM(config)
    params = parameter_breakdown(model)

    short = estimate_flops_per_token(config, params["non_embedding"], 128)
    long = estimate_flops_per_token(config, params["non_embedding"], 512)
    assert long > short, "the attention term must grow with sequence length"
    assert short > 6 * params["body"], "the head's matmul must be charged"


def test_compute_loss_components_stay_on_device():
    """The regression that mattered: floats here meant a host sync per microbatch."""
    config = ModernConfig.tiny()
    model = ModernLM(config)
    tokens = torch.randint(0, config.vocab_size, (2, 17))
    _, components, _ = compute_loss(model, tokens, TrainSettings())

    assert components, "components must not be empty"
    for name, value in components.items():
        assert isinstance(value, torch.Tensor), f"{name} was converted to a Python scalar"
        assert not value.requires_grad, f"{name} keeps the autograd graph alive"


def test_batch_shape_preserves_loss_gradient_and_adamw_step():
    """One full batch and four token-weighted microbatches are equivalent."""
    torch.manual_seed(2026)
    config = ModernConfig.tiny()
    full = ModernLM(config)
    accumulated = copy.deepcopy(full)
    tokens = torch.randint(0, config.vocab_size, (8, 17))

    full_optimizer = torch.optim.AdamW(full.parameters(), lr=1e-3)
    accumulated_optimizer = torch.optim.AdamW(accumulated.parameters(), lr=1e-3)
    settings = TrainSettings()

    full_loss, _, _ = compute_loss(full, tokens, settings)
    full_loss.backward()

    accumulated_loss = torch.zeros(())
    for microbatch in tokens.chunk(4):
        loss, _, _ = compute_loss(accumulated, microbatch, settings)
        accumulated_loss += loss.detach() / 4
        (loss / 4).backward()

    assert torch.allclose(full_loss.detach(), accumulated_loss, atol=1e-6)
    for full_parameter, accumulated_parameter in zip(full.parameters(),
                                                       accumulated.parameters()):
        assert full_parameter.grad is not None and accumulated_parameter.grad is not None
        assert torch.allclose(full_parameter.grad, accumulated_parameter.grad,
                              atol=2e-6, rtol=1e-5)

    full_optimizer.step()
    accumulated_optimizer.step()
    for full_parameter, accumulated_parameter in zip(full.parameters(),
                                                       accumulated.parameters()):
        assert torch.allclose(full_parameter, accumulated_parameter,
                              atol=2e-6, rtol=1e-5)


def _corpus(path: Path, tokens: int = 40_000, seed: int = 7) -> Path:
    rng = np.random.default_rng(seed)
    rng.integers(0, 256, size=tokens, dtype=np.uint16).tofile(path)
    return path


def _tiny_run(tmp_path: Path, **overrides) -> list[dict]:
    config = ModernConfig.tiny()
    settings = TrainSettings(microbatch_size=2, gradient_accumulation=2,
                             sequence_length=config.max_seq_len,
                             warmup_updates=1, planned_total_tokens=4096,
                             checkpoint_tokens=10**9)
    target = overrides.pop("target_tokens", 4 * 2 * 2 * config.max_seq_len)
    run_dir = tmp_path / overrides.pop("run_dir_name", "run")
    train(config, settings, target_tokens=target, run_dir=run_dir,
          device=torch.device("cpu"), resume=overrides.pop("resume", None),
          eval_batches=1, log_every=1, compile_model=False,
          train_path=_corpus(tmp_path / "train.bin"),
          heldout_path=_corpus(tmp_path / "heldout.bin", seed=8),
          **overrides)
    return [json.loads(line) for line in (run_dir / "train.jsonl").read_text().splitlines()]


def test_training_log_reports_segments_and_a_training_only_rate(tmp_path):
    records = _tiny_run(tmp_path)
    updates = [r for r in records if r["event"] == "update"]
    assert updates, "no update records were written"

    last = updates[-1]
    assert last["seconds_data"] > 0.0
    assert last["seconds_step"] > 0.0
    # The rates have different numerators as well as different denominators: the
    # first update's tokens and compile/warmup time are both excluded from the
    # training-only rate. On a tiny CPU run that warmup update can be faster than
    # later updates, so training-only is not mathematically required to exceed
    # end-to-end. Verify the accounting identities instead of their incidental
    # ordering.
    tokens_per_update = records[0]["tokens_per_update"]
    measured_tokens = (last["optimizer_step"] - 1) * tokens_per_update
    training_seconds = last["seconds_data"] + last["seconds_step"]
    assert math.isclose(last["training_tokens_per_second"],
                        measured_tokens / training_seconds, rel_tol=1e-12)
    assert math.isclose(last["tokens_per_second"],
                        last["tokens_seen"] / last["elapsed_seconds"], rel_tol=1e-12)
    assert last["training_tflops"] > 0.0
    assert "mfu" not in last, "MFU must not appear without a declared device peak"
    # The first update's compile/warmup cost is held out of the training rate.
    assert last["seconds_compile_and_warmup"] > 0.0


def test_run_identity_record_opens_the_log(tmp_path):
    records = _tiny_run(tmp_path)
    assert records[0]["event"] == "run_identity"
    assert records[0]["parameters"]["non_embedding"] > 0
    assert records[0]["gradient_accumulation"] == 2
    assert records[0]["interventions"] == []


def test_step_profile_breaks_the_update_into_segments(tmp_path):
    records = _tiny_run(tmp_path, profile_every=2, device_peak_tflops=100.0)
    profiles = [r for r in records if r["event"] == "step_profile"]
    assert profiles, "--profile-every did not emit a step_profile record"

    profile = profiles[0]
    for segment in ("ms_data", "ms_forward", "ms_backward", "ms_optimizer"):
        assert profile[segment] > 0.0, f"{segment} was not timed"
    assert profile["step_mfu"] > 0.0
    # The timer's internal cursor must not leak into the record as a segment.
    assert "ms__since" not in profile
    parts = sum(profile[f"ms_{s}"] for s in ("data", "forward", "backward", "optimizer"))
    assert abs(parts - profile["ms_update"]) < 1e-6


def test_resume_records_a_changed_batch_shape_as_an_intervention(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.with_suffix(".json").write_text(json.dumps(
        {"settings": {"microbatch_size": 16, "gradient_accumulation": 4,
                      "learning_rate": 3e-4}}))

    changed = settings_drift(checkpoint, TrainSettings(microbatch_size=64,
                                                       gradient_accumulation=1))
    fields = {entry["field"]: entry for entry in changed}
    assert fields["microbatch_size"]["from"] == 16
    assert fields["microbatch_size"]["to"] == 64
    assert fields["gradient_accumulation"]["to"] == 1
    assert "learning_rate" not in fields, "an unchanged setting is not an intervention"


def test_resume_without_a_sidecar_says_so_rather_than_claiming_nothing_changed(tmp_path):
    changed = settings_drift(tmp_path / "absent.pt", TrainSettings())
    assert changed and changed[0]["field"] == "*"


def test_terminal_checkpoint_persists_its_own_segment_time(tmp_path):
    config = ModernConfig.tiny()
    settings = TrainSettings(microbatch_size=2, gradient_accumulation=1,
                             sequence_length=config.max_seq_len, warmup_updates=1,
                             planned_total_tokens=8192, checkpoint_tokens=10**9)
    tokens_per_update = settings.microbatch_size * settings.sequence_length
    run_dir = tmp_path / "one-checkpoint"
    train(config, settings, target_tokens=tokens_per_update, run_dir=run_dir,
          device=torch.device("cpu"), resume=None, eval_batches=1, log_every=1,
          compile_model=False, train_path=_corpus(tmp_path / "train.bin"),
          heldout_path=_corpus(tmp_path / "heldout.bin", seed=8))

    metadata = json.loads((run_dir / "latest.json").read_text())
    assert metadata["state"]["segment_seconds"]["checkpoint"] > 0.0, (
        "a terminal checkpoint must carry its own write time into the next resume")


def test_resumed_run_carries_segment_time_and_logs_the_intervention(tmp_path):
    config = ModernConfig.tiny()
    tokens_per_update = 2 * 2 * config.max_seq_len
    common = dict(eval_batches=1, log_every=1, compile_model=False,
                  train_path=_corpus(tmp_path / "train.bin"),
                  heldout_path=_corpus(tmp_path / "heldout.bin", seed=8),
                  device=torch.device("cpu"))
    run_dir = tmp_path / "resumed"
    first = TrainSettings(microbatch_size=2, gradient_accumulation=2,
                          sequence_length=config.max_seq_len, warmup_updates=1,
                          planned_total_tokens=8192, checkpoint_tokens=tokens_per_update)
    train(config, first, target_tokens=2 * tokens_per_update, run_dir=run_dir, resume=None,
          **common)
    first_metadata = json.loads((run_dir / "latest.json").read_text())
    first_checkpoint_seconds = first_metadata["state"]["segment_seconds"]["checkpoint"]
    assert first_checkpoint_seconds > 0.0

    # Same tokens per update, different execution shape -- the change this
    # measurement work exists to make visible.
    second = replace(first, microbatch_size=4, gradient_accumulation=1)
    train(config, second, target_tokens=4 * tokens_per_update, run_dir=run_dir,
          resume=run_dir / "latest.pt", **common)

    records = [json.loads(line) for line in (run_dir / "train.jsonl").read_text().splitlines()]
    identities = [r for r in records if r["event"] == "run_identity"]
    assert len(identities) == 2
    fields = {entry["field"]: entry for entry in identities[1]["interventions"]}
    assert fields["microbatch_size"]["from"] == 2 and fields["microbatch_size"]["to"] == 4
    assert fields["gradient_accumulation"]["to"] == 1

    # Everything after the second run_identity belongs to the resumed process.
    boundary = records.index(identities[1])
    last_before = [r for r in records[:boundary] if r["event"] == "update"][-1]
    first_after = [r for r in records[boundary:] if r["event"] == "update"][0]
    assert first_after["seconds_step"] > last_before["seconds_step"], (
        "the resumed run restarted its segment clock instead of carrying it forward")
    assert first_after["seconds_evaluation"] >= last_before["seconds_evaluation"]
    assert first_after["seconds_checkpoint"] >= first_checkpoint_seconds, (
        "the checkpoint wrote its timing only after serializing the resume state")
