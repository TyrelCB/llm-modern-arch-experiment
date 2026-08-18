"""Resumable pretraining for ModernLM on the shared FineMath corpus.

Hyperparameters mirror the DeepSeek-V4 reference run exactly (sequence length
512, 32,768 target tokens per optimizer update, AdamW wd 0.1, lr 3e-4, 2,000
warmup updates, cosine decay to 0.1x, grad clip 1.0, seed 2026) so the only
intended difference between the two runs is the architecture.

Token accounting matches the reference: the trainer counts *supervised
next-token labels*, not input-array lengths.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import dataclass, asdict, replace
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .config import ModernConfig
from .data import PackedTokenStream, default_paths
from .model import ModernLM
from .losses import DEFAULT_CHUNK, chunked_cross_entropy
from .low_precision import (
    SUPPORTED_PRECISIONS,
    canonical_model_state_dict,
    configure_low_precision,
    load_canonical_model_state_dict,
    low_precision_autocast,
)
from .muon import build_optimizer, split_adamw_params
from .perf import (SegmentClock, estimate_flops_per_token, parameter_breakdown,
                   summarize)

CHECKPOINT_FORMAT = 1


@dataclass
class TrainSettings:
    sequence_length: int = 512
    # 32,768 tokens per optimizer update, as every run since the first has used.
    # The SHAPE changed on 2026-08-18: 16x4 was inherited from the DeepSeek-V4
    # comparison as a pinned control and was never tuned for this hardware.
    # 64x1 produces the identical gradient -- token-weighted accumulation makes
    # the two exactly equivalent -- and measured 1.04-1.09x faster compiled,
    # because one pass over 32,768-row GEMMs beats four over 8,192-row ones
    # ([D024](../../docs/decisions.md#d024)). Above ~600M bodies mb 64 stops
    # fitting; those runs pass the shape explicitly.
    microbatch_size: int = 64
    gradient_accumulation: int = 1
    learning_rate: float = 3e-4
    warmup_updates: int = 2000
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    min_lr_ratio: float = 0.1
    # "cosine" (every run before 2026-08-15) or "wsd". WSD holds the peak LR
    # flat and spends only the last `wsd_decay_fraction` of updates decaying,
    # linearly, to `min_lr_ratio` -- set that to 0.0 for the decay-to-zero form
    # the published WSD results use.
    lr_schedule: str = "cosine"
    wsd_decay_fraction: float = 0.2
    planned_total_tokens: int = 250_000_000
    checkpoint_tokens: int = 10_000_000
    mtp_weight: float = 0.1
    aux_weight: float = 0.01
    # Compute the vocabulary loss in row slices, recomputing each slice's logits
    # in the backward pass instead of storing the whole [tokens, 16384] tensor.
    # Same mathematical loss with a different bf16 reduction path; trades one
    # extra projection matmul for lower peak allocation
    # ([D030](../../docs/decisions.md#d030)). Memory-only opt-in under D032.
    chunked_cross_entropy: bool = False
    cross_entropy_chunk: int = DEFAULT_CHUNK
    # Projection GEMM precision. BF16 remains the accepted default under D033;
    # FP8 and NVFP4 are opt-in Transformer Engine paths whose exact recipe is
    # recorded in the run identity and checkpoint settings.
    precision: str = "bf16"
    seed: int = 2026
    optimizer: str = "adamw"
    muon_learning_rate: float = 0.02
    # Muon decays as p *= 1 - lr*weight_decay, so decay is coupled to the
    # learning rate. Sharing `weight_decay` with AdamW while Muon's LR is ~17x
    # larger meant every Muon LR change also changed regularization strength by
    # the same factor -- the two are not separable in any run before this flag.
    # None keeps the old behaviour (share `weight_decay`) so existing runs stay
    # reproducible and resumable.
    muon_weight_decay: float | None = None

    def effective_muon_weight_decay(self) -> float:
        return (self.weight_decay if self.muon_weight_decay is None
                else self.muon_weight_decay)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def learning_rate_at(update: int, settings: TrainSettings, total_updates: int) -> float:
    """Linear warmup then decay, planned against the full token budget.

    Two shapes, selected by `settings.lr_schedule`:

    cosine  Decays from the peak the moment warmup ends, reaching
            `min_lr_ratio * learning_rate` exactly at `total_updates`.
    wsd     Holds the peak flat, then decays linearly to the same floor over
            the final `wsd_decay_fraction` of the run. The stable phase means a
            checkpoint mid-run is still at peak LR, so a separate short decay
            can be branched off any point -- one long run yields several budget
            points, where cosine needs a fresh run per budget.

    Both are planned against the FULL budget: a `planned_total_tokens` smaller
    than the real one drives the schedule to its floor early.
    """
    if update < settings.warmup_updates:
        return settings.learning_rate * (update + 1) / settings.warmup_updates
    progress = (update - settings.warmup_updates) / max(
        1, total_updates - settings.warmup_updates)
    progress = min(1.0, max(0.0, progress))
    floor = settings.learning_rate * settings.min_lr_ratio

    if settings.lr_schedule == "wsd":
        decay = min(1.0, max(0.0, settings.wsd_decay_fraction))
        if decay <= 0.0:
            return settings.learning_rate
        stable = 1.0 - decay
        if progress <= stable:
            return settings.learning_rate
        # Linear from peak at `stable` down to `floor` at 1.0.
        t = (progress - stable) / decay
        return settings.learning_rate + t * (floor - settings.learning_rate)

    return floor + 0.5 * (settings.learning_rate - floor) * (1 + math.cos(math.pi * progress))


def compute_loss(model: ModernLM, tokens: torch.Tensor, settings: TrainSettings,
                 *, is_first_microbatch: bool | None = None
                 ) -> tuple[torch.Tensor, dict[str, torch.Tensor], int]:
    """Next-token loss over `tokens` [B, S+1]; returns (loss, components, n_targets).

    `components` holds DETACHED DEVICE TENSORS, not Python floats. Converting
    them here cost a host synchronization per component per microbatch -- three
    per microbatch with MTP and MoE on, twelve per optimizer update at
    accumulation 4 -- each one draining the queue and stalling the CPU until the
    GPU caught up. That is instrumentation charging itself to the thing it
    measures, and it confounded the batch-shape comparison it was supposed to
    inform: the 16x4 arm paid four times as many stalls per update as 64x1.
    Callers accumulate these tensors and convert once, at a logging boundary
    ([D023](../../docs/decisions.md#d023)).
    """
    inputs, labels = tokens[:, :-1], tokens[:, 1:]
    # torch.compile's OptimizedModule proxies attribute access to the wrapped
    # module, so `.config` and `.lm_head` resolve for both compiled and eager
    # models.
    config = model.config
    chunked = settings.chunked_cross_entropy
    with low_precision_autocast(model, is_first_microbatch):
        output = model(inputs,
                       return_aux_loss=config.use_moe,
                       return_mtp_logits=config.use_mtp,
                       return_hidden=chunked)
    if chunked:
        # The head projection happens inside the loss, a slice at a time, so the
        # full logit tensor is never allocated.
        main = chunked_cross_entropy(output.hidden, model.lm_head.weight, labels,
                                     chunk=settings.cross_entropy_chunk)
    else:
        # bf16 logits straight into cross_entropy: it upcasts internally for the
        # softmax, avoiding a full-precision [B, S, V] materialization.
        main = F.cross_entropy(
            output.logits.reshape(-1, output.logits.shape[-1]), labels.reshape(-1))
    total = main
    components = {"main": main.detach()}

    if output.aux_loss is not None and settings.aux_weight:
        total = total + settings.aux_weight * output.aux_loss
        components["aux"] = output.aux_loss.detach()
    if output.mtp_logits is not None and output.mtp_logits.shape[1] > 0 and settings.mtp_weight:
        mtp_labels = labels[:, 1:]
        mtp = F.cross_entropy(
            output.mtp_logits.reshape(-1, output.mtp_logits.shape[-1]),
            mtp_labels.reshape(-1))
        total = total + settings.mtp_weight * mtp
        components["mtp"] = mtp.detach()

    components["total"] = total.detach()
    return total, components, labels.numel()


@torch.no_grad()
def evaluate(model: ModernLM, stream: PackedTokenStream, settings: TrainSettings,
             batches: int, device: torch.device, amp: bool) -> dict[str, float]:
    model.eval()
    # Weighted on device and read once at the end: `batches` host syncs inside
    # the loop would serialize evaluation the same way they serialized training
    # ([D023](../../docs/decisions.md#d023)).
    weighted = torch.zeros((), device=device, dtype=torch.float32)
    weights = 0
    position = 0
    for batch_index in range(batches):
        tokens = stream.batch(position, settings.microbatch_size, device)
        position += settings.microbatch_size
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            _, components, n_targets = compute_loss(
                model, tokens, settings,
                is_first_microbatch=(batch_index == 0))
        weighted += components["main"].float() * n_targets
        weights += n_targets
    model.train()
    main_loss = float(weighted) / weights
    return {"main_loss": main_loss, "perplexity": math.exp(min(20.0, main_loss))}


def save_checkpoint(path: Path, model, optimizer, config: ModernConfig,
                    settings: TrainSettings, state: dict) -> None:
    payload = {
        "format_version": CHECKPOINT_FORMAT,
        "config": config.to_dict(),
        "settings": asdict(settings),
        "state": state,
        "model": canonical_model_state_dict(model),
        "optimizer": optimizer.state_dict(),
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)
    meta = {k: payload[k] for k in ("format_version", "config", "settings", "state")}
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")


def milestone_tokens(total_tokens: int, percents: Sequence[int],
                     decay_start_tokens: int | None = None) -> list[int]:
    """Token counts that must be checkpointed and never pruned.

    The percentages are of the FULL run, so a milestone means the same thing
    across rungs: "the 30% checkpoint" is comparable between a 50M and a 600M
    run regardless of their token budgets or checkpoint intervals.

    `decay_start_tokens` is added as a milestone in its own right. Under WSD
    that is the fork point -- the last moment the model is still at peak LR, so
    it is the only checkpoint an extended or differently-decayed run can
    legitimately branch from. The 40M-interval grid missed it by 2.8M tokens on
    the first SiameseNorm run, which left 800M-or-nothing as the fork choice.
    """
    marks = {round(total_tokens * p / 100) for p in percents}
    if decay_start_tokens is not None:
        marks.add(int(decay_start_tokens))
    return sorted(m for m in marks if m > 0)


def classify_checkpoints(run_dir: Path, protected: Sequence[int],
                         tolerance: int) -> tuple[list[Path], list[Path]]:
    """Split checkpoints into (keep_forever, prunable_recent).

    A checkpoint is protected when its token count is within `tolerance` of a
    milestone -- checkpoints land on whichever step first crosses a threshold,
    never exactly on it, so an equality test would protect nothing.
    """
    keep, recent = [], []
    for path in sorted(run_dir.glob("checkpoint-*.pt")):
        try:
            tokens = int(path.stem.split("-")[-1])
        except ValueError:
            recent.append(path)
            continue
        if any(abs(tokens - mark) <= tolerance for mark in protected):
            keep.append(path)
        else:
            recent.append(path)
    return keep, recent


def prune_checkpoints(run_dir: Path, keep_last: int,
                      protected: Sequence[int] = (),
                      tolerance: int = 0) -> None:
    """Keep every milestone forever, plus the `keep_last` most recent others.

    Two retention policies with different jobs. Milestones are the scientific
    record: fixed fractions of the run, comparable across rungs, kept for good.
    The rolling window is crash recovery: disposable, and only the newest few
    are worth the disk.

    A 2B-token run at a 50M interval writes 40 milestones; at ~1.7GiB each that
    is ~70GiB, which does not fit comfortably alongside the corpus. The
    per-milestone JSON metadata is never pruned, so the loss trajectory stays
    fully recoverable after the weights are gone. `latest.pt` is written
    separately and is never a pruning candidate.
    """
    if keep_last <= 0 and not protected:
        return
    _, recent = classify_checkpoints(run_dir, protected, tolerance)
    if keep_last <= 0:
        stale = recent
    else:
        stale = recent[:-keep_last] if len(recent) > keep_last else []
    for path in stale:
        path.unlink(missing_ok=True)


def load_checkpoint(path: Path, model, optimizer) -> dict:
    # map_location="cpu" is required: RNG state ByteTensors must stay on CPU for
    # torch.set_rng_state / set_rng_state_all to accept them. load_state_dict
    # moves the model/optimizer tensors onto the device afterwards.
    payload = torch.load(path, map_location="cpu", weights_only=False)
    load_canonical_model_state_dict(model, payload["model"])
    if optimizer is not None:
        # load_state_dict overwrites every non-tensor group key from the
        # checkpoint, including `lr_scale` -- so a resume silently restored the
        # ORIGINAL run's Muon LR and discarded whatever --muon-learning-rate the
        # CLI asked for. That made continued pretraining impossible to retune: a
        # 5x LR reduction changed weight drift by 0.06pp because the flag never
        # reached the optimizer. Restore the state, then put the freshly built
        # hyperparameters back.
        hyperparameters = [
            {k: v for k, v in group.items() if k != "params"}
            for group in optimizer.param_groups
        ]
        optimizer.load_state_dict(payload["optimizer"])
        for group, saved in zip(optimizer.param_groups, hyperparameters):
            group.update(saved)
    rng = payload.get("rng")
    if rng:
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch"])
        if rng.get("cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng["cuda"])
    state = payload["state"]
    # A checkpoint cannot serialize the duration of its own write before that
    # write finishes. `train` refreshes these two fields in the tiny JSON
    # sidecar afterwards; overlay only timing here so model, optimizer, RNG, and
    # trajectory counters continue to come from the atomic tensor checkpoint.
    sidecar = path.with_suffix(".json")
    if sidecar.exists():
        try:
            recorded = json.loads(sidecar.read_text()).get("state", {})
        except (OSError, json.JSONDecodeError):
            recorded = {}
        for field in ("elapsed_seconds", "segment_seconds"):
            if field in recorded:
                state[field] = recorded[field]
    return state


def refresh_checkpoint_timing(path: Path, state: dict) -> None:
    """Persist post-write timing without serializing model weights again.

    `save_checkpoint` necessarily captures state from just before its own I/O.
    Once both weight files and pruning finish, this atomically refreshes the
    timing fields in their small sidecars. `load_checkpoint` overlays exactly
    those fields on resume; scientific trajectory state still comes from the
    tensor checkpoint.
    """
    sidecar = path.with_suffix(".json")
    if not sidecar.exists():
        return
    try:
        metadata = json.loads(sidecar.read_text())
    except (OSError, json.JSONDecodeError):
        return
    recorded = metadata.setdefault("state", {})
    for field in ("elapsed_seconds", "segment_seconds"):
        if field in state:
            recorded[field] = state[field]
    temporary = sidecar.with_name(sidecar.name + ".tmp")
    temporary.write_text(json.dumps(metadata, indent=2) + "\n")
    os.replace(temporary, sidecar)


def settings_drift(resume: Path, settings: TrainSettings) -> list[dict]:
    """Settings this resume changes relative to the checkpoint it continues.

    A trajectory that changes hyperparameters mid-run is a different experiment
    from either endpoint, and the change has to be recoverable from the run's own
    record rather than from someone's memory of which flags they typed. The 300M
    champion's provisional status is partly this: its batch shape changed during
    training and nothing in the run says where ([D014](../../docs/decisions.md#d014),
    [D020](../../docs/decisions.md#d020)).

    Reads the checkpoint's JSON sidecar, not the checkpoint: the sidecar carries
    the same `settings` block and costs no tensor load. A checkpoint written
    before sidecars, or one whose sidecar is missing, reports the fact instead of
    claiming nothing changed.
    """
    sidecar = resume.with_suffix(".json")
    if not sidecar.exists():
        return [{"field": "*", "note": "no checkpoint sidecar; prior settings unknown"}]
    try:
        previous = json.loads(sidecar.read_text()).get("settings", {})
    except (OSError, json.JSONDecodeError) as error:
        return [{"field": "*", "note": f"unreadable checkpoint sidecar: {error}"}]

    # Checkpoints predating the low-precision integration ran the canonical
    # BF16-autocast path. Treat that implicit value as explicit so merely
    # resuming an old run does not manufacture a precision intervention.
    previous.setdefault("precision", "bf16")
    current = asdict(settings)
    changed = []
    for field in sorted(set(previous) | set(current)):
        before, after = previous.get(field), current.get(field)
        if before != after:
            changed.append({"field": field, "from": before, "to": after})
    return changed


def train(config: ModernConfig, settings: TrainSettings, *, target_tokens: int,
          run_dir: Path, device: torch.device, resume: Path | None,
          eval_batches: int, log_every: int, compile_model: bool,
          train_path: Path, heldout_path: Path,
          keep_last_checkpoints: int = 0,
          milestone_percents: Sequence[int] = (),
          checkpoint_minutes: float = 0.0,
          profile_every: int = 0,
          device_peak_tflops: float | None = None) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(settings.seed)
    amp = device.type == "cuda" and torch.cuda.is_bf16_supported()

    clock = SegmentClock()
    clock.enter("setup")

    model = ModernLM(config).to(device)
    precision = configure_low_precision(model, settings.precision, device)
    n_params = model.num_params()
    params = parameter_breakdown(model)
    flops_per_token = estimate_flops_per_token(config, params["non_embedding"],
                                               settings.sequence_length)
    if compile_model:
        model = torch.compile(model)

    if settings.optimizer == "muon":
        optimizer = build_optimizer(
            model, learning_rate=settings.learning_rate,
            muon_learning_rate=settings.muon_learning_rate,
            weight_decay=settings.weight_decay,
            muon_weight_decay=settings.effective_muon_weight_decay())
    else:
        decay, no_decay = split_adamw_params(model)
        optimizer = torch.optim.AdamW(
            [{"params": decay, "weight_decay": settings.weight_decay},
             {"params": no_decay, "weight_decay": 0.0}],
            lr=settings.learning_rate, betas=(0.9, 0.95), eps=1e-8)

    train_stream = PackedTokenStream(train_path, settings.sequence_length, settings.seed)
    heldout_stream = PackedTokenStream(heldout_path, settings.sequence_length, settings.seed + 1)

    tokens_per_update = (settings.microbatch_size * settings.gradient_accumulation
                         * settings.sequence_length)
    total_updates = max(1, settings.planned_total_tokens // tokens_per_update)

    # The WSD fork point in TOKENS. Decay is keyed to progress through the
    # post-warmup updates, not to raw tokens, so this has to be derived from the
    # same arithmetic lr_for_update uses -- computing it as a plain fraction of
    # the token budget lands in the wrong place by the length of warmup.
    decay_start_tokens = None
    if settings.lr_schedule == "wsd" and 0.0 < settings.wsd_decay_fraction < 1.0:
        stable_updates = settings.warmup_updates + (
            (total_updates - settings.warmup_updates) * (1.0 - settings.wsd_decay_fraction))
        decay_start_tokens = int(stable_updates * tokens_per_update)

    milestones = milestone_tokens(settings.planned_total_tokens, milestone_percents,
                                  decay_start_tokens)
    # A checkpoint lands on whichever step first crosses a threshold, so protect
    # anything within one update's worth of tokens of a milestone.
    #
    # One update is only enough when checkpoints are far denser than milestones.
    # At a coarse --checkpoint-tokens the nearest checkpoint to a milestone can
    # be most of an interval away -- a 30M interval against 10% marks of a
    # 5.93B run misses every one by up to 15M tokens -- and the milestone is
    # then protected by nothing and rotates out with the window. Widen the
    # tolerance to half a checkpoint interval so the closest checkpoint on
    # either side always qualifies, which is the checkpoint that actually
    # records that milestone.
    milestone_tolerance = max(tokens_per_update, settings.checkpoint_tokens // 2)
    if milestones:
        print(json.dumps({"event": "milestones", "tokens": milestones,
                          "decay_start_tokens": decay_start_tokens}), flush=True)

    state = {"micro_step": 0, "optimizer_step": 0, "tokens_seen": 0,
             "elapsed_seconds": 0.0, "next_checkpoint_tokens": settings.checkpoint_tokens,
             "next_milestone_index": 0, "last_checkpoint_seconds": 0.0}
    interventions = []
    if resume is not None and resume.exists():
        interventions = settings_drift(resume, settings)
        state = load_checkpoint(resume, model, optimizer)
        # Checkpoints written before these keys existed resume without them.
        state.setdefault("last_checkpoint_seconds", state.get("elapsed_seconds", 0.0))
        state["next_milestone_index"] = sum(
            1 for m in milestones if m <= state.get("tokens_seen", 0))
        print(json.dumps({"event": "resumed", "state": state}), flush=True)
    else:
        # `torch.compile` is lazy. Without a warmup forward, the initial held-out
        # evaluation pays for compilation and the segment ledger calls that cost
        # "evaluation". Compile one representative eval graph first so the
        # recorded evaluation bucket means evaluation rather than compiler work.
        if compile_model:
            with clock.section("compile_and_warmup"):
                evaluate(model, heldout_stream, settings, 1, device, amp)
        with clock.section("evaluation"):
            initial = evaluate(model, heldout_stream, settings, eval_batches, device, amp)
        state["initial_evaluation"] = initial
        print(json.dumps({"event": "initial_evaluation", **initial}), flush=True)

    log_path = run_dir / "train.jsonl"
    log_handle = log_path.open("a")

    # Carry a resumed run's prior time forward WITHOUT crediting it to any
    # segment: earlier segments were not recorded, and inventing an attribution
    # for them would corrupt the very rates this exists to make honest. Runs
    # started under this code carry real buckets across resumes instead.
    carried = state.get("segment_seconds")
    if carried is None:
        carried = {"unattributed": state.get("elapsed_seconds", 0.0)} if state.get(
            "elapsed_seconds") else {}
    # Buckets ADD across the resume boundary. Overwriting instead would drop the
    # earlier run's setup and compile time on every restart, so a repeatedly
    # resumed run would look like it never paid either.
    totals = dict(carried)
    for bucket, seconds in clock.snapshot().items():
        totals[bucket] = totals.get(bucket, 0.0) + seconds
    clock = SegmentClock(totals)
    clock.enter("setup")

    manifest = {"event": "run_identity", "parameters": params,
                "flops_per_token": flops_per_token,
                "tokens_per_update": tokens_per_update,
                "microbatch_size": settings.microbatch_size,
                "gradient_accumulation": settings.gradient_accumulation,
                "compiled": compile_model, "device": device.type,
                "precision": precision.to_dict(),
                "device_peak_tflops": device_peak_tflops,
                "resumed_from": str(resume) if resume is not None else None,
                "interventions": interventions}
    log_handle.write(json.dumps(manifest) + "\n")
    log_handle.flush()
    print(json.dumps(manifest), flush=True)

    model.train()
    # The first update pays for compilation, autotuning, and allocator warmup.
    # Charging it to `step` would depress a short run's throughput by minutes of
    # one-time cost, which is exactly the conflation D020 objects to.
    first_update = True
    marks: dict[str, float] = {}

    def mark(segment: str) -> None:
        """Close a profiled sub-segment, synchronizing so it means something.

        Called only on profiled updates. Without the sync these would time
        kernel *submission*, and the backward pass -- 61% of the step by the FP8
        measurement -- would appear to cost almost nothing.
        """
        if device.type == "cuda":
            torch.cuda.synchronize()
        now = time.perf_counter()
        marks[segment] = marks.get(segment, 0.0) + (now - marks.pop("_since", now))
        marks["_since"] = time.perf_counter()

    while state["tokens_seen"] < target_tokens:
        lr = learning_rate_at(state["optimizer_step"], settings, total_updates)
        # `lr` follows the AdamW base; a group carrying `lr_scale` (Muon) keeps
        # the same warmup/cosine shape rescaled to its own base magnitude.
        for group in optimizer.param_groups:
            scale = group.get("lr_scale")
            group["lr"] = lr if scale is None else lr * (scale / settings.learning_rate)

        profiling = (profile_every > 0
                     and (state["optimizer_step"] + 1) % profile_every == 0)
        if profiling:
            marks.clear()
            if device.type == "cuda":
                torch.cuda.synchronize()
            marks["_since"] = time.perf_counter()

        # Everything from here to the next `clock.enter` is training: the
        # per-interval logging sync included, because that sync is the CPU
        # waiting for training work to finish, not overhead of its own. On the
        # first update every bucket, data included, is compile/warmup -- its
        # tokens are excluded from the training rate, so its time has to be too
        # or the two sides of that ratio disagree.
        step_bucket = "compile_and_warmup" if first_update else "step"
        data_bucket = "compile_and_warmup" if first_update else "data"
        clock.enter(step_bucket)
        optimizer.zero_grad(set_to_none=True)
        update_tokens = 0
        update_components: dict[str, torch.Tensor] = {}
        remaining = target_tokens - state["tokens_seen"]

        # Token-weighted accumulation so a partial final update stays exact.
        planned = []
        for _ in range(settings.gradient_accumulation):
            if remaining <= 0:
                break
            batch_targets = settings.microbatch_size * settings.sequence_length
            planned.append(min(batch_targets, remaining))
            remaining -= planned[-1]
        if not planned:
            break
        planned_total = sum(planned)

        for microbatch_index, micro_targets in enumerate(planned):
            with clock.section(data_bucket):
                tokens = train_stream.batch(state["micro_step"], settings.microbatch_size,
                                            device)
            state["micro_step"] += settings.microbatch_size
            if profiling:
                mark("data")
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                loss, components, n_targets = compute_loss(
                    model, tokens, settings,
                    is_first_microbatch=(microbatch_index == 0))
            if profiling:
                mark("forward")
            scale = micro_targets / planned_total
            (loss * scale).backward()
            if profiling:
                mark("backward")
            update_tokens += micro_targets
            # Tensor accumulation: no host sync until a logging boundary reads
            # these. `scale` is a Python float, so this stays a device-side
            # scalar multiply-add ([D023](../../docs/decisions.md#d023)).
            for key, value in components.items():
                weighted = value * scale
                previous = update_components.get(key)
                update_components[key] = weighted if previous is None else previous + weighted

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), settings.grad_clip)
        optimizer.step()
        if profiling:
            mark("optimizer")
        state["optimizer_step"] += 1
        state["tokens_seen"] += update_tokens
        # Tokens whose cost landed in the training buckets. The warmup update's
        # tokens are excluded because its time was, and dividing one by the
        # other is how a five-figure trainer reported 10.9M tok/s.
        if not first_update:
            state["training_tokens"] = state.get("training_tokens", 0) + update_tokens
        first_update = False

        if profiling:
            marks.pop("_since", None)
            profile = {"event": "step_profile", "optimizer_step": state["optimizer_step"],
                       "tokens_seen": state["tokens_seen"],
                       "update_tokens": update_tokens,
                       **{f"ms_{segment}": value * 1000 for segment, value in marks.items()}}
            measured = sum(marks.values())
            profile["ms_update"] = measured * 1000
            profile["step_tokens_per_second"] = update_tokens / max(1e-9, measured)
            profile["step_tflops"] = flops_per_token * update_tokens / max(1e-9, measured) / 1e12
            if device_peak_tflops:
                profile["step_mfu"] = profile["step_tflops"] / device_peak_tflops
            log_handle.write(json.dumps(profile) + "\n")
            log_handle.flush()
            print(json.dumps(profile), flush=True)

        if state["optimizer_step"] % log_every == 0:
            # The one deliberate host sync per logging interval. It reads the
            # accumulated loss tensors and, as a side effect, re-anchors the
            # segment clock to real GPU progress rather than queue depth.
            losses = {f"loss_{key}": float(value) for key, value in update_components.items()}
            timing = summarize(clock, state["tokens_seen"],
                               training_tokens=state.get("training_tokens", 0),
                               flops_per_token=flops_per_token,
                               device_peak_tflops=device_peak_tflops)
            state["elapsed_seconds"] = timing["elapsed_seconds"]
            state["segment_seconds"] = clock.snapshot()
            record = {"event": "update", "optimizer_step": state["optimizer_step"],
                      "tokens_seen": state["tokens_seen"], "lr": lr,
                      "grad_norm": float(grad_norm), **timing, **losses}
            log_handle.write(json.dumps(record) + "\n")
            log_handle.flush()
            print(json.dumps(record), flush=True)
        else:
            state["elapsed_seconds"] = clock.total()

        crossed = state["tokens_seen"] >= state["next_checkpoint_tokens"]
        # A milestone must be captured even when it falls between grid points --
        # that is the whole reason the WSD fork point was missed before.
        index = state["next_milestone_index"]
        milestone_due = index < len(milestones) and state["tokens_seen"] >= milestones[index]
        # Wall-clock checkpoints are crash insurance: they bound how much
        # compute a failure can destroy, independent of throughput.
        due_seconds = checkpoint_minutes * 60.0
        time_due = due_seconds > 0 and (
            state["elapsed_seconds"] - state["last_checkpoint_seconds"]) >= due_seconds
        if crossed or milestone_due or time_due or state["tokens_seen"] >= target_tokens:
            # Evaluation has to wait for the training queue on the same device
            # anyway. Anchor that wait in the open training bucket before
            # switching segments, or the tail of the last optimizer step is
            # silently charged to evaluation when this boundary falls between
            # logging updates.
            if device.type == "cuda":
                torch.cuda.synchronize()
            clock.enter("evaluation")
            metrics = evaluate(model, heldout_stream, settings, eval_batches, device, amp)
            state["elapsed_seconds"] = clock.total()
            state["segment_seconds"] = clock.snapshot()
            state["heldout"] = metrics
            record = {"event": "evaluation", "tokens_seen": state["tokens_seen"],
                      "elapsed_seconds": state["elapsed_seconds"],
                      "seconds_evaluation": state["segment_seconds"].get("evaluation", 0.0),
                      "parameters": n_params, **metrics}
            log_handle.write(json.dumps(record) + "\n")
            log_handle.flush()
            print(json.dumps(record), flush=True)
            while state["next_checkpoint_tokens"] <= state["tokens_seen"]:
                state["next_checkpoint_tokens"] += settings.checkpoint_tokens
            while (state["next_milestone_index"] < len(milestones)
                   and milestones[state["next_milestone_index"]] <= state["tokens_seen"]):
                state["next_milestone_index"] += 1
            state["last_checkpoint_seconds"] = state["elapsed_seconds"]
            clock.enter("checkpoint")
            # The tensor payload is necessarily written from the pre-I/O state.
            # After both large files finish, refresh_checkpoint_timing patches
            # only the small sidecars with the completed write duration.
            checkpoint_path = run_dir / f"checkpoint-{state['tokens_seen']:012d}.pt"
            latest_path = run_dir / "latest.pt"
            save_checkpoint(checkpoint_path, model, optimizer, config, settings, state)
            save_checkpoint(latest_path, model, optimizer, config, settings, state)
            prune_checkpoints(run_dir, keep_last_checkpoints,
                              protected=milestones, tolerance=milestone_tolerance)
            state["elapsed_seconds"] = clock.total()
            state["segment_seconds"] = clock.snapshot()
            refresh_checkpoint_timing(checkpoint_path, state)
            refresh_checkpoint_timing(latest_path, state)

    clock.close()
    log_handle.close()
    final = summarize(clock, state["tokens_seen"],
                      training_tokens=state.get("training_tokens", 0),
                      flops_per_token=flops_per_token,
                      device_peak_tflops=device_peak_tflops)
    print(json.dumps({"event": "complete", "tokens_seen": state["tokens_seen"],
                      **final, "heldout": state.get("heldout")}), flush=True)


def main() -> None:
    paths = default_paths()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-tokens", type=int, required=True)
    parser.add_argument("--run-dir", type=Path, default=Path("runs/modern-145m"))
    parser.add_argument("--train", type=Path, default=paths["train"])
    parser.add_argument("--heldout", type=Path, default=paths["heldout"])
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--microbatch-size", type=int, default=64,
                        help="rows per forward/backward pass. The default pairs "
                             "with accumulation 1 for 32,768 tokens per update, "
                             "the same budget every run has used, in the shape "
                             "that measured fastest (D024). Peak memory at 64x1 "
                             "is 40.2GB at 300M and 86.9GB at 1B of a shared "
                             "121GB pool -- drop to 32x2 or 16x4 above 600M")
    parser.add_argument("--gradient-accumulation", type=int, default=1,
                        help="micro-batches per optimizer update. On one GPU "
                             "this buys nothing but overhead; it exists for "
                             "fitting a token budget that a single pass cannot")
    parser.add_argument("--eval-batches", type=int, default=32)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--warmup-updates", type=int, default=2000)
    parser.add_argument("--lr-schedule", choices=("cosine", "wsd"), default="cosine",
                        help="cosine decays from the end of warmup; wsd holds "
                             "the peak then decays over the final fraction")
    parser.add_argument("--wsd-decay-fraction", type=float, default=0.2,
                        help="wsd only: fraction of updates spent decaying")
    parser.add_argument("--min-lr-ratio", type=float, default=0.1,
                        help="floor as a fraction of --learning-rate; 0.0 "
                             "decays to zero, which is the usual WSD form")
    parser.add_argument("--planned-total-tokens", type=int, default=250_000_000)
    parser.add_argument("--checkpoint-tokens", type=int, default=10_000_000,
                        help="evaluate and checkpoint on each crossed multiple")
    parser.add_argument("--milestone-percents", type=str, default="",
                        help="comma-separated percentages of the run to checkpoint "
                             "and never prune, e.g. '10,20,30,40,50,60,70,80,90,100'. "
                             "The WSD decay start is always added when applicable, "
                             "since it is the only legitimate fork point for an "
                             "extended or differently-decayed run.")
    parser.add_argument("--checkpoint-minutes", type=float, default=0.0,
                        help="also checkpoint every N minutes of wall clock, for "
                             "crash recovery. These rotate under --keep-last-checkpoints; "
                             "milestones never do.")
    parser.add_argument("--keep-last-checkpoints", type=int, default=0,
                        help="if >0, retain only the N most recent milestone "
                             "checkpoints (latest.pt is always kept)")
    parser.add_argument("--optimizer", choices=("adamw", "muon"), default="adamw")
    parser.add_argument("--muon-learning-rate", type=float, default=0.02,
                        help="peak LR for the Muon group; --learning-rate still "
                             "drives the AdamW group and the schedule shape")
    parser.add_argument("--muon-weight-decay", type=float, default=None,
                        help="weight decay for the Muon group only; defaults to "
                             "--weight-decay. Muon shrinks by lr*weight_decay, so "
                             "sharing one value across a ~17x LR gap couples decay "
                             "to LR -- set this to vary them independently")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--precision", choices=SUPPORTED_PRECISIONS, default="bf16",
                        help="projection GEMM precision. fp8/nvfp4 use Transformer "
                             "Engine while embedding, norms, router, and LM head "
                             "stay on the BF16-autocast path")
    parser.add_argument("--profile-every", type=int, default=0,
                        help="every N updates, time data/forward/backward/optimizer "
                             "with explicit synchronization and log a step_profile "
                             "record. That update is slower than a normal one, which "
                             "is why this is sampled rather than always on; the "
                             "always-on segment totals stay synchronization-free")
    parser.add_argument("--device-peak-tflops", type=float, default=None,
                        help="device peak dense throughput in TFLOP/s at the training "
                             "dtype, used to turn measured FLOP/s into MFU. No default: "
                             "a guessed constant would silently rescale every MFU "
                             "number derived from it")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    # Model size. Defaults reproduce dense_145m, so existing commands are
    # unchanged; pass these to train a different width/depth on the same code.
    parser.add_argument("--dim", type=int, default=None)
    parser.add_argument("--n-layers", type=int, default=None)
    parser.add_argument("--n-heads", type=int, default=None)
    parser.add_argument("--n-kv-heads", type=int, default=None)
    parser.add_argument("--ffn-dim", type=int, default=None)
    parser.add_argument("--siamese-norm", action="store_true",
                        help="two-stream SiameseNorm residual (arXiv 2602.08064) "
                             "instead of single-stream Pre-LN")
    parser.add_argument("--chunked-cross-entropy", action="store_true",
                        help="compute the vocabulary loss in row slices, recomputing "
                             "each slice's logits in the backward pass. Same exact-"
                             "arithmetic loss but a different bf16 reduction; saves "
                             "peak memory at a measured throughput cost (D030/D032)")
    parser.add_argument("--cross-entropy-chunk", type=int, default=DEFAULT_CHUNK,
                        help="rows per slice for --chunked-cross-entropy; does not "
                             "change the loss, only peak memory and launch count")
    parser.add_argument("--fuse-projections", action="store_true",
                        help="fuse Q/K/V and SwiGLU gate/up into single matmuls. "
                             "Same exact-arithmetic function; changes floating-point "
                             "reduction order and showed no compiled GB10 speedup "
                             "at 50M/300M (D028/D031). It "
                             "changes the state dict: convert an existing checkpoint "
                             "with scripts/convert_projection_fusion.py before "
                             "resuming into it")
    args = parser.parse_args()

    settings = TrainSettings(
        microbatch_size=args.microbatch_size,
        gradient_accumulation=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        warmup_updates=args.warmup_updates,
        lr_schedule=args.lr_schedule,
        wsd_decay_fraction=args.wsd_decay_fraction,
        min_lr_ratio=args.min_lr_ratio,
        planned_total_tokens=args.planned_total_tokens,
        checkpoint_tokens=args.checkpoint_tokens,
        optimizer=args.optimizer,
        muon_learning_rate=args.muon_learning_rate,
        muon_weight_decay=args.muon_weight_decay,
        chunked_cross_entropy=args.chunked_cross_entropy,
        cross_entropy_chunk=args.cross_entropy_chunk,
        precision=args.precision,
        seed=args.seed)
    config = ModernConfig.dense_145m()
    overrides = {name: getattr(args, name)
                 for name in ("dim", "n_layers", "n_heads", "n_kv_heads", "ffn_dim")
                 if getattr(args, name) is not None}
    if args.siamese_norm:
        overrides["use_siamese_norm"] = True
    if args.fuse_projections:
        overrides["fuse_projections"] = True
    if overrides:
        config = replace(config, **overrides)
        print(json.dumps({"event": "model_size", "parameters": None, **overrides}),
              flush=True)

    train(config, settings,
          target_tokens=args.target_tokens, run_dir=args.run_dir,
          device=torch.device(args.device), resume=args.resume,
          eval_batches=args.eval_batches, log_every=args.log_every,
          compile_model=not args.no_compile,
          train_path=args.train, heldout_path=args.heldout,
          keep_last_checkpoints=args.keep_last_checkpoints,
          milestone_percents=[int(p) for p in args.milestone_percents.split(",") if p.strip()],
          checkpoint_minutes=args.checkpoint_minutes,
          profile_every=args.profile_every,
          device_peak_tflops=args.device_peak_tflops)


if __name__ == "__main__":
    main()
