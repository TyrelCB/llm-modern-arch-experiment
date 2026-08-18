"""Honest performance accounting: where wall clock goes, and what it bought.

Two things were wrong with how throughput was reported before this module.

The logged `tokens_per_second` was cumulative -- `tokens_seen / elapsed` since
the run started -- so it charged training with compile time, every held-out
evaluation, and every checkpoint write. A run that evaluates often reported a
lower "throughput" than an identical run that evaluates rarely, and neither
number was the rate the GPU trains at. `SegmentClock` separates those buckets so
`training_tokens_per_second` means only the tokens-per-second of the training
segments ([D023](../../docs/decisions.md#d023)).

And nothing here knew what fraction of the machine a run was using. The repo has
throughput numbers going back to the first commit and not one utilization
figure, so "35,975 tok/s" has never been checkable against what the hardware can
do. `estimate_flops_per_token` supplies the numerator; the denominator has to be
passed in, because a peak-FLOPs constant guessed from a spec sheet is a fiction
that would quietly rescale every MFU number computed from it.
"""
from __future__ import annotations

import time
from contextlib import contextmanager

from .config import ModernConfig


class SegmentClock:
    """Wall-clock time bucketed by what the process was doing.

    Buckets are disjoint by construction: `enter` closes whichever bucket was
    open before opening the next, so the totals sum to the time between the
    first `enter` and the last `close` and cannot double-count. That matters
    more than sub-millisecond precision -- the failure this replaces was a
    number that silently included evaluation, not one that was off by a
    microsecond.

    No CUDA synchronization happens here. On a device queue these are CPU-side
    boundaries, so a bucket measures when work was *submitted* unless something
    else forces a sync. In practice the training loop syncs at every logging
    boundary (it reads accumulated loss tensors there) and at every evaluation,
    which re-anchors the clock often enough that bucket totals over a logging
    interval are real. Per-update numbers are not; use `profile_update` for
    those.
    """

    def __init__(self, base: dict[str, float] | None = None) -> None:
        self.totals: dict[str, float] = dict(base or {})
        self._current: str | None = None
        self._since: float = 0.0

    def enter(self, bucket: str) -> None:
        now = time.perf_counter()
        if self._current is not None:
            self.totals[self._current] = self.totals.get(self._current, 0.0) + (now - self._since)
        self._current = bucket
        self._since = now

    def close(self) -> None:
        """Close the open bucket without opening another."""
        if self._current is not None:
            now = time.perf_counter()
            self.totals[self._current] = self.totals.get(self._current, 0.0) + (now - self._since)
            self._current = None

    @contextmanager
    def section(self, bucket: str):
        """Time a block, then return to whatever bucket was open before it."""
        previous = self._current
        self.enter(bucket)
        try:
            yield
        finally:
            if previous is None:
                self.close()
            else:
                self.enter(previous)

    def snapshot(self) -> dict[str, float]:
        """Bucket totals including the time accrued in the open bucket so far."""
        totals = dict(self.totals)
        if self._current is not None:
            totals[self._current] = (totals.get(self._current, 0.0)
                                     + (time.perf_counter() - self._since))
        return totals

    def total(self) -> float:
        return sum(self.snapshot().values())


def parameter_breakdown(model) -> dict[str, int]:
    """Stored, embedding, head, and compute-bearing parameter counts.

    `body` (blocks plus the final norm) is the historical scale axis and is kept
    for continuity, but it is not the compute axis: the untied 16,384-token head
    is a dense matmul on every position, and at the small rungs it is a large
    fraction of the FLOPs the model spends. `non_embedding` adds the head back
    and is what `estimate_flops_per_token` charges for
    ([D016](../../docs/decisions.md#d016)).
    """
    embedding = head = 0
    for name, parameter in model.named_parameters():
        lowered = name.lower()
        if "lm_head" in lowered:
            head += parameter.numel()
        elif "embed" in lowered:
            embedding += parameter.numel()
    stored = sum(parameter.numel() for parameter in model.parameters())
    return {
        "stored": stored,
        "embedding": embedding,
        "lm_head": head,
        "body": stored - embedding - head,
        "non_embedding": stored - embedding,
    }


def estimate_flops_per_token(config: ModernConfig, non_embedding_params: int,
                             sequence_length: int | None = None) -> float:
    """Forward+backward FLOPs per supervised token, the usual 6N + attention form.

    6N covers every dense matmul: 2 FLOPs per multiply-accumulate, one forward
    pass and two backward passes (activation and weight gradients) over each
    parameter that multiplies an activation. Embedding lookups are excluded --
    they are a gather, not a matmul -- while `lm_head` is included, which is the
    correction [D016](../../docs/decisions.md#d016) asks for.

    The attention score and context matmuls are not parameter-weighted, so they
    are added separately: per layer and per token, 2 * 2 * T * head_dim *
    n_heads forward, tripled for the backward pass.

    This is an estimate of *useful* arithmetic, not of issued instructions. It
    charges the full T x T score matrix even though causal masking means half of
    it is discarded, which is the convention MFU comparisons use; a flash kernel
    that skips the masked half will therefore report an MFU slightly below what
    it truly achieves. It also ignores norms, RoPE, softmax, and the elementwise
    SwiGLU product -- all bandwidth-bound work that costs real time and almost no
    FLOPs, which is exactly why MFU on this box reads low and why the wins so far
    came from moving fewer bytes rather than from doing less arithmetic.
    """
    seq = sequence_length if sequence_length is not None else config.max_seq_len
    dense = 6.0 * non_embedding_params
    attention = 12.0 * config.n_layers * seq * config.head_dim * config.n_heads
    return dense + attention


# Buckets that a token actually passes through. `data` is included because
# building a microbatch is on the critical path -- PackedTokenStream gathers rows
# in a Python loop and copies from pageable memory -- so excluding it would
# report a rate no configuration can sustain.
TRAINING_BUCKETS = ("data", "step")


def summarize(clock: SegmentClock, tokens: int, *, training_tokens: int | None = None,
              flops_per_token: float | None = None,
              device_peak_tflops: float | None = None,
              training_buckets: tuple[str, ...] = TRAINING_BUCKETS) -> dict[str, float | None]:
    """Segment totals plus the derived rates, for one log record.

    `tokens_per_second` stays end-to-end so it remains comparable with every
    historical `train.jsonl`; `training_tokens_per_second` is the new number and
    is the one to quote. MFU appears only when the caller supplies the device's
    peak -- see this module's docstring.

    `training_tokens` must count only the tokens whose time landed in the
    training buckets. Passing the run's full token count instead divides tokens
    the warmup update produced by a duration that excludes the warmup update,
    which reported 10.9M tok/s on a model that trains at five figures. The rate
    is None, not zero, until at least one token has been measured: no
    observation is a different statement from an observed standstill.
    """
    totals = clock.snapshot()
    for bucket in training_buckets:                 # a bucket at zero is a fact
        totals.setdefault(bucket, 0.0)              # worth logging, not a gap
    counted = tokens if training_tokens is None else training_tokens
    end_to_end = sum(totals.values())
    training = sum(totals.get(bucket, 0.0) for bucket in training_buckets)

    record: dict[str, float | None] = {
        f"seconds_{bucket}": value for bucket, value in sorted(totals.items())}
    record["elapsed_seconds"] = end_to_end
    record["tokens_per_second"] = tokens / max(1e-9, end_to_end)
    if counted <= 0 or training <= 0.0:
        record["training_tokens_per_second"] = None
        return record

    record["training_tokens_per_second"] = counted / training
    if flops_per_token:
        achieved = flops_per_token * counted / training
        record["training_tflops"] = achieved / 1e12
        if device_peak_tflops:
            record["mfu"] = achieved / (device_peak_tflops * 1e12)
    return record
