"""Cross-entropy that never materializes the full logit tensor.

At the batch shape [D024](../../docs/decisions.md#d024) settled on -- 64 rows of
512 tokens, 32,768 targets per update -- the vocabulary projection produces a
[32,768, 16,384] tensor. In bf16 that is 1.07 GB, autograd saves it for the
backward pass, and the gradient with respect to it is another 1.07 GB. Roughly
2.1 GB of traffic, per micro-batch, for a quantity that is summarized into one
scalar and then thrown away. On a box whose every measured win has come from
moving fewer bytes, that is the largest single tensor in the step.

`chunked_cross_entropy` computes the same loss a slice of rows at a time and
recomputes each slice's logits during the backward pass instead of storing them.
Peak logit memory becomes one chunk rather than the whole batch. The trade is one
extra projection matmul: FLOPs go up, bytes go down. Which of those the hardware
cares about is a measurement (`scripts/bench_cross_entropy.py`), not a deduction
([D030](../../docs/decisions.md#d030)).

The arithmetic is the textbook one -- `logsumexp(z) - z[target]`, averaged over
unignored targets -- so this is a semantics-preserving change in exact arithmetic.
In floating point it reduces in a different order than one big `F.cross_entropy`,
which under Muon is not a difference that stays small ([D029](../../docs/decisions.md#d029)).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

# Rows per slice. 4,096 x 16,384 in fp32 is 268 MB of probabilities, the largest
# thing this holds at once, against 32,768 rows' 2.1 GB for the unchunked path.
# Smaller chunks save more memory and launch more kernels; this is the knob to
# turn if the bench says the trade landed wrong.
DEFAULT_CHUNK = 4096


def _accumulate_dtype(compute_dtype: torch.dtype) -> torch.dtype:
    """Where the softmax normalizer is summed.

    Reduced precision is upcast to fp32, which is what `F.cross_entropy` does
    internally and where a bf16 cross-entropy would otherwise lose accuracy. A
    caller already working in fp32 or fp64 keeps its own precision -- an
    unconditional `.float()` here would silently DOWNCAST an fp64 computation,
    which is the sort of thing that makes a parity test pass for the wrong
    reason.
    """
    if compute_dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    return compute_dtype


def _matmul_dtype(hidden: torch.Tensor) -> torch.dtype:
    """The dtype the projection would run in under the caller's autocast.

    Captured in the forward pass and reused in the backward one. A custom
    autograd Function's backward runs with autocast OFF, so recomputing the
    logits without pinning this would silently do the recomputation in fp32 --
    a different number than the forward produced, and slower than the matmul it
    is meant to replace.
    """
    device_type = hidden.device.type
    if torch.is_autocast_enabled(device_type):
        return torch.get_autocast_dtype(device_type)
    return hidden.dtype


class _ChunkedCrossEntropy(torch.autograd.Function):
    """Loss over `hidden @ weight.T` computed and differentiated in row slices."""

    @staticmethod
    def forward(ctx, hidden: torch.Tensor, weight: torch.Tensor, labels: torch.Tensor,
                chunk: int, ignore_index: int) -> torch.Tensor:
        compute_dtype = _matmul_dtype(hidden)
        accumulate = _accumulate_dtype(compute_dtype)
        rows = hidden.shape[0]

        # Accumulate on the device. Reading a running total back to Python per
        # chunk would reintroduce exactly the synchronization D023 removed.
        total = torch.zeros((), device=hidden.device, dtype=accumulate)
        counted = torch.zeros((), device=hidden.device, dtype=accumulate)

        for start in range(0, rows, chunk):
            stop = min(start + chunk, rows)
            slice_labels = labels[start:stop]
            logits = (hidden[start:stop].to(compute_dtype)
                      @ weight.to(compute_dtype).T).to(accumulate)
            # log-sum-exp in fp32: the softmax normalizer is where a bf16
            # cross-entropy loses accuracy, which is why F.cross_entropy upcasts
            # internally too.
            log_denominator = torch.logsumexp(logits, dim=-1)
            safe = slice_labels.clamp_min(0).unsqueeze(1)
            target = logits.gather(1, safe).squeeze(1)
            keep = (slice_labels != ignore_index).to(accumulate)
            total += ((log_denominator - target) * keep).sum()
            counted += keep.sum()

        ctx.save_for_backward(hidden, weight, labels)
        ctx.chunk = chunk
        ctx.ignore_index = ignore_index
        ctx.compute_dtype = compute_dtype
        # An all-ignored batch has no defined mean; report zero loss rather than
        # a NaN that would poison the optimizer two steps later.
        return total / counted.clamp_min(1.0)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        hidden, weight, labels = ctx.saved_tensors
        chunk, ignore_index = ctx.chunk, ctx.ignore_index
        compute_dtype = ctx.compute_dtype
        accumulate = _accumulate_dtype(compute_dtype)
        rows = hidden.shape[0]

        counted = (labels != ignore_index).sum().clamp_min(1).to(accumulate)
        scale = grad_output / counted

        grad_hidden = torch.zeros_like(hidden) if ctx.needs_input_grad[0] else None
        grad_weight = (torch.zeros_like(weight, dtype=accumulate)
                       if ctx.needs_input_grad[1] else None)
        weight_compute = weight.to(compute_dtype)

        for start in range(0, rows, chunk):
            stop = min(start + chunk, rows)
            slice_hidden = hidden[start:stop].to(compute_dtype)
            slice_labels = labels[start:stop]
            logits = (slice_hidden @ weight_compute.T).to(accumulate)

            # softmax in place, from the same logsumexp the forward used:
            # exp(z - lse) is the softmax, and doing it in place keeps one
            # chunk-sized fp32 tensor alive instead of two.
            logits.sub_(torch.logsumexp(logits, dim=-1, keepdim=True)).exp_()
            probabilities = logits
            rows_index = torch.arange(stop - start, device=hidden.device)
            probabilities[rows_index, slice_labels.clamp_min(0)] -= 1.0
            probabilities *= (slice_labels != ignore_index).to(accumulate).unsqueeze(1)
            probabilities *= scale

            grad_logits = probabilities.to(compute_dtype)
            if grad_hidden is not None:
                grad_hidden[start:stop] = (grad_logits @ weight_compute).to(hidden.dtype)
            if grad_weight is not None:
                # Per-chunk GEMM in the compute dtype, accumulated in fp32: the
                # sum runs over as many chunks as the batch has, and adding
                # bf16 partials into a bf16 total would lose the small ones.
                grad_weight += (grad_logits.T @ slice_hidden).to(accumulate)

        return (grad_hidden,
                None if grad_weight is None else grad_weight.to(weight.dtype),
                None, None, None)


def chunked_cross_entropy(hidden: torch.Tensor, weight: torch.Tensor,
                          labels: torch.Tensor, *, chunk: int = DEFAULT_CHUNK,
                          ignore_index: int = -100) -> torch.Tensor:
    """Mean cross-entropy of `hidden @ weight.T` against `labels`.

    `hidden` is [N, D] or [..., D] and is flattened; `weight` is the vocabulary
    projection [V, D]; `labels` holds one target index per row, with
    `ignore_index` marking positions that do not count.

    Equivalent to `F.cross_entropy(hidden @ weight.T, labels, ignore_index=...)`
    but with peak logit memory proportional to `chunk` rather than to N.
    """
    flat_hidden = hidden.reshape(-1, hidden.shape[-1])
    flat_labels = labels.reshape(-1)
    if flat_hidden.shape[0] != flat_labels.shape[0]:
        raise ValueError(
            f"{flat_hidden.shape[0]} hidden rows against {flat_labels.shape[0]} labels")
    if chunk <= 0:
        raise ValueError("chunk must be positive")
    return _ChunkedCrossEntropy.apply(flat_hidden, weight, flat_labels, chunk,
                                      ignore_index)


def reference_cross_entropy(hidden: torch.Tensor, weight: torch.Tensor,
                            labels: torch.Tensor, *,
                            ignore_index: int = -100) -> torch.Tensor:
    """The unchunked equivalent, for tests and for the bench's baseline arm."""
    logits = F.linear(hidden.reshape(-1, hidden.shape[-1]), weight)
    return F.cross_entropy(logits, labels.reshape(-1), ignore_index=ignore_index)
