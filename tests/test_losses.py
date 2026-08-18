"""Parity and memory behaviour for the chunked vocabulary cross-entropy.

Two claims to keep honest. The arithmetic claim: the chunked loss is the same
loss, exactly in exact arithmetic, so in float64 it must match `F.cross_entropy`
to the last bits. The memory claim: it must not leave a full [tokens, vocab]
tensor alive for the backward pass -- which is checkable without a GPU by asking
autograd what it saved ([D030](../docs/decisions.md#d030)).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from modern_lm.config import ModernConfig
from modern_lm.losses import chunked_cross_entropy, reference_cross_entropy
from modern_lm.model import ModernLM
from modern_lm.train import TrainSettings, compute_loss

ROWS, DIM, VOCAB = 300, 32, 97


def inputs(dtype=torch.float64, ignore: bool = False, seed: int = 0):
    torch.manual_seed(seed)
    hidden = torch.randn(ROWS, DIM, dtype=dtype, requires_grad=True)
    weight = (torch.randn(VOCAB, DIM, dtype=dtype) * 0.02).requires_grad_()
    labels = torch.randint(0, VOCAB, (ROWS,))
    if ignore:
        labels[::7] = -100
    return hidden, weight, labels


def paired(dtype=torch.float64, ignore: bool = False):
    """The same problem twice, so both paths differentiate independent tensors."""
    hidden, weight, labels = inputs(dtype, ignore)
    return (hidden, weight, labels,
            hidden.detach().clone().requires_grad_(),
            weight.detach().clone().requires_grad_())


@pytest.mark.parametrize("ignore", [False, True])
def test_float64_matches_the_reference_exactly(ignore):
    """Exact arithmetic, exact agreement -- anything else is a real difference."""
    hidden, weight, labels, ref_hidden, ref_weight = paired(ignore=ignore)

    chunked_cross_entropy(hidden, weight, labels, chunk=64).backward()
    reference_cross_entropy(ref_hidden, ref_weight, labels).backward()

    assert (hidden.grad - ref_hidden.grad).abs().max() < 1e-15
    assert (weight.grad - ref_weight.grad).abs().max() < 1e-15


def test_float32_matches_to_rounding():
    hidden, weight, labels, ref_hidden, ref_weight = paired(torch.float32)

    loss = chunked_cross_entropy(hidden, weight, labels, chunk=64)
    reference = reference_cross_entropy(ref_hidden, ref_weight, labels)
    loss.backward()
    reference.backward()

    assert abs(loss.item() - reference.item()) < 1e-5
    assert (hidden.grad - ref_hidden.grad).abs().max() < 1e-6
    assert (weight.grad - ref_weight.grad).abs().max() < 1e-6


@pytest.mark.parametrize("chunk", [1, 7, 64, ROWS, ROWS * 4])
def test_the_chunk_size_is_not_a_hyperparameter(chunk):
    """Including chunks that do not divide the batch, and one larger than it."""
    hidden, weight, labels, ref_hidden, ref_weight = paired()

    loss = chunked_cross_entropy(hidden, weight, labels, chunk=chunk)
    reference = reference_cross_entropy(ref_hidden, ref_weight, labels)
    loss.backward()
    reference.backward()

    assert abs(loss.item() - reference.item()) < 1e-12
    assert (weight.grad - ref_weight.grad).abs().max() < 1e-14


def test_ignored_positions_contribute_nothing():
    hidden, weight, labels = inputs(ignore=True)
    loss = chunked_cross_entropy(hidden, weight, labels, chunk=16)
    loss.backward()

    # An ignored row must receive no gradient at all, not merely a small one.
    ignored = (labels == -100).nonzero().squeeze(1)
    assert ignored.numel() > 0
    assert torch.equal(hidden.grad[ignored], torch.zeros_like(hidden.grad[ignored]))


def test_an_entirely_ignored_batch_is_zero_not_nan():
    """A NaN here would surface as a dead optimizer two steps later."""
    hidden, weight, _ = inputs()
    labels = torch.full((ROWS,), -100)

    loss = chunked_cross_entropy(hidden, weight, labels, chunk=16)
    loss.backward()

    assert loss.item() == 0.0
    assert torch.isfinite(hidden.grad).all()


def test_bf16_autocast_matches_the_standard_path():
    """The dtype production actually trains in."""
    hidden, weight, labels, ref_hidden, ref_weight = paired(torch.float32)

    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss = chunked_cross_entropy(hidden, weight, labels, chunk=64)
        reference = reference_cross_entropy(ref_hidden, ref_weight, labels)
    loss.backward()
    reference.backward()

    assert abs(loss.item() - reference.item()) < 1e-3, "bf16 tolerance, not a free pass"
    # Measured 2.3e-3 -- bf16's own noise floor on a weight gradient, not an
    # error introduced by chunking; the test below establishes which is which.
    relative = ((weight.grad - ref_weight.grad).norm() / ref_weight.grad.norm()).item()
    assert relative < 1e-2, f"weight gradients diverged by {relative:.2e}"


def test_chunking_is_no_less_accurate_than_the_standard_path_in_bf16():
    """The two bf16 paths differ by 2.3e-3; this says neither is the wrong one.

    Both are compared against the same problem solved in float64. If chunking
    were losing accuracy -- accumulating partial sums badly, say -- its error
    against the truth would exceed the standard path's. It does not: both sit on
    bf16's floor, and per-chunk GEMMs accumulated in fp32 land a hair closer.
    """
    rows, dim, vocab = 512, 64, 1024
    torch.manual_seed(0)
    base_hidden = torch.randn(rows, dim) * 0.5
    base_weight = torch.randn(vocab, dim) * 0.02
    labels = torch.randint(0, vocab, (rows,))

    def weight_gradient(loss_fn, dtype, autocast, **kwargs):
        hidden = base_hidden.clone().to(dtype).requires_grad_()
        weight = base_weight.clone().to(dtype).requires_grad_()
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
            loss_fn(hidden, weight, labels, **kwargs).backward()
        return weight.grad.double()

    truth = weight_gradient(reference_cross_entropy, torch.float64, False)
    standard = weight_gradient(reference_cross_entropy, torch.float32, True)
    chunked = weight_gradient(chunked_cross_entropy, torch.float32, True, chunk=128)

    standard_error = ((standard - truth).norm() / truth.norm()).item()
    chunked_error = ((chunked - truth).norm() / truth.norm()).item()
    assert chunked_error <= standard_error * 1.1, (
        f"chunking lost accuracy: {chunked_error:.2e} against {standard_error:.2e}")


def test_mismatched_rows_and_labels_are_refused():
    hidden, weight, _ = inputs()
    with pytest.raises(ValueError, match="labels"):
        chunked_cross_entropy(hidden, weight, torch.zeros(ROWS + 1, dtype=torch.long))


def test_a_nonpositive_chunk_is_refused():
    hidden, weight, labels = inputs()
    with pytest.raises(ValueError, match="chunk"):
        chunked_cross_entropy(hidden, weight, labels, chunk=0)


def _saved_tensor_sizes(model, tokens, settings) -> list[int]:
    """Element counts of everything autograd retained for the backward pass."""
    sizes: list[int] = []

    def pack(tensor):
        sizes.append(tensor.numel())
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        compute_loss(model, tokens, settings)
    return sizes


def test_no_full_logit_tensor_survives_into_the_backward_pass():
    """The memory claim, checked directly rather than asserted in a comment.

    The standard path saves the logits and cross-entropy's own intermediate --
    two tensors of tokens x vocab. At the production shape those are 1.07 GB
    each; here they are small, but their presence or absence is the same fact.
    """
    torch.manual_seed(2026)
    config = ModernConfig.tiny()
    model = ModernLM(config)
    batch, length = 4, 32
    tokens = torch.randint(0, config.vocab_size, (batch, length + 1))
    full_logits = batch * length * config.vocab_size

    standard = _saved_tensor_sizes(model, tokens, TrainSettings())
    chunked = _saved_tensor_sizes(
        model, tokens, TrainSettings(chunked_cross_entropy=True, cross_entropy_chunk=16))

    assert [s for s in standard if s >= full_logits], (
        "fixture is wrong: the standard path should retain a full logit tensor")
    assert not [s for s in chunked if s >= full_logits], (
        "the chunked path retained a full-size logit tensor")
    assert sum(chunked) < sum(standard)


def test_compute_loss_agrees_with_the_standard_path_everywhere():
    """Every parameter's gradient, not just the head's."""
    torch.manual_seed(2026)
    model = ModernLM(ModernConfig.tiny())
    tokens = torch.randint(0, 128, (2, 17))

    standard, _, _ = compute_loss(model, tokens, TrainSettings())
    standard.backward()
    reference_grads = {name: p.grad.clone() for name, p in model.named_parameters()}

    model.zero_grad(set_to_none=True)
    chunked, _, _ = compute_loss(
        model, tokens, TrainSettings(chunked_cross_entropy=True, cross_entropy_chunk=8))
    chunked.backward()

    assert abs(standard.item() - chunked.item()) < 1e-5
    for name, parameter in model.named_parameters():
        expected = reference_grads[name]
        relative = ((parameter.grad - expected).norm()
                    / expected.norm().clamp_min(1e-12)).item()
        assert relative < 1e-5, f"{name} gradient diverged by {relative:.2e}"


def test_compute_loss_returns_the_same_target_count():
    """Token accounting must not change with the loss implementation."""
    model = ModernLM(ModernConfig.tiny())
    tokens = torch.randint(0, 128, (3, 33))

    _, _, standard = compute_loss(model, tokens, TrainSettings())
    _, _, chunked = compute_loss(
        model, tokens, TrainSettings(chunked_cross_entropy=True))
    assert standard == chunked == 3 * 32


def test_it_composes_with_fused_projections():
    """The two Tier-2 changes touch different parts of the step; both can be on."""
    torch.manual_seed(2026)
    plain = ModernLM(ModernConfig.tiny(fuse_projections=False))
    torch.manual_seed(2026)
    fused = ModernLM(ModernConfig.tiny(fuse_projections=True))
    tokens = torch.randint(0, 128, (2, 17))
    settings = TrainSettings(chunked_cross_entropy=True, cross_entropy_chunk=8)

    plain_loss, _, _ = compute_loss(plain, tokens, settings)
    fused_loss, _, _ = compute_loss(fused, tokens, settings)
    assert plain_loss.item() == fused_loss.item()


def test_hidden_states_reach_the_caller_only_when_asked():
    model = ModernLM(ModernConfig.tiny())
    tokens = torch.randint(0, 128, (2, 16))

    standard = model(tokens)
    assert standard.logits is not None and standard.hidden is None

    withheld = model(tokens, return_hidden=True)
    assert withheld.logits is None and withheld.hidden is not None
    # The head applied afterwards must reproduce exactly what forward would have.
    assert torch.equal(model.lm_head(withheld.hidden), standard.logits)
