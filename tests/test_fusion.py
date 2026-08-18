"""Parity for fused Q/K/V and gate/up projections, and for checkpoint conversion.

[D003](../docs/decisions.md#d003) requires a semantics-preserving systems change
to demonstrate output, loss, gradient, optimizer-step and resume parity before
anyone measures its throughput. This is that demonstration, plus the measurement
that qualifies it: under Muon the change is NOT bit-exact end to end, because
Newton-Schulz runs in bf16 and amplifies float32-epsilon gradient differences.
The tolerances below are the measured sizes, not guesses
([D028](../docs/decisions.md#d028)).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from modern_lm.config import ModernConfig
from modern_lm.fusion import convert_checkpoint, fuse_state_dict, unfuse_state_dict
from modern_lm.model import ModernLM
from modern_lm.muon import Muon, build_optimizer, split_adamw_params
from modern_lm.train import TrainSettings, compute_loss


def build(fused: bool, seed: int = 2026, **overrides) -> ModernLM:
    torch.manual_seed(seed)
    return ModernLM(ModernConfig.tiny(fuse_projections=fused, **overrides))


def stack(model: ModernLM, prefix: str, names: tuple[str, ...]) -> torch.Tensor:
    state = model.state_dict()
    return torch.cat([state[f"{prefix}.{name}.weight"] for name in names], dim=0)


def optimizer_for(model: ModernLM):
    return build_optimizer(model, learning_rate=3e-4, muon_learning_rate=5e-3,
                           weight_decay=0.1)


def train_steps(model, optimizer, steps: int, seed: int = 7) -> list[float]:
    settings = TrainSettings(optimizer="muon")
    generator = torch.Generator().manual_seed(seed)
    losses = []
    for _ in range(steps):
        tokens = torch.randint(0, 128, (2, 17), generator=generator)
        optimizer.zero_grad(set_to_none=True)
        loss, _, _ = compute_loss(model, tokens, settings)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses


def test_fusion_preserves_parameter_count_and_names():
    plain, fused = build(False), build(True)
    assert plain.num_params() == fused.num_params()

    names = {name for name, _ in fused.named_parameters()}
    assert "blocks.0.attn.qkv_proj.weight" in names
    assert "blocks.0.feed_forward.gate_up_proj.weight" in names
    assert not {n for n in names if n.endswith(("q_proj.weight", "gate_proj.weight"))}


def test_a_fresh_fused_model_initializes_to_the_same_weights():
    """Same seed, same weights -- so the two are comparable from step zero."""
    plain, fused = build(False), build(True)
    state = fused.state_dict()

    assert torch.equal(stack(plain, "blocks.0.attn", ("q_proj", "k_proj", "v_proj")),
                       state["blocks.0.attn.qkv_proj.weight"])
    assert torch.equal(stack(plain, "blocks.0.feed_forward", ("gate_proj", "up_proj")),
                       state["blocks.0.feed_forward.gate_up_proj.weight"])
    # The residual-path rescale draws after the block weights; if fusion had
    # shifted the RNG stream, this would be the first casualty.
    assert torch.equal(plain.state_dict()["blocks.0.attn.o_proj.weight"],
                       state["blocks.0.attn.o_proj.weight"])


def test_forward_and_loss_are_identical():
    plain, fused = build(False), build(True)
    tokens = torch.randint(0, 128, (2, 17))

    assert torch.equal(plain(tokens[:, :-1]).logits, fused(tokens[:, :-1]).logits)
    settings = TrainSettings()
    plain_loss, _, _ = compute_loss(plain, tokens, settings)
    fused_loss, _, _ = compute_loss(fused, tokens, settings)
    assert plain_loss.item() == fused_loss.item()


def test_gradients_agree_to_float32_rounding():
    """One GEMM instead of three reduces in a different order; nothing more."""
    plain, fused = build(False), build(True)
    tokens = torch.randint(0, 128, (2, 17))
    for model in (plain, fused):
        loss, _, _ = compute_loss(model, tokens, TrainSettings())
        loss.backward()

    plain_grad = torch.cat([dict(plain.named_parameters())[
        f"blocks.0.attn.{name}_proj.weight"].grad for name in "qkv"], dim=0)
    fused_grad = dict(fused.named_parameters())["blocks.0.attn.qkv_proj.weight"].grad
    relative = (plain_grad - fused_grad).norm() / plain_grad.norm()
    assert relative < 1e-6, f"gradients diverge by {relative:.2e}, more than rounding"


def test_adamw_trajectories_stay_together():
    """Without Muon's bf16 stage the two runs track at float32 epsilon."""
    results = []
    for fused in (False, True):
        model = build(fused)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
        settings = TrainSettings()
        generator = torch.Generator().manual_seed(7)
        for _ in range(5):
            tokens = torch.randint(0, 128, (2, 17), generator=generator)
            optimizer.zero_grad(set_to_none=True)
            loss, _, _ = compute_loss(model, tokens, settings)
            loss.backward()
            optimizer.step()
        results.append(model)

    plain = stack(results[0], "blocks.0.attn", ("q_proj", "k_proj", "v_proj"))
    fused = results[1].state_dict()["blocks.0.attn.qkv_proj.weight"]
    relative = ((plain - fused).norm() / plain.norm()).item()
    assert relative < 1e-6, f"AdamW trajectories diverged by {relative:.2e}"


def test_muon_orthogonalizes_each_block_exactly_as_separate_matrices():
    """The load-bearing test: identical gradients must give identical updates.

    Orthogonalization is not separable, so a fused matrix stepped as one matrix
    gets a different update than its parts would. Feeding both paths the same
    gradients removes every other source of difference, leaving only whether the
    block splitting is right.
    """
    torch.manual_seed(0)
    dim, kv_rows = 64, 32
    weights = [torch.randn(rows, dim) for rows in (dim, kv_rows, kv_rows)]
    grads = [torch.randn(rows, dim) * 0.01 for rows in (dim, kv_rows, kv_rows)]

    separate = [torch.nn.Parameter(w.clone()) for w in weights]
    for parameter, grad in zip(separate, grads):
        parameter.grad = grad.clone()
    fused = torch.nn.Parameter(torch.cat(weights, dim=0))
    fused.grad = torch.cat(grads, dim=0).clone()

    separate_optimizer = Muon(separate, lr=0.005, weight_decay=0.1)
    fused_optimizer = Muon([{"params": [fused],
                             "row_blocks": (dim, kv_rows, kv_rows)}],
                           lr=0.005, weight_decay=0.1)
    for _ in range(3):
        separate_optimizer.step()
        fused_optimizer.step()

    expected = torch.cat([p.detach() for p in separate], dim=0)
    assert torch.equal(expected, fused.detach()), (
        "block-aware Muon does not reproduce the separate-matrix update")


def test_naive_fusion_would_have_changed_the_optimizer():
    """Guards the reason row_blocks exists, so nobody 'simplifies' it away."""
    torch.manual_seed(0)
    dim = 64
    weights = [torch.randn(dim, dim) for _ in range(3)]
    grads = [torch.randn(dim, dim) * 0.01 for _ in range(3)]

    separate = [torch.nn.Parameter(w.clone()) for w in weights]
    for parameter, grad in zip(separate, grads):
        parameter.grad = grad.clone()
    naive = torch.nn.Parameter(torch.cat(weights, dim=0))
    naive.grad = torch.cat(grads, dim=0).clone()

    separate_optimizer = Muon(separate, lr=0.005, weight_decay=0.1)
    naive_optimizer = Muon([naive], lr=0.005, weight_decay=0.1)   # no row_blocks
    for _ in range(3):
        separate_optimizer.step()
        naive_optimizer.step()

    expected = torch.cat([p.detach() for p in separate], dim=0)
    relative = ((naive.detach() - expected).norm() / expected.norm()).item()
    assert relative > 1e-4, (
        "naive fusion now matches the separate update; if Muon stopped using "
        "bf16 Newton-Schulz this test needs rewriting, not deleting")


def test_muon_group_layout_is_unchanged_for_an_unfused_model():
    """An unfused model must take exactly the path it always did."""
    optimizer = optimizer_for(build(False))
    groups = optimizer.optimizers[0].param_groups
    assert len(groups) == 1
    assert not groups[0].get("row_blocks")


def test_fused_model_groups_by_block_signature():
    optimizer = optimizer_for(build(True))
    signatures = [group.get("row_blocks") for group in optimizer.optimizers[0].param_groups]
    assert () in signatures, "o_proj and down_proj must stay unblocked"
    assert any(len(s) == 3 for s in signatures if s), "qkv needs a three-way split"
    assert any(len(s) == 2 for s in signatures if s), "gate/up needs a two-way split"


def test_full_muon_trajectories_stay_within_the_measured_tolerance():
    """End to end the two runs diverge, and this pins how much.

    bf16 Newton-Schulz turns a float32-epsilon gradient difference into a ~1e-3
    relative weight difference within a few steps. That is a property of Muon,
    not of fusion -- any change to GEMM reduction order does it -- but it is why
    fusion cannot be called bit-exact under Muon and must be validated as an
    approximate numerical change on a live trajectory.
    """
    plain, fused = build(False), build(True)
    plain_losses = train_steps(plain, optimizer_for(plain), 5)
    fused_losses = train_steps(fused, optimizer_for(fused), 5)

    assert plain_losses[0] == fused_losses[0], "step 0 runs before any divergence"
    for a, b in zip(plain_losses, fused_losses):
        assert abs(a - b) < 1e-3, f"loss trajectories separated: {a} vs {b}"

    expected = stack(plain, "blocks.0.attn", ("q_proj", "k_proj", "v_proj"))
    actual = fused.state_dict()["blocks.0.attn.qkv_proj.weight"]
    relative = ((expected - actual).norm() / expected.norm()).item()
    assert relative < 1e-2, f"weights diverged by {relative:.2e}"


def test_generation_matches_with_and_without_the_cache():
    plain, fused = build(False), build(True)
    plain.eval()
    fused.eval()
    prompt = torch.randint(0, 128, (1, 8))

    for use_cache in (False, True):
        a = plain.generate(prompt, max_new_tokens=6, use_cache=use_cache)
        b = fused.generate(prompt, max_new_tokens=6, use_cache=use_cache)
        assert torch.equal(a, b), f"generation differs with use_cache={use_cache}"


def test_grouped_query_attention_splits_on_the_right_rows():
    """The k/v blocks are smaller than q under GQA; a wrong split would mix them."""
    plain = build(False, n_kv_heads=2)
    fused = build(True, n_kv_heads=2)
    tokens = torch.randint(0, 128, (2, 17))
    assert torch.equal(plain(tokens).logits, fused(tokens).logits)


def _checkpoint(model, optimizer, steps: int = 3) -> dict:
    train_steps(model, optimizer, steps)
    return {"config": model.config.to_dict(), "model": model.state_dict(),
            "optimizer": optimizer.state_dict()}


def test_state_dict_conversion_round_trips_bitwise():
    model = build(False)
    original = {key: value.clone() for key, value in model.state_dict().items()}
    fused = fuse_state_dict(original, model.config)
    restored = unfuse_state_dict(fused, model.config)

    assert set(restored) == set(original)
    for key, value in original.items():
        assert torch.equal(value, restored[key]), key


def test_converted_checkpoint_loads_and_produces_identical_output():
    model = build(False)
    payload = _checkpoint(model, optimizer_for(model))

    converted = convert_checkpoint(payload, to_fused=True)
    assert converted["config"]["fuse_projections"] is True

    rebuilt = ModernLM(ModernConfig(**converted["config"]))
    rebuilt.load_state_dict(converted["model"])
    tokens = torch.randint(0, 128, (2, 16))
    model.eval()
    rebuilt.eval()
    with torch.no_grad():
        assert torch.equal(model(tokens).logits, rebuilt(tokens).logits)


def test_conversion_carries_optimizer_momentum_across():
    """A resume that silently zeroed momentum would look fine and train worse."""
    model = build(False)
    optimizer = optimizer_for(model)
    payload = _checkpoint(model, optimizer)

    converted = convert_checkpoint(payload, to_fused=True)
    rebuilt = ModernLM(ModernConfig(**converted["config"]))
    rebuilt.load_state_dict(converted["model"])
    rebuilt_optimizer = optimizer_for(rebuilt)
    rebuilt_optimizer.load_state_dict(converted["optimizer"])

    before = dict(model.named_parameters())
    source = torch.cat([optimizer.optimizers[0].state[
        before[f"blocks.0.attn.{name}_proj.weight"]]["momentum_buffer"]
        for name in "qkv"], dim=0)
    target = rebuilt_optimizer.optimizers[0].state[
        dict(rebuilt.named_parameters())["blocks.0.attn.qkv_proj.weight"]]["momentum_buffer"]

    assert source.abs().sum() > 0, "the fixture never accumulated momentum"
    assert torch.equal(source, target), "momentum did not survive conversion"


def test_conversion_round_trips_the_whole_payload():
    model = build(False)
    payload = _checkpoint(model, optimizer_for(model))
    restored = convert_checkpoint(convert_checkpoint(payload, to_fused=True),
                                  to_fused=False)

    for key, value in payload["model"].items():
        assert torch.equal(value, restored["model"][key]), key
    original_state = payload["optimizer"]["combined"][0]["state"]
    restored_state = restored["optimizer"]["combined"][0]["state"]
    assert set(original_state) == set(restored_state)
    for index, entry in original_state.items():
        assert torch.equal(entry["momentum_buffer"],
                           restored_state[index]["momentum_buffer"]), index


def test_converting_a_checkpoint_that_is_already_converted_is_refused():
    model = build(True)
    payload = {"config": model.config.to_dict(), "model": model.state_dict()}
    with pytest.raises(ValueError, match="already"):
        convert_checkpoint(payload, to_fused=True)


def test_unfusing_a_checkpoint_whose_config_disagrees_is_refused():
    """A wrong config would split the rows in the wrong places, silently."""
    model = build(True)
    payload = {"config": model.config.to_dict(), "model": model.state_dict()}
    payload["config"]["ffn_dim"] = model.config.ffn_dim + 8
    with pytest.raises(ValueError, match="rows"):
        convert_checkpoint(payload, to_fused=False)


def test_adamw_checkpoints_convert_too():
    """Not every run uses Muon; the plain-AdamW state dict has one index space."""
    model = build(False)
    # Built the way train.py builds it: decay group first, then the 1-D ones.
    decay, no_decay = split_adamw_params(model)
    optimizer = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1},
                                   {"params": no_decay, "weight_decay": 0.0}], lr=3e-4)
    settings = TrainSettings()
    tokens = torch.randint(0, 128, (2, 17))
    loss, _, _ = compute_loss(model, tokens, settings)
    loss.backward()
    optimizer.step()

    payload = {"config": model.config.to_dict(), "model": model.state_dict(),
               "optimizer": optimizer.state_dict()}
    converted = convert_checkpoint(payload, to_fused=True)

    rebuilt = ModernLM(ModernConfig(**converted["config"]))
    rebuilt.load_state_dict(converted["model"])
    rebuilt_decay, rebuilt_no_decay = split_adamw_params(rebuilt)
    rebuilt_optimizer = torch.optim.AdamW(
        [{"params": rebuilt_decay, "weight_decay": 0.1},
         {"params": rebuilt_no_decay, "weight_decay": 0.0}], lr=3e-4)
    rebuilt_optimizer.load_state_dict(converted["optimizer"])
    state = rebuilt_optimizer.state[
        dict(rebuilt.named_parameters())["blocks.0.attn.qkv_proj.weight"]]
    assert state["exp_avg"].shape[0] == sum(rebuilt.blocks[0].attn.qkv_splits)
