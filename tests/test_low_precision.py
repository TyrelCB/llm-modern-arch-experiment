"""Precision recipe, projection boundary, and checkpoint portability tests."""
from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from modern_lm.config import ModernConfig
from modern_lm.low_precision import (
    _recipe_for,
    _replace_linears,
    canonical_model_state_dict,
    configure_low_precision,
    load_canonical_model_state_dict,
    low_precision_autocast,
    validate_precision,
)
from modern_lm.model import ModernLM


class _CurrentScaling:
    pass


class _NVFP4:
    def __init__(self, **options):
        self.options = options


RECIPE = SimpleNamespace(
    Float8CurrentScaling=_CurrentScaling,
    NVFP4BlockScaling=_NVFP4,
)


class _FakeTELinear(nn.Linear):
    """CPU stand-in exposing the state-dict behavior of TE Linear."""

    def __init__(self, in_features, out_features, *, bias, params_dtype,
                 device, name):
        super().__init__(in_features, out_features, bias=bias,
                         device=device, dtype=params_dtype)
        self.te_name = name

    def get_extra_state(self):
        return {"disposable_quantizer_cache": True}

    def set_extra_state(self, state):
        pass


class _FakeTE:
    Linear = _FakeTELinear

    @staticmethod
    def autocast(**kwargs):
        return nullcontext()


def test_precision_names_are_validated_and_normalized():
    assert validate_precision("FP8") == "fp8"
    assert validate_precision("nvfp4") == "nvfp4"
    with pytest.raises(ValueError, match="bf16, fp8, nvfp4"):
        validate_precision("fp16")


def test_fp8_uses_current_scaling_recipe():
    selected, disabled = _recipe_for("fp8", RECIPE, (12, 1))
    assert isinstance(selected, _CurrentScaling)
    assert disabled == ()


@pytest.mark.parametrize(
    ("capability", "options", "disabled"),
    [
        ((10, 0), {}, ()),
        ((10, 3), {}, ()),
        ((12, 1), {"disable_stochastic_rounding": True},
         ("stochastic_rounding",)),
        ((12, 0), {"disable_stochastic_rounding": True, "disable_rht": True},
         ("stochastic_rounding", "random_hadamard_transform")),
    ],
)
def test_nvfp4_recipe_matches_blackwell_family_constraints(
        capability, options, disabled):
    selected, actual_disabled = _recipe_for("nvfp4", RECIPE, capability)
    assert selected.options == options
    assert actual_disabled == disabled


def test_nvfp4_rejects_pre_blackwell_gpu():
    with pytest.raises(RuntimeError, match="Blackwell"):
        _recipe_for("nvfp4", RECIPE, (9, 0))


def test_conversion_is_projection_only_weight_exact_and_rng_neutral():
    torch.manual_seed(7)
    model = ModernLM(ModernConfig.tiny(fuse_projections=True))
    original = {name: value.detach().clone()
                for name, value in model.state_dict().items()}
    rng_before = torch.get_rng_state().clone()

    converted, skipped = _replace_linears(model, _FakeTE)

    assert skipped == ()
    assert converted
    assert all(name.startswith(("blocks.", "mtp.")) for name in converted)
    assert type(model.lm_head) is nn.Linear
    assert all("router" not in name for name in converted)
    assert model.blocks[0].attn.qkv_proj.muon_row_blocks == model.blocks[0].attn.qkv_splits
    assert model.blocks[0].feed_forward.gate_up_proj.muon_row_blocks == (
        model.config.ffn_dim, model.config.ffn_dim)
    assert torch.equal(torch.get_rng_state(), rng_before)
    for name, value in original.items():
        assert torch.equal(model.state_dict()[name], value), name


def test_canonical_checkpoint_round_trips_between_native_and_te_modules():
    torch.manual_seed(11)
    native = ModernLM(ModernConfig.tiny())
    te_model = ModernLM(ModernConfig.tiny())
    _replace_linears(te_model, _FakeTE)

    load_canonical_model_state_dict(te_model, native.state_dict())
    canonical = canonical_model_state_dict(te_model)
    assert not any(key.endswith("._extra_state") for key in canonical)

    restored = ModernLM(ModernConfig.tiny())
    load_canonical_model_state_dict(restored, canonical)
    for name, value in native.state_dict().items():
        assert torch.equal(restored.state_dict()[name], value), name


def test_checkpoint_loading_remains_strict_outside_te_caches():
    model = ModernLM(ModernConfig.tiny())
    state = model.state_dict()
    state["not_a_projection._extra_state"] = {"unexpected": True}
    with pytest.raises(RuntimeError, match="unexpected keys"):
        load_canonical_model_state_dict(model, state)


def test_bf16_configuration_is_idempotent_and_context_is_noop():
    model = ModernLM(ModernConfig.tiny())
    first = configure_low_precision(model, "bf16", torch.device("cpu"))
    second = configure_low_precision(model, "bf16", torch.device("cpu"))
    assert first == second
    assert first.backend == "torch"
    with low_precision_autocast(model):
        value = 3
    assert value == 3
    with pytest.raises(RuntimeError, match="already configured"):
        configure_low_precision(model, "fp8", torch.device("cpu"))


def test_low_precision_requires_cuda_before_importing_optional_backend():
    model = ModernLM(ModernConfig.tiny())
    with pytest.raises(RuntimeError, match="requires a CUDA device"):
        configure_low_precision(model, "fp8", torch.device("cpu"))
