"""Optional GB10 integration tests for the real Transformer Engine backend.

The normal CPU environment skips these. Run them from the optional environment:

    PYTHONPATH=src .venv/bin/python -m pytest tests/test_low_precision_gpu.py -q
"""
from __future__ import annotations

import importlib.util

import pytest
import torch

from modern_lm.config import ModernConfig
from modern_lm.low_precision import (
    canonical_model_state_dict,
    configure_low_precision,
    load_canonical_model_state_dict,
)
from modern_lm.model import ModernLM
from modern_lm.muon import build_optimizer
from modern_lm.train import TrainSettings, compute_loss


def _te_available() -> bool:
    try:
        return importlib.util.find_spec("transformer_engine") is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not _te_available(),
    reason="requires CUDA and the optional Transformer Engine environment",
)


@pytest.mark.parametrize("precision", ("fp8", "nvfp4"))
def test_real_backend_optimizer_compile_and_checkpoint_portability(precision):
    device = torch.device("cuda")
    torch.manual_seed(2026)
    torch.cuda.manual_seed_all(2026)
    config = ModernConfig.tiny()
    reference = ModernLM(config)
    initial = reference.state_dict()

    model = ModernLM(config).to(device)
    load_canonical_model_state_dict(model, initial)
    report = configure_low_precision(model, precision, device)
    assert report.converted_linears == 14
    assert not report.skipped_linears
    if precision == "nvfp4" and torch.cuda.get_device_capability(device) == (12, 1):
        assert report.disabled_features == ("stochastic_rounding",)

    # This is the production ordering: precision conversion, compile, then
    # optimizer construction. TE deliberately graph-breaks around its Linear;
    # this test establishes functional compatibility, not a full-graph claim.
    model = torch.compile(model)
    settings = TrainSettings(optimizer="muon", precision=precision,
                             sequence_length=32)
    optimizer = build_optimizer(
        model, learning_rate=settings.learning_rate,
        muon_learning_rate=settings.muon_learning_rate,
        weight_decay=settings.weight_decay,
        muon_weight_decay=settings.effective_muon_weight_decay())
    microbatches = [
        torch.randint(0, config.vocab_size, (2, 33), device=device)
        for _ in range(2)
    ]
    before = model._orig_mod.blocks[0].attn.q_proj.weight.detach().clone()

    optimizer.zero_grad(set_to_none=True)
    losses = []
    for index, tokens in enumerate(microbatches):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, _, _ = compute_loss(
                model, tokens, settings, is_first_microbatch=(index == 0))
        (loss / len(microbatches)).backward()
        losses.append(loss.detach())
    assert all(torch.isfinite(loss) for loss in losses)
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all()
               for parameter in model.parameters())
    optimizer.step()
    after = model._orig_mod.blocks[0].attn.q_proj.weight.detach()
    assert not torch.equal(before, after)

    # TE quantizer state is disposable. A canonical checkpoint and optimizer
    # payload load into a native model with the same parameter ordering.
    canonical = canonical_model_state_dict(model)
    assert not any(key.endswith("._extra_state") for key in canonical)
    native = ModernLM(config).to(device)
    load_canonical_model_state_dict(native, canonical)
    native_optimizer = build_optimizer(
        native, learning_rate=settings.learning_rate,
        muon_learning_rate=settings.muon_learning_rate,
        weight_decay=settings.weight_decay,
        muon_weight_decay=settings.effective_muon_weight_decay())
    native_optimizer.load_state_dict(optimizer.state_dict())
    assert torch.equal(native.blocks[0].attn.q_proj.weight, after)
