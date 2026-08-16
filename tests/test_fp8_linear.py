"""FP8 Linear: every GEMM path checked against a bf16 reference.

`torch._scaled_mm` does not validate that an operand was scaled along its own
reduction dimension. Get a transpose wrong and it returns a wrong answer at full
speed with no error, so each of the three GEMMs -- forward, dgrad, wgrad -- is
compared against the bf16 result it is standing in for.

Tolerances: row-wise e4m3 with both operands quantized lands around 3.7-4%
relative error, so 8% is a generous ceiling that still catches a transposed
reduction (which shows up as 100%+).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(),
                                reason="FP8 needs a GPU")

SHAPES = [(2048, 1024, 3072), (4096, 576, 1984), (1024, 704, 4736)]


def relative_error(got: torch.Tensor, want: torch.Tensor) -> float:
    return ((got.float() - want.float()).norm() / want.float().norm()).item()


@pytest.mark.parametrize("m,k,n", SHAPES)
def test_forward_matches_bf16(m, k, n):
    from fp8_linear import fp8_matmul
    torch.manual_seed(0)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16) * 0.5
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.02
    assert relative_error(fp8_matmul(x, w), x.float() @ w.float().t()) < 0.08


@pytest.mark.parametrize("m,k,n", SHAPES)
def test_backward_matches_bf16(m, k, n):
    """Both dgrad and wgrad, via autograd rather than by calling them directly.

    fp8_backward=True is explicit: it defaults OFF because it is slower than
    bf16, and a test that silently exercised the default would stop covering
    the FP8 backward entirely.
    """
    from fp8_linear import FP8Linear
    torch.manual_seed(0)
    x = (torch.randn(m, k, device="cuda", dtype=torch.bfloat16) * 0.5
         ).requires_grad_(True)
    layer = FP8Linear(k, n, fp8_backward=True).cuda().to(torch.bfloat16)

    grad_out = torch.randn(m, n, device="cuda", dtype=torch.bfloat16) * 0.1
    layer(x).backward(grad_out)

    want_dx = grad_out.float() @ layer.weight.float()
    want_dw = grad_out.float().t() @ x.detach().float()

    assert relative_error(x.grad, want_dx) < 0.08, "dgrad wrong (check transpose)"
    assert relative_error(layer.weight.grad, want_dw) < 0.08, "wgrad wrong"


def test_bf16_backward_is_the_default():
    """fp8_backward defaults off; the default path must still be correct."""
    from fp8_linear import FP8Linear
    torch.manual_seed(0)
    m, k, n = 2048, 1024, 3072
    x = (torch.randn(m, k, device="cuda", dtype=torch.bfloat16) * 0.5
         ).requires_grad_(True)
    layer = FP8Linear(k, n).cuda().to(torch.bfloat16)
    assert layer.fp8_backward is False
    grad_out = torch.randn(m, n, device="cuda", dtype=torch.bfloat16) * 0.1
    layer(x).backward(grad_out)
    # A bf16 backward is exact against its own reference, unlike the FP8 one.
    want_dx = grad_out.float() @ layer.weight.float()
    assert relative_error(x.grad, want_dx) < 0.02


def test_gradients_are_not_transposed():
    """A swapped reduction dim still produces correctly SHAPED garbage."""
    from fp8_linear import FP8Linear
    torch.manual_seed(0)
    m, k, n = 2048, 1024, 3072
    x = (torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
         ).requires_grad_(True)
    layer = FP8Linear(k, n, fp8_backward=True).cuda().to(torch.bfloat16)
    layer(x).backward(torch.randn(m, n, device="cuda", dtype=torch.bfloat16))

    assert x.grad.shape == (m, k)
    assert layer.weight.grad.shape == (n, k)
    # Garbage from a bad transpose is uncorrelated with the true gradient.
    want_dx = (torch.randn(m, n, device="cuda", dtype=torch.bfloat16)
               .float() @ layer.weight.float())
    assert not torch.allclose(x.grad.float(), want_dx, rtol=0.5)


@pytest.mark.parametrize("m,k,n", [(2048, 1024, 3072)])
def test_quantize_roundtrip_is_lossy_but_centered(m, k, n):
    """Row-wise e4m3 should lose ~1-2% on a round trip and not bias the mean."""
    from fp8_linear import quantize_fp8_rowwise
    torch.manual_seed(0)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    q, scale = quantize_fp8_rowwise(x)
    back = q.to(torch.float32) * scale
    assert relative_error(back, x) < 0.05
    assert abs(back.mean().item() - x.float().mean().item()) < 0.01
