"""A drop-in FP8 Linear, for the same throughput question NVFP4 answered "no" to.

FP8's economics differ from NVFP4's in one structural way: scaling is per-row,
not per-16-element block. That removes the packing and the swizzled scale
layout entirely -- the quantizer is an amax, a divide, and a cast. Whether that
is cheap ENOUGH is the open question, since the FP8 GEMM is only ~2x bf16 where
FP4's is ~3.5x.

Same scope caveat as nvfp4_linear: forward GEMM only, straight-through bf16
backward. This measures the ceiling, not a training recipe.
"""
from __future__ import annotations

import torch
import torch.nn as nn

E4M3_MAX = 448.0
_QUANTIZE = None


def quantize_fp8_rowwise(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """bf16 [rows, K] -> (e4m3 values, fp32 per-row scales).

    Row-wise rather than per-tensor: a single outlier row would otherwise crush
    the whole tensor's resolution. This is the scaling granularity
    `torch._scaled_mm` expects for its RowWise path.
    """
    scale = (x.abs().amax(dim=-1, keepdim=True) / E4M3_MAX).clamp(min=1e-12)
    return (x / scale).to(torch.float8_e4m3fn), scale.float()


def _compiled_quantize():
    global _QUANTIZE
    if _QUANTIZE is None:
        _QUANTIZE = torch.compile(quantize_fp8_rowwise, dynamic=False)
    return _QUANTIZE


def fp8_matmul(a: torch.Tensor, weight: torch.Tensor,
               compiled: bool = True) -> torch.Tensor:
    """a [M, K] @ weight.T with weight [N, K], both bf16, via the FP8 kernel."""
    quantize = _compiled_quantize() if compiled else quantize_fp8_rowwise
    a8, scale_a = quantize(a)
    w8, scale_w = quantize(weight)
    return torch._scaled_mm(a8, w8.t(), scale_a=scale_a, scale_b=scale_w.t(),
                            out_dtype=torch.bfloat16)


class FP8Linear(nn.Module):
    """nn.Linear whose forward GEMM runs in FP8; bf16 straight-through backward."""

    def __init__(self, in_features: int, out_features: int, bias: bool = False,
                 fp8_backward: bool = False):
        super().__init__()
        # fp8_backward defaults off: it is correct but 0.76x vs bf16, while the
        # forward-only path is 0.99x. See _FP8Function for why.
        self.fp8_backward = fp8_backward
        self.in_features, self.out_features = in_features, out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.normal_(self.weight, std=0.02)
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        out = _FP8Function.apply(x.reshape(-1, shape[-1]), self.weight,
                                 self.fp8_backward)
        out = out.reshape(*shape[:-1], self.out_features)
        return out if self.bias is None else out + self.bias


class _FP8Function(torch.autograd.Function):
    """Forward and both backward GEMMs in FP8.

    The forward is only ~29% of step time and the backward ~61%, so quantizing
    the forward alone caps the achievable speedup near 1.03x however fast the
    GEMM gets. Both backward GEMMs have to move for FP8 to be worth anything.

    Every `_scaled_mm` RowWise call needs each operand scaled along its OWN
    reduction dimension, which is what dictates the transposes below:

        forward  out  [M,N] = x [M,K] @ w.T      reduce K: scale x and w over K
        dgrad    dx   [M,K] = grad [M,N] @ w     reduce N: scale grad over N,
                                                 and w over N -- which is w.T's
                                                 row dim, hence the transpose
        wgrad    dw   [N,K] = grad.T [N,M] @ x   reduce M: scale grad.T and x.T
                                                 over M, so both transpose

    A transpose that gets the reduction dim wrong does not error, it silently
    returns garbage, so the tests check every path against a bf16 reference.

    MEASURED RESULT: this is SLOWER than bf16 -- 0.76x end to end at 300M's
    shape, against 0.99x for the forward-only version. The transposes are why.
    `_scaled_mm` requires `a.stride(1) == 1`, so a column-major view is rejected
    and `grad.T` / `x.T` must be materialized. At [8192, 3072] that copy costs
    0.851ms while the whole bf16 wgrad it feeds costs 0.501ms. Quantization
    itself is fine (0.33ms); the layout requirement is what kills it.

    Escaping this needs a kernel that scales along an arbitrary axis, or one
    fused transpose+quantize pass -- which is what TransformerEngine ships and
    what the published 22%-throughput FP8 recipes rely on. Kept as the honest
    negative result: the backward is reachable through the public API but not
    profitably.
    """

    @staticmethod
    def forward(ctx, x, weight, fp8_backward=False):
        ctx.save_for_backward(x, weight)
        ctx.fp8_backward = fp8_backward
        return fp8_matmul(x, weight)

    @staticmethod
    def backward(ctx, grad):
        x, weight = ctx.saved_tensors
        grad = grad.to(torch.bfloat16).contiguous()
        if not ctx.fp8_backward:
            return (grad @ weight.to(torch.bfloat16),
                    grad.t() @ x.to(torch.bfloat16), None)
        quantize = _compiled_quantize()

        # dgrad = grad @ weight, reducing over N.
        grad8, grad_scale = quantize(grad)
        weight_t = weight.to(torch.bfloat16).t().contiguous()   # [K, N]
        wt8, wt_scale = quantize(weight_t)
        dx = torch._scaled_mm(grad8, wt8.t(), scale_a=grad_scale,
                              scale_b=wt_scale.t(), out_dtype=torch.bfloat16)

        # wgrad = grad.T @ x, reducing over M (the token axis).
        grad_t8, grad_t_scale = quantize(grad.t().contiguous())
        x_t8, x_t_scale = quantize(x.to(torch.bfloat16).t().contiguous())
        dw = torch._scaled_mm(grad_t8, x_t8.t(), scale_a=grad_t_scale,
                              scale_b=x_t_scale.t(), out_dtype=torch.bfloat16)

        return dx, dw, None
