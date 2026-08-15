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

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.normal_(self.weight, std=0.02)
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        out = _FP8Function.apply(x.reshape(-1, shape[-1]), self.weight)
        out = out.reshape(*shape[:-1], self.out_features)
        return out if self.bias is None else out + self.bias


class _FP8Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        ctx.save_for_backward(x, weight)
        return fp8_matmul(x, weight)

    @staticmethod
    def backward(ctx, grad):
        x, weight = ctx.saved_tensors
        grad = grad.to(torch.bfloat16)
        return (grad @ weight.to(torch.bfloat16),
                grad.t() @ x.to(torch.bfloat16))
