"""A drop-in NVFP4 Linear, and the quantization it needs.

Scope: this is a THROUGHPUT PROBE, not a training recipe. It quantizes the
forward GEMM only and keeps the backward in bf16 by using a straight-through
estimator. Published NVFP4 pretraining recipes additionally need random
Hadamard transforms on gradients, stochastic rounding, and bf16 first/last
blocks; none of that is here. Use it to answer "how much wall clock could FP4
buy at these shapes", then decide whether the real recipe is worth writing.

The kernel's scale layout is not the obvious one. Scales are per 16 elements
along K, but the tensor must be flat and padded: rows to a multiple of 128,
block count to a multiple of 4. An unpadded [rows, K/16] fails with "Invalid
scaling configuration".
"""
from __future__ import annotations

import torch
import torch.nn as nn

# e2m1 spans +/-6 with 3 mantissa steps per binade; 6.0 is the max magnitude.
FP4_MAX = 6.0
# e4m3 block scales saturate at 448.
E4M3_MAX = 448.0
BLOCK = 16


def _pad(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def quantize_nvfp4(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """bf16/fp32 [rows, K] -> (packed fp4, swizzled e4m3 block scales, global).

    Two-level scaling, as NVFP4 specifies: a per-tensor fp32 scale brings the
    tensor into e4m3's range, then a per-16-element e4m3 scale captures local
    dynamic range. The product is what the kernel multiplies back in.
    """
    rows, k = x.shape
    assert k % BLOCK == 0, f"K={k} must be a multiple of {BLOCK}"

    blocks = x.abs().reshape(rows, k // BLOCK, BLOCK).amax(dim=-1)
    # Per-tensor scale so the largest block scale lands at e4m3's ceiling.
    global_scale = (blocks.max() / (FP4_MAX * E4M3_MAX)).clamp(min=1e-12)
    block_scales = (blocks / (FP4_MAX * global_scale)).clamp(min=1e-12, max=E4M3_MAX)
    block_e4m3 = block_scales.to(torch.float8_e4m3fn)

    # Dequantized scale, so quantization matches what the kernel will apply.
    effective = block_e4m3.to(torch.float32) * global_scale
    scaled = x.reshape(rows, k // BLOCK, BLOCK) / effective.unsqueeze(-1).clamp(min=1e-12)
    packed = _pack_e2m1(scaled.reshape(rows, k).clamp(-FP4_MAX, FP4_MAX))

    return packed, swizzle_scales(block_e4m3), global_scale


def swizzle_scales(block_e4m3: torch.Tensor) -> torch.Tensor:
    """[rows, nblocks] -> the kernel's 128x4 swizzled, flat scale tensor.

    Determined empirically by poking single scale elements and seeing which
    output row moved: row r block b lands at

        (r % 32) * 16 + (r // 32 % 4) * 4 + (b % 4)

    within each [128 rows x 4 blocks] tile. A plain row-major fill silently
    produces wrong numerics rather than an error -- it cost a 50% relative
    error before this was tracked down.
    """
    rows, nblocks = block_e4m3.shape
    padded_rows, padded_blocks = _pad(rows, 128), _pad(nblocks, 4)

    full = torch.zeros(padded_rows, padded_blocks, device=block_e4m3.device,
                       dtype=block_e4m3.dtype)
    full[:rows, :nblocks] = block_e4m3

    tiles = full.reshape(padded_rows // 128, 128, padded_blocks // 4, 4)
    # (row_tile, block_tile, row%32, row//32, block) -> flat
    tiles = tiles.permute(0, 2, 1, 3).reshape(
        padded_rows // 128, padded_blocks // 4, 4, 32, 4)
    tiles = tiles.permute(0, 1, 3, 2, 4)
    return tiles.reshape(-1).contiguous()


# e2m1 representable magnitudes; index is the 3-bit magnitude code.
_E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
# Midpoints between consecutive levels: round-to-nearest becomes a bucketize.
_E2M1_EDGES = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])


def _pack_e2m1(x: torch.Tensor) -> torch.Tensor:
    """Round to nearest e2m1 level and pack two values per byte.

    Uses a bucketize against the 7 midpoints rather than an argmin over the 8
    levels. The argmin form materializes an [M, K, 8] tensor, which dominated
    runtime: quantizing one activation cost 16x the matmul it was feeding.
    """
    edges = _E2M1_EDGES.to(x.device, x.dtype)
    code = torch.bucketize(x.abs(), edges).to(torch.uint8)
    code = code | ((x < 0).to(torch.uint8) << 3)
    low, high = code[:, 0::2], code[:, 1::2]
    return (low | (high << 4)).contiguous().view(torch.float4_e2m1fn_x2)


_QUANTIZE: object | None = None


def _compiled_quantize():
    """Compile once, lazily. Eager quantization costs 16x the matmul it feeds;
    compiled it is ~9x cheaper, which is the difference between 'hopeless' and
    'break-even'."""
    global _QUANTIZE
    if _QUANTIZE is None:
        _QUANTIZE = torch.compile(quantize_nvfp4, dynamic=False)
    return _QUANTIZE


def nvfp4_matmul(a: torch.Tensor, weight: torch.Tensor,
                 compiled: bool = True) -> torch.Tensor:
    """a [M, K] bf16 @ weight.T, with weight [N, K] bf16, via the FP4 kernel."""
    quantize = _compiled_quantize() if compiled else quantize_nvfp4
    a4, sa, ga = quantize(a)
    w4, sw, gw = quantize(weight)
    out = torch._scaled_mm(a4, w4.t(), scale_a=sa, scale_b=sw,
                           out_dtype=torch.bfloat16)
    return out * (ga * gw)


class NVFP4Linear(nn.Module):
    """nn.Linear whose forward GEMM runs in NVFP4.

    Backward is a straight-through bf16 matmul: gradients flow as if the
    forward had been bf16. That is deliberately not a real FP4 training recipe
    -- it isolates forward-GEMM throughput without the numerics work.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.normal_(self.weight, std=0.02)
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        out = _NVFP4Function.apply(flat, self.weight)
        out = out.reshape(*shape[:-1], self.out_features)
        return out if self.bias is None else out + self.bias


class _NVFP4Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        ctx.save_for_backward(x, weight)
        return nvfp4_matmul(x, weight)

    @staticmethod
    def backward(ctx, grad):
        x, weight = ctx.saved_tensors
        # Under autocast the incoming grad can be fp32 while the saved
        # activations are bf16; match both operands to the grad's precision.
        grad = grad.to(torch.bfloat16)
        return (grad @ weight.to(torch.bfloat16),
                grad.t() @ x.to(torch.bfloat16))
