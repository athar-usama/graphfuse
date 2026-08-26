"""Forward and backward Triton kernels for ``residual + gelu(x + bias)``,
tanh-approximation GELU, bias broadcast along the last dimension.

Both kernels are fully elementwise over a flattened ``(rows * hidden,)`` view;
the only cross-row interaction is ``dbias``, which every row contributes to,
so the backward kernel accumulates it with ``tl.atomic_add`` exactly the way
sparseflash's backward kernels accumulate the shared ``dK``/``dV`` columns:
many blocks write overlapping locations, so atomics are correctness, not an
optimization.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

_SQRT_2_OVER_PI = tl.constexpr(0.7978845608028654)
_GELU_COEFF = tl.constexpr(0.044715)
_GELU_COEFF3 = tl.constexpr(3.0 * 0.044715)

_BLOCK_SIZE = 1024


@triton.jit
def _tanh(u):
    return 1.0 - 2.0 / (tl.exp(2.0 * u) + 1.0)


@triton.jit
def _fwd_kernel(
    x_ptr, bias_ptr, residual_ptr, out_ptr,
    numel, hidden,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    col = offsets % hidden

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + col, mask=mask, other=0.0).to(tl.float32)
    residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    t = x + bias
    u = _SQRT_2_OVER_PI * (t + _GELU_COEFF * t * t * t)
    gelu = 0.5 * t * (1.0 + _tanh(u))
    out = gelu + residual

    tl.store(out_ptr + offsets, out.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _bwd_kernel(
    x_ptr, bias_ptr, grad_out_ptr,
    dx_ptr, dbias_ptr,
    numel, hidden,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    col = offsets % hidden

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + col, mask=mask, other=0.0).to(tl.float32)
    grad_out = tl.load(grad_out_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    t = x + bias
    t2 = t * t
    u = _SQRT_2_OVER_PI * (t + _GELU_COEFF * t2 * t)
    tanh_u = _tanh(u)
    du_dt = _SQRT_2_OVER_PI * (1.0 + _GELU_COEFF3 * t2)
    gelu_grad = 0.5 * (1.0 + tanh_u) + 0.5 * t * (1.0 - tanh_u * tanh_u) * du_dt

    dx = grad_out * gelu_grad

    tl.store(dx_ptr + offsets, dx.to(dx_ptr.dtype.element_ty), mask=mask)
    tl.atomic_add(dbias_ptr + col, dx, mask=mask)


def fused_bias_gelu_residual_forward(x: torch.Tensor, bias: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
    assert x.shape == residual.shape
    assert bias.shape == (x.shape[-1],)
    x = x.contiguous()
    bias = bias.contiguous()
    residual = residual.contiguous()

    out = torch.empty_like(x)
    numel = x.numel()
    hidden = x.shape[-1]
    grid = (triton.cdiv(numel, _BLOCK_SIZE),)
    _fwd_kernel[grid](x, bias, residual, out, numel, hidden, BLOCK_SIZE=_BLOCK_SIZE)
    return out


def fused_bias_gelu_residual_backward(
    x: torch.Tensor, bias: torch.Tensor, grad_output: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    x = x.contiguous()
    bias = bias.contiguous()
    grad_output = grad_output.contiguous()

    dx = torch.empty_like(x)
    dbias = torch.zeros_like(bias, dtype=torch.float32)
    numel = x.numel()
    hidden = x.shape[-1]
    grid = (triton.cdiv(numel, _BLOCK_SIZE),)
    _bwd_kernel[grid](x, bias, grad_output, dx, dbias, numel, hidden, BLOCK_SIZE=_BLOCK_SIZE)
    return dx, dbias.to(bias.dtype)
