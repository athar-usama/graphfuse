from __future__ import annotations

import torch
from torch.autograd import gradcheck

from graphfuse.reference import bias_gelu_residual_reference


def test_output_shape_and_dtype():
    x = torch.randn(4, 8)
    bias = torch.randn(8)
    residual = torch.randn(4, 8)
    out = bias_gelu_residual_reference(x, bias, residual)
    assert out.shape == x.shape
    assert out.dtype == x.dtype


def test_bias_and_residual_both_actually_change_the_output():
    torch.manual_seed(0)
    x = torch.randn(4, 8)
    residual = torch.randn(4, 8)
    zero_bias = torch.zeros(8)
    real_bias = torch.randn(8)
    out_zero = bias_gelu_residual_reference(x, zero_bias, residual)
    out_real = bias_gelu_residual_reference(x, real_bias, residual)
    assert not torch.allclose(out_zero, out_real)
    assert not torch.allclose(out_real - residual, out_real)


def test_gradcheck_against_finite_differences():
    torch.manual_seed(0)
    x = torch.randn(4, 8, dtype=torch.float64, requires_grad=True)
    bias = torch.randn(8, dtype=torch.float64, requires_grad=True)
    residual = torch.randn(4, 8, dtype=torch.float64, requires_grad=True)
    assert gradcheck(bias_gelu_residual_reference, (x, bias, residual), eps=1e-6, atol=1e-5)
