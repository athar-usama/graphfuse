"""Tier 2: the registered custom op's Triton forward and backward, compared
directly against the already-gradchecked reference in ``test_reference.py``.
No finite differences here, on purpose: gradchecking the kernel's own output
directly would mostly measure floating-point rounding against a probe that
assumes double precision, which this kernel does not compute in.
"""

from __future__ import annotations

import pytest
import torch

import graphfuse.ops  # noqa: F401  registers torch.ops.graphfuse.*
from graphfuse.reference import bias_gelu_residual_reference

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU")

CONFIGS = [(64, 128), (37, 300), (1, 8), (500, 17)]


@pytest.mark.parametrize("rows,hidden", CONFIGS)
def test_forward_matches_reference(rows, hidden):
    torch.manual_seed(0)
    x = torch.randn(rows, hidden, device="cuda", dtype=torch.float32)
    bias = torch.randn(hidden, device="cuda", dtype=torch.float32)
    residual = torch.randn(rows, hidden, device="cuda", dtype=torch.float32)

    expected = bias_gelu_residual_reference(x, bias, residual)
    actual = torch.ops.graphfuse.fused_bias_gelu_residual(x, bias, residual)
    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("rows,hidden", CONFIGS)
def test_backward_matches_reference(rows, hidden):
    torch.manual_seed(0)
    x_ref = torch.randn(rows, hidden, device="cuda", dtype=torch.float32, requires_grad=True)
    bias_ref = torch.randn(hidden, device="cuda", dtype=torch.float32, requires_grad=True)
    residual_ref = torch.randn(rows, hidden, device="cuda", dtype=torch.float32, requires_grad=True)

    x_kernel = x_ref.detach().clone().requires_grad_()
    bias_kernel = bias_ref.detach().clone().requires_grad_()
    residual_kernel = residual_ref.detach().clone().requires_grad_()

    out_ref = bias_gelu_residual_reference(x_ref, bias_ref, residual_ref)
    out_kernel = torch.ops.graphfuse.fused_bias_gelu_residual(x_kernel, bias_kernel, residual_kernel)
    torch.testing.assert_close(out_kernel, out_ref, atol=1e-4, rtol=1e-4)

    grad_out = torch.randn_like(out_ref)
    out_ref.backward(grad_out)
    out_kernel.backward(grad_out.clone())

    torch.testing.assert_close(x_kernel.grad, x_ref.grad, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(bias_kernel.grad, bias_ref.grad, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(residual_kernel.grad, residual_ref.grad, atol=1e-3, rtol=1e-3)
