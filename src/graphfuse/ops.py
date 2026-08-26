"""Registers ``fused_bias_gelu_residual`` as a real PyTorch custom op via
``torch.library.custom_op``, not a bare Python function. That distinction
matters for this project specifically: the FX pattern matcher rewrites a
matched subgraph into a call to this op, and that rewritten graph gets handed
to Inductor to compile the rest of the model. A plain Python callable
embedded in an FX graph is opaque to Inductor in the wrong way, either
breaking tracing or forcing a graph break; a registered custom op comes with
a fake-tensor shape rule and an autograd formula, so Dynamo, AOTAutograd, and
Inductor all know exactly what it does without needing to look inside it.

The backward gets the same treatment, as its own custom op, not as a plain
Python function called from ``register_autograd``'s backward callable.
AOTAutograd traces that callable to build the backward graph, so if it calls
straight into a Triton kernel launch, tracing hits a FakeTensor/FunctionalTensor
with no real data pointer and fails outright: "please wrap the custom kernel
into an opaque custom op" is PyTorch's own error message for exactly this
mistake, hit directly while wiring this up.
"""

from __future__ import annotations

import torch
from torch import Tensor

from .kernels._fused_bias_gelu_residual import (
    fused_bias_gelu_residual_backward,
    fused_bias_gelu_residual_forward,
)


@torch.library.custom_op("graphfuse::fused_bias_gelu_residual", mutates_args=())
def fused_bias_gelu_residual(x: Tensor, bias: Tensor, residual: Tensor) -> Tensor:
    return fused_bias_gelu_residual_forward(x, bias, residual)


@fused_bias_gelu_residual.register_fake
def _fused_bias_gelu_residual_fake(x: Tensor, bias: Tensor, residual: Tensor) -> Tensor:
    return torch.empty_like(x)


@torch.library.custom_op("graphfuse::fused_bias_gelu_residual_backward", mutates_args=())
def _fused_bias_gelu_residual_backward_op(x: Tensor, bias: Tensor, grad_output: Tensor) -> tuple[Tensor, Tensor]:
    return fused_bias_gelu_residual_backward(x, bias, grad_output)


@_fused_bias_gelu_residual_backward_op.register_fake
def _fused_bias_gelu_residual_backward_op_fake(
    x: Tensor, bias: Tensor, grad_output: Tensor
) -> tuple[Tensor, Tensor]:
    return torch.empty_like(x), torch.empty_like(bias)


def _setup_context(ctx, inputs, output) -> None:
    x, bias, _residual = inputs
    ctx.save_for_backward(x, bias)


def _backward(ctx, grad_output: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    x, bias = ctx.saved_tensors
    dx, dbias = _fused_bias_gelu_residual_backward_op(x, bias, grad_output)
    dresidual = grad_output
    return dx, dbias, dresidual


fused_bias_gelu_residual.register_autograd(_backward, setup_context=_setup_context)
