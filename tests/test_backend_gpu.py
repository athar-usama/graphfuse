"""End-to-end: torch.compile(model, backend=graphfuse_backend) against the
real Dynamo/AOTAutograd/Inductor pipeline, not a hand-built graph. Checks
both that the fusion actually fires once per block and that the compiled
model's forward and backward numerically match eager, params included.
"""

from __future__ import annotations

import pytest
import torch

from graphfuse.backend import graphfuse_backend
from graphfuse.model import FusibleStack

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU")


def test_compiled_output_matches_eager_and_fuses_every_block():
    torch.manual_seed(0)
    depth = 4
    model = FusibleStack(hidden=128, depth=depth).cuda()
    x = torch.randn(8, 128, device="cuda")

    expected = model(x)

    torch._dynamo.reset()
    graphfuse_backend.last_fusion_count = 0
    compiled = torch.compile(model, backend=graphfuse_backend, fullgraph=True)
    actual = compiled(x)

    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)
    assert graphfuse_backend.last_fusion_count == depth


def test_gradients_flow_through_the_compiled_model():
    torch.manual_seed(0)
    model_eager = FusibleStack(hidden=64, depth=3).cuda()
    model_compiled = FusibleStack(hidden=64, depth=3).cuda()
    model_compiled.load_state_dict(model_eager.state_dict())

    x_eager = torch.randn(6, 64, device="cuda", requires_grad=True)
    x_compiled = x_eager.detach().clone().requires_grad_()

    out_eager = model_eager(x_eager)

    torch._dynamo.reset()
    out_compiled = torch.compile(model_compiled, backend=graphfuse_backend, fullgraph=True)(x_compiled)

    grad_out = torch.randn_like(out_eager)
    out_eager.backward(grad_out)
    out_compiled.backward(grad_out.clone())

    torch.testing.assert_close(out_compiled, out_eager, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(x_compiled.grad, x_eager.grad, atol=1e-3, rtol=1e-3)
    for (_, p_eager), (_, p_compiled) in zip(
        model_eager.named_parameters(), model_compiled.named_parameters(), strict=True
    ):
        assert p_eager.grad is not None
        torch.testing.assert_close(p_compiled.grad, p_eager.grad, atol=1e-3, rtol=1e-3)
