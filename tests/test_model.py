from __future__ import annotations

import torch

from graphfuse.model import FusibleBlock, FusibleStack


def test_block_preserves_shape_and_actually_uses_the_residual():
    torch.manual_seed(0)
    block = FusibleBlock(hidden=16)
    x = torch.randn(5, 16)
    out = block(x)
    assert out.shape == x.shape
    with torch.no_grad():
        block.proj.weight.zero_()
        block.bias.zero_()
    zeroed_out = block(x)
    torch.testing.assert_close(zeroed_out, x + torch.nn.functional.gelu(torch.zeros_like(x), approximate="tanh"))


def test_stack_chains_blocks_and_preserves_shape():
    torch.manual_seed(0)
    stack = FusibleStack(hidden=32, depth=5)
    assert len(stack.blocks) == 5
    x = torch.randn(3, 32)
    out = stack(x)
    assert out.shape == x.shape


def test_gradients_reach_every_block():
    stack = FusibleStack(hidden=8, depth=4)
    x = torch.randn(2, 8, requires_grad=True)
    out = stack(x)
    out.sum().backward()
    assert x.grad is not None
    for block in stack.blocks:
        assert block.proj.weight.grad is not None
        assert block.bias.grad is not None
