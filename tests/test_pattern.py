"""Unit tests for the FX rewrite in isolation: hand-built graphs with the
right node shapes and ``meta["example_value"]`` entries (what Dynamo's real
fake-tensor propagation would have set), no torch.compile, no GPU, no Triton.
This is deliberately its own tier below the end-to-end GPU backend test in
``test_backend_gpu.py``: it proves the *matcher* is correct independent of
whether the fused op executes correctly, the same separation of concerns
sparseflash's tiered correctness tests use.
"""

from __future__ import annotations

import operator

import torch
import torch.fx as fx
import torch.nn.functional as F

import graphfuse.ops  # noqa: F401  registers torch.ops.graphfuse.*
from graphfuse.pattern import find_and_fuse_bias_gelu_residual

FUSED_TARGET = torch.ops.graphfuse.fused_bias_gelu_residual.default


def _build_graph(*, bias_first: bool, residual_first: bool, approximate: str = "tanh", shared_add1: bool = False):
    graph = fx.Graph()
    x = graph.placeholder("x")
    x.meta["example_value"] = torch.empty(4, 8)
    bias = graph.placeholder("bias")
    bias.meta["example_value"] = torch.empty(8)
    residual = graph.placeholder("residual")
    residual.meta["example_value"] = torch.empty(4, 8)

    add1_args = (bias, x) if bias_first else (x, bias)
    add1 = graph.call_function(operator.add, args=add1_args)
    add1.meta["example_value"] = torch.empty(4, 8)

    gelu = graph.call_function(F.gelu, args=(add1,), kwargs={"approximate": approximate})
    gelu.meta["example_value"] = torch.empty(4, 8)

    add2_args = (residual, gelu) if residual_first else (gelu, residual)
    add2 = graph.call_function(operator.add, args=add2_args)
    add2.meta["example_value"] = torch.empty(4, 8)

    outputs = [add2]
    if shared_add1:
        extra = graph.call_function(torch.sin, args=(add1,))
        outputs.append(extra)

    graph.output(tuple(outputs) if len(outputs) > 1 else outputs[0])
    gm = fx.GraphModule(torch.nn.Module(), graph)
    return gm, x, bias, residual


def test_fuses_the_straightforward_ordering():
    gm, *_ = _build_graph(bias_first=False, residual_first=False)
    assert find_and_fuse_bias_gelu_residual(gm) == 1
    targets = [n.target for n in gm.graph.nodes if n.op == "call_function"]
    assert targets == [FUSED_TARGET]


def test_fuses_regardless_of_commutative_argument_order():
    gm, *_ = _build_graph(bias_first=True, residual_first=True)
    assert find_and_fuse_bias_gelu_residual(gm) == 1
    targets = [n.target for n in gm.graph.nodes if n.op == "call_function"]
    assert targets == [FUSED_TARGET]


def test_leaves_exact_gelu_alone():
    gm, *_ = _build_graph(bias_first=False, residual_first=False, approximate="none")
    assert find_and_fuse_bias_gelu_residual(gm) == 0


def test_leaves_shared_intermediate_alone():
    gm, *_ = _build_graph(bias_first=False, residual_first=False, shared_add1=True)
    assert find_and_fuse_bias_gelu_residual(gm) == 0


def test_refuses_to_guess_when_ranks_are_ambiguous():
    graph = fx.Graph()
    x = graph.placeholder("x")
    x.meta["example_value"] = torch.empty(4, 8)
    not_bias = graph.placeholder("not_bias")
    not_bias.meta["example_value"] = torch.empty(4, 8)
    residual = graph.placeholder("residual")
    residual.meta["example_value"] = torch.empty(4, 8)

    add1 = graph.call_function(operator.add, args=(x, not_bias))
    add1.meta["example_value"] = torch.empty(4, 8)
    gelu = graph.call_function(F.gelu, args=(add1,), kwargs={"approximate": "tanh"})
    gelu.meta["example_value"] = torch.empty(4, 8)
    add2 = graph.call_function(operator.add, args=(gelu, residual))
    add2.meta["example_value"] = torch.empty(4, 8)
    graph.output(add2)
    gm = fx.GraphModule(torch.nn.Module(), graph)

    assert find_and_fuse_bias_gelu_residual(gm) == 0
