"""Finds ``residual + gelu(x + bias, approximate="tanh")`` inside a captured
FX graph and rewrites it into a single call to
``torch.ops.graphfuse.fused_bias_gelu_residual``.

This runs on the *raw* graph torch.compile hands to a custom backend, before
AOTAutograd decomposes anything to ATen ops, so node targets are still the
ordinary Python-level callables (``operator.add``, ``torch.nn.functional.gelu``)
a user's model actually calls, not their ``aten::`` equivalents. That is a
deliberate scope limit, not an oversight: a bias fused directly into a
preceding ``nn.Linear`` (``F.linear`` with ``bias=True``), or a GELU written
by hand instead of via ``F.gelu``, would not appear in this exact shape and
this pass would correctly leave it alone rather than guess.
"""

from __future__ import annotations

import operator

import torch
import torch.fx as fx
import torch.nn.functional as F

_ADD_TARGETS = (operator.add, torch.add)


def _is_add(node: fx.Node) -> bool:
    return node.op == "call_function" and node.target in _ADD_TARGETS and len(node.args) == 2


def _is_tanh_gelu(node: fx.Node) -> bool:
    if node.op != "call_function" or node.target is not F.gelu:
        return False
    approximate = node.kwargs.get("approximate")
    if approximate is None and len(node.args) >= 2:
        approximate = node.args[1]
    return approximate == "tanh"


def _rank(node: fx.Node) -> int | None:
    example = node.meta.get("example_value")
    if example is None:
        return None
    return example.dim()


def _split_bias_and_x(a: fx.Node, b: fx.Node) -> tuple[fx.Node, fx.Node] | None:
    """Returns (x_node, bias_node), using rank to tell the broadcast bias
    apart from the full-rank activation. Refuses to guess if either operand
    isn't a graph node (a literal constant) or ranks are ambiguous.
    """
    if not (isinstance(a, fx.Node) and isinstance(b, fx.Node)):
        return None
    rank_a, rank_b = _rank(a), _rank(b)
    if rank_a is None or rank_b is None or rank_a == rank_b:
        return None
    return (a, b) if rank_a > rank_b else (b, a)


def find_and_fuse_bias_gelu_residual(gm: fx.GraphModule) -> int:
    """Mutates ``gm.graph`` in place. Returns the number of fusions applied."""
    graph = gm.graph
    fused = 0

    for add2 in list(graph.nodes):
        if not _is_add(add2):
            continue

        arg0, arg1 = add2.args
        gelu_node = None
        residual_node = None
        for candidate, other in ((arg0, arg1), (arg1, arg0)):
            if isinstance(candidate, fx.Node) and _is_tanh_gelu(candidate) and len(candidate.users) == 1:
                gelu_node, residual_node = candidate, other
                break
        if gelu_node is None or not isinstance(residual_node, fx.Node):
            continue

        add1 = gelu_node.args[0]
        if not (isinstance(add1, fx.Node) and _is_add(add1) and len(add1.users) == 1):
            continue

        split = _split_bias_and_x(*add1.args)
        if split is None:
            continue
        x_node, bias_node = split

        with graph.inserting_before(add2):
            fused_node = graph.call_function(
                torch.ops.graphfuse.fused_bias_gelu_residual.default,
                args=(x_node, bias_node, residual_node),
            )
        add2.replace_all_uses_with(fused_node)
        graph.erase_node(add2)
        graph.erase_node(gelu_node)
        graph.erase_node(add1)
        fused += 1

    if fused:
        graph.lint()
        gm.recompile()
    return fused
