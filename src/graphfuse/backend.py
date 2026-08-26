"""The actual custom torch.compile backend: run the FX rewrite in
``pattern.py`` against the raw Dynamo graph, then hand the *rewritten* graph
to Inductor's own ``compile_fx`` for everything else. This is deliberately
not a replacement compiler: graphfuse owns exactly the three nodes it
recognizes, and Inductor still owns layout, memory planning, remaining
fusions, and codegen for the rest of the model. ``compile_fx`` is the same
function ``torch.compile(model)`` (``backend="inductor"``) calls internally.
"""

from __future__ import annotations

import torch.fx as fx
from torch._inductor.compile_fx import compile_fx

from . import ops  # noqa: F401  side effect: registers torch.ops.graphfuse.*
from .pattern import find_and_fuse_bias_gelu_residual


def graphfuse_backend(gm: fx.GraphModule, example_inputs: list):
    num_fused = find_and_fuse_bias_gelu_residual(gm)
    graphfuse_backend.last_fusion_count = num_fused
    return compile_fx(gm, example_inputs)


graphfuse_backend.last_fusion_count = 0
