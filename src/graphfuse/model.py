"""A small stack of residual blocks shaped so the ``bias-add -> gelu ->
residual-add`` epilogue this project fuses appears in the traced graph
exactly once per block, unforced. Each block's ``Linear`` carries no bias of
its own (``bias=False``); the bias is a separate parameter, added explicitly
after the matmul, which is what keeps it as its own graph node instead of
being folded into the matmul the way ``nn.Linear(bias=True)`` folds it.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class FusibleBlock(nn.Module):
    """``out = x + gelu(x @ W^T + bias, approximate="tanh")``.

    This is the shape of a single linear-plus-activation residual sub-block:
    the same shape an adapter/LoRA residual branch has on its own, and the
    shape one half of a Macaron/sandwich-style transformer MLP has when its
    residual wraps a single linear rather than the usual two.
    """

    def __init__(self, hidden: int):
        super().__init__()
        self.proj = nn.Linear(hidden, hidden, bias=False)
        self.bias = nn.Parameter(torch.zeros(hidden))

    def forward(self, x: Tensor) -> Tensor:
        h = self.proj(x)
        return x + F.gelu(h + self.bias, approximate="tanh")


class FusibleStack(nn.Module):
    def __init__(self, hidden: int, depth: int):
        super().__init__()
        self.blocks = nn.ModuleList(FusibleBlock(hidden) for _ in range(depth))

    def forward(self, x: Tensor) -> Tensor:
        for block in self.blocks:
            x = block(x)
        return x
