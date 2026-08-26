"""A plain-PyTorch implementation of the exact epilogue graphfuse fuses:
``residual + gelu(x + bias)``, using the tanh approximation of GELU (the same
approximation GPT-2/nanoGPT-style blocks use, not the erf-based exact form).
This is both the gradcheck ground truth for the kernel and the shape the FX
pattern matcher in ``pattern.py`` looks for.
"""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor


def bias_gelu_residual_reference(x: Tensor, bias: Tensor, residual: Tensor) -> Tensor:
    return F.gelu(x + bias, approximate="tanh") + residual
