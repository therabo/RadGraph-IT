"""Shared MLP scorer body — faithful port of ``training_v2/src/dygie/feedforward.py``.

Structure (the ``.net`` Sequential and its layer indices) is preserved exactly so the parameter
names in the state dict match the checkpoint.
"""

from __future__ import annotations

import torch
from torch import nn


class FeedForward(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], dropout: float):
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for dim in hidden_dims:
            layers += [nn.Linear(prev, dim), nn.ReLU(), nn.Dropout(dropout)]
            prev = dim
        self.net = nn.Sequential(*layers)
        self.output_dim = prev

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
