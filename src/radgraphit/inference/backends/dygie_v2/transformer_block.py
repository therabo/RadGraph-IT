"""Span-attention scorer body — faithful port of ``training_v2/src/dygie/transformer_block.py``.

Used by the NER / relation heads in the ``medbit_transformer`` arm in place of the plain
``FeedForward`` scorer. Submodule names (``input_proj``, ``encoder``) are preserved for state-dict
compatibility. Spans are the "sequence"; a sinusoidal positional encoding keyed on each span's
start-word index gives attention a notion of span order/distance.
"""

from __future__ import annotations

import math

import torch
from torch import nn


def sinusoidal_position_encoding(positions: torch.Tensor, d_model: int) -> torch.Tensor:
    """positions: (N,) int64 span start indices. Returns (N, d_model)."""
    div_term = torch.exp(
        torch.arange(0, d_model, 2, device=positions.device, dtype=torch.float32)
        * (-math.log(10000.0) / d_model)
    )
    angles = positions.unsqueeze(1).to(torch.float32) * div_term.unsqueeze(0)  # (N, d_model/2)
    pe = torch.zeros(positions.size(0), d_model, device=positions.device)
    pe[:, 0::2] = torch.sin(angles)
    pe[:, 1::2] = torch.cos(angles[:, : pe[:, 1::2].size(1)])
    return pe


class TransformerBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        activation: str = "gelu",
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model) if input_dim != d_model else nn.Identity()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            norm_first=True,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=num_layers, enable_nested_tensor=False
        )
        self.d_model = d_model
        self.output_dim = d_model

    def forward(self, span_embeddings: torch.Tensor, start_positions: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(span_embeddings) + sinusoidal_position_encoding(
            start_positions, self.d_model
        )
        return self.encoder(x.unsqueeze(0)).squeeze(0)
