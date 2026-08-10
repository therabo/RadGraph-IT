"""Mention pruner — faithful port of ``training_v2/src/dygie/pruner.py``.

Score every span, keep the top-k in original span order. The ``scorer`` submodule name is preserved
for state-dict compatibility. batch_size=1 path: no masking, no batch dimension.
"""

from __future__ import annotations

import torch
from torch import nn


class Pruner(nn.Module):
    def __init__(self, scorer: nn.Module):
        super().__init__()
        self.scorer = scorer  # (num_spans, dim) -> (num_spans, 1)

    def forward(
        self, span_embeddings: torch.Tensor, num_items_to_keep: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        scores = self.scorer(span_embeddings).squeeze(-1)  # (num_spans,)
        k = max(1, min(num_items_to_keep, span_embeddings.size(0)))
        _, top_indices = scores.topk(k)
        top_indices, _ = torch.sort(top_indices)
        top_scores = scores[top_indices]
        top_embeddings = span_embeddings[top_indices]
        return top_embeddings, top_indices, top_scores
