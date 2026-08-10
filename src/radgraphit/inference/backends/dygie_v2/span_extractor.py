"""Endpoint span extractor — faithful port of ``training_v2/src/dygie/span_extractor.py``.

concat(start emb, end emb, [mean-pooled interior emb if ``mean_pool``], width emb). The
``width_embedding`` submodule name is preserved for state-dict compatibility. Inference runs the
batch_size=1 path (no batch dim, no span masking), exactly as upstream.
"""

from __future__ import annotations

import torch
from torch import nn


class EndpointSpanExtractor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_width_embeddings: int,
        span_width_embedding_dim: int,
        mean_pool: bool = False,
    ):
        super().__init__()
        self.width_embedding = nn.Embedding(num_width_embeddings, span_width_embedding_dim)
        self.mean_pool = mean_pool
        self.output_dim = (3 if mean_pool else 2) * input_dim + span_width_embedding_dim

    def get_output_dim(self) -> int:
        return self.output_dim

    def forward(self, word_embeddings: torch.Tensor, spans: torch.LongTensor) -> torch.Tensor:
        """word_embeddings: (num_words, input_dim). spans: (num_spans, 2) inclusive [start, end]."""
        starts, ends = spans[:, 0], spans[:, 1]
        start_emb = word_embeddings[starts]
        end_emb = word_embeddings[ends]
        width_emb = self.width_embedding(ends - starts)  # 0-indexed width (widths 1..N -> 0..N-1)
        if not self.mean_pool:
            return torch.cat([start_emb, end_emb, width_emb], dim=-1)
        mean_emb = self._interior_mean(word_embeddings, starts, ends)
        return torch.cat([start_emb, end_emb, mean_emb, width_emb], dim=-1)

    @staticmethod
    def _interior_mean(
        word_embeddings: torch.Tensor, starts: torch.Tensor, ends: torch.Tensor
    ) -> torch.Tensor:
        widths = ends - starts + 1  # (num_spans,)
        max_width = int(widths.max().item())
        offsets = torch.arange(max_width, device=word_embeddings.device).unsqueeze(0)  # (1, W)
        mask = offsets < widths.unsqueeze(1)  # (num_spans, W)
        idx = (starts.unsqueeze(1) + offsets).clamp(max=word_embeddings.size(0) - 1)
        gathered = word_embeddings[idx] * mask.unsqueeze(-1)  # (num_spans, W, input_dim)
        return gathered.sum(dim=1) / widths.unsqueeze(1).to(word_embeddings.dtype)
