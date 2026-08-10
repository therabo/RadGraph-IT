"""Relation head — inference-only port of ``training_v2/src/dygie/relation_head.py``.

Prune to top-K spans, score every ordered pair, decode. The parameter-bearing structure
(``pruners`` / ``span_transformers`` / ``relation_feedforwards`` / ``relation_scorers`` ModuleDicts,
the null-column trick, and the optional ``span_pooling`` / ``transformer_params`` /
``relation_context`` / ``relation_feedforward_params`` variants) is preserved exactly so the state
dict matches the checkpoint. Loss and metrics are dropped; decode returns plain
``(s1, e1, s2, e2, label, raw_score, softmax_score)`` tuples.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from .feedforward import FeedForward
from .pruner import Pruner
from .transformer_block import TransformerBlock
from .vocab import Vocabulary

RelationTuple = tuple[int, int, int, int, str, float, float]


class RelationHead(nn.Module):
    def __init__(
        self,
        vocab: Vocabulary,
        span_emb_dim: int,
        word_emb_dim: int,
        feedforward_params: dict,
        spans_per_word: float,
        transformer_params: dict | None = None,
        use_between_context: bool = False,
        relation_feedforward_params: dict | None = None,
    ):
        super().__init__()
        self.vocab = vocab
        self.spans_per_word = spans_per_word
        self.namespaces = [ns for ns in vocab.get_namespaces() if ns.endswith("relation_labels")]
        self.n_labels = {ns: vocab.get_vocab_size(ns) for ns in self.namespaces}
        self.use_transformer = transformer_params is not None
        self.use_between_context = use_between_context
        between_dim = word_emb_dim if use_between_context else 0
        relation_feedforward_params = relation_feedforward_params or feedforward_params

        self.pruners = nn.ModuleDict()
        self.span_transformers = nn.ModuleDict()
        self.relation_feedforwards = nn.ModuleDict()
        self.relation_scorers = nn.ModuleDict()

        for ns in self.namespaces:
            mention_ff = FeedForward(
                span_emb_dim, feedforward_params["hidden_dims"], feedforward_params["dropout"]
            )
            mention_scorer = nn.Sequential(mention_ff, nn.Linear(mention_ff.output_dim, 1))
            self.pruners[ns] = Pruner(mention_scorer)

            if self.use_transformer:
                assert transformer_params is not None
                block = TransformerBlock(span_emb_dim, **transformer_params)
                self.span_transformers[ns] = block
                self.relation_scorers[ns] = nn.Linear(
                    3 * block.output_dim + between_dim, self.n_labels[ns] - 1
                )
            else:
                relation_ff = FeedForward(
                    3 * span_emb_dim + between_dim,
                    relation_feedforward_params["hidden_dims"],
                    relation_feedforward_params["dropout"],
                )
                self.relation_feedforwards[ns] = relation_ff
                self.relation_scorers[ns] = nn.Linear(relation_ff.output_dim, self.n_labels[ns] - 1)

    @staticmethod
    def _pair_embeddings(top_embeddings: torch.Tensor) -> torch.Tensor:
        k = top_embeddings.size(0)
        e1 = top_embeddings.unsqueeze(1).expand(k, k, -1)
        e2 = top_embeddings.unsqueeze(0).expand(k, k, -1)
        return torch.cat([e1, e2, e1 * e2], dim=-1)

    @staticmethod
    def _between_context(
        word_embeddings: torch.Tensor, top_spans: list[tuple[int, int]]
    ) -> torch.Tensor:
        device, dtype = word_embeddings.device, word_embeddings.dtype
        starts = torch.tensor([s[0] for s in top_spans], dtype=torch.long, device=device)
        ends = torch.tensor([s[1] for s in top_spans], dtype=torch.long, device=device)

        zero_row = word_embeddings.new_zeros(1, word_embeddings.size(-1))
        prefix = torch.cat([zero_row, word_embeddings.cumsum(dim=0)], dim=0)  # (num_words + 1, dim)

        low = torch.minimum(ends.unsqueeze(1), ends.unsqueeze(0))
        high = torch.maximum(starts.unsqueeze(1), starts.unsqueeze(0))
        count = (high - low - 1).clamp(min=0)

        seg_sum = prefix[high] - prefix[low + 1]  # (K, K, dim)
        mean = seg_sum / count.clamp(min=1).unsqueeze(-1).to(dtype)
        return torch.where((count > 0).unsqueeze(-1), mean, seg_sum.new_zeros(()))

    def predict(
        self,
        dataset: str,
        spans: list[tuple[int, int]],
        span_embeddings: torch.Tensor,
        num_words: int,
        word_embeddings: torch.Tensor,
    ) -> list[RelationTuple]:
        ns = f"{dataset}__relation_labels"
        num_to_keep = math.ceil(num_words * self.spans_per_word)

        top_embeddings, top_indices, top_scores = self.pruners[ns](span_embeddings, num_to_keep)
        top_spans = [spans[i] for i in top_indices.tolist()]

        if self.use_transformer:
            starts = torch.tensor(
                [s[0] for s in top_spans], dtype=torch.long, device=span_embeddings.device
            )
            top_embeddings = self.span_transformers[ns](top_embeddings, starts)

        pair_embeddings = self._pair_embeddings(top_embeddings)
        if self.use_between_context:
            between = self._between_context(word_embeddings, top_spans)
            pair_embeddings = torch.cat([pair_embeddings, between], dim=-1)

        k = top_embeddings.size(0)
        if self.use_transformer:
            scores = self.relation_scorers[ns](pair_embeddings.view(k * k, -1)).view(k, k, -1)
        else:
            projected = self.relation_feedforwards[ns](pair_embeddings.view(k * k, -1))
            scores = self.relation_scorers[ns](projected).view(k, k, -1)
        scores = scores + top_scores.view(k, 1, 1) + top_scores.view(1, k, 1)
        dummy = scores.new_zeros(k, k, 1)
        scores = torch.cat([dummy, scores], dim=-1)  # (K, K, n_labels), null at index 0

        raw, predicted = scores.max(dim=-1)
        soft, _ = torch.softmax(scores, dim=-1).max(dim=-1)

        predictions: list[RelationTuple] = []
        for i, j in (predicted != 0).nonzero(as_tuple=False).tolist():
            label = self.vocab.get_token_from_index(int(predicted[i, j]), ns)
            (s1, e1), (s2, e2) = top_spans[i], top_spans[j]
            predictions.append(
                (int(s1), int(e1), int(s2), int(e2), label, float(raw[i, j]), float(soft[i, j]))
            )
        return predictions
