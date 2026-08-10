"""NER head — inference-only port of ``training_v2/src/dygie/ner_head.py``.

The parameter-bearing structure (``scorers`` + ``classifiers`` ModuleDicts, and the null-column
trick: score ``n_labels - 1`` real classes, prepend a fixed-0 null column) is preserved exactly so
the state dict matches the checkpoint. Loss and metrics are dropped — inference only needs the
decode step, which returns plain ``(start, end, label, raw_score, softmax_score)`` tuples.
"""

from __future__ import annotations

import torch
from torch import nn

from .feedforward import FeedForward
from .transformer_block import TransformerBlock
from .vocab import Vocabulary

NerTuple = tuple[int, int, str, float, float]


class NERHead(nn.Module):
    def __init__(
        self,
        vocab: Vocabulary,
        span_emb_dim: int,
        feedforward_params: dict,
        transformer_params: dict | None = None,
    ):
        super().__init__()
        self.vocab = vocab
        self.namespaces = [ns for ns in vocab.get_namespaces() if ns.endswith("ner_labels")]
        self.n_labels = {ns: vocab.get_vocab_size(ns) for ns in self.namespaces}
        self.use_transformer = transformer_params is not None

        self.scorers = nn.ModuleDict()
        self.classifiers = nn.ModuleDict()
        for ns in self.namespaces:
            if self.use_transformer:
                assert transformer_params is not None
                block = TransformerBlock(span_emb_dim, **transformer_params)
                self.scorers[ns] = block
                body_out = block.output_dim
            else:
                ff = FeedForward(
                    span_emb_dim, feedforward_params["hidden_dims"], feedforward_params["dropout"]
                )
                self.scorers[ns] = ff
                body_out = ff.output_dim
            self.classifiers[ns] = nn.Linear(body_out, self.n_labels[ns] - 1)

    def predict(
        self, dataset: str, spans: list[tuple[int, int]], span_embeddings: torch.Tensor
    ) -> list[NerTuple]:
        ns = f"{dataset}__ner_labels"
        if self.use_transformer:
            starts = torch.tensor(
                [s[0] for s in spans], dtype=torch.long, device=span_embeddings.device
            )
            hidden = self.scorers[ns](span_embeddings, starts)
        else:
            hidden = self.scorers[ns](span_embeddings)
        scores = self.classifiers[ns](hidden)  # (num_spans, n_labels - 1)
        dummy = scores.new_zeros(scores.size(0), 1)  # null-label score, fixed at 0
        scores = torch.cat([dummy, scores], dim=-1)  # (num_spans, n_labels)

        softmax_scores = torch.softmax(scores, dim=-1)
        raw, predicted = scores.max(dim=-1)
        soft, _ = softmax_scores.max(dim=-1)

        predictions: list[NerTuple] = []
        for i in (predicted != 0).nonzero(as_tuple=True)[0].tolist():
            label = self.vocab.get_token_from_index(int(predicted[i]), ns)
            start, end = spans[i]
            predictions.append((int(start), int(end), label, float(raw[i]), float(soft[i])))
        return predictions
