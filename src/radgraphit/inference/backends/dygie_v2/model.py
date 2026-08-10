"""The joint model — inference-only port of ``training_v2/src/dygie/model.py``.

Wires ``embedder -> span_extractor -> ner + relation``. The four submodule names (``embedder``,
``span_extractor``, ``ner``, ``relation``) and everything beneath them are preserved exactly, so
``state_dict()`` keys line up with the training checkpoint and ``load_state_dict(strict=True)``
succeeds. The training-only pieces (loss weighting, xavier re-init, metrics) are gone — every
parameter is overwritten by the checkpoint at load time anyway.

``embedder`` is injectable purely so the loader can be unit-tested without downloading an encoder;
production always uses the default :class:`MismatchedEmbedder`.
"""

from __future__ import annotations

import torch
from torch import nn

from .ner_head import NERHead, NerTuple
from .relation_head import RelationHead, RelationTuple
from .span_extractor import EndpointSpanExtractor
from .spans import enumerate_spans
from .tokenizer_embedder import MismatchedEmbedder
from .vocab import Vocabulary


class DyGIEModel(nn.Module):
    def __init__(
        self,
        vocab: Vocabulary,
        dataset: str,
        *,
        encoder_name: str,
        encoder_revision: str | None,
        max_length: int,
        max_span_width: int,
        feature_size: int,
        feedforward_params: dict,
        relation_spans_per_word: float,
        span_pooling: bool = False,
        transformer_params: dict | None = None,
        relation_context: bool = False,
        relation_feedforward_params: dict | None = None,
        train_encoder: bool = True,
        embedder: MismatchedEmbedder | None = None,
    ):
        super().__init__()
        self.vocab = vocab
        self.dataset = dataset
        self.max_span_width = max_span_width

        self.embedder = (
            embedder
            if embedder is not None
            else MismatchedEmbedder(
                encoder_name,
                max_length,
                train_encoder,
                revision=encoder_revision,
            )
        )
        self.span_extractor = EndpointSpanExtractor(
            self.embedder.get_output_dim(),
            num_width_embeddings=max_span_width,
            span_width_embedding_dim=feature_size,
            mean_pool=span_pooling,
        )
        span_emb_dim = self.span_extractor.get_output_dim()

        self.ner = NERHead(vocab, span_emb_dim, feedforward_params, transformer_params)
        self.relation = RelationHead(
            vocab,
            span_emb_dim,
            self.embedder.get_output_dim(),
            feedforward_params,
            relation_spans_per_word,
            transformer_params,
            relation_context,
            relation_feedforward_params,
        )

    @torch.no_grad()
    def predict(self, tokens: tuple[str, ...]) -> tuple[list[NerTuple], list[RelationTuple]]:
        """Run NER + relation inference on one report's tokens.

        Returns ``(ner_predictions, relation_predictions)`` as plain tuples. Assumes a non-empty
        token list (the service guarantees this).
        """
        self.eval()
        words = list(tokens)
        spans = enumerate_spans(words, self.max_span_width)
        device = next(self.parameters()).device

        word_embeddings = self.embedder(words)
        spans_tensor = torch.tensor(spans, dtype=torch.long, device=device)
        span_embeddings = self.span_extractor(word_embeddings, spans_tensor)

        ner = self.ner.predict(self.dataset, spans, span_embeddings)
        relations = self.relation.predict(
            self.dataset, spans, span_embeddings, len(words), word_embeddings
        )
        return ner, relations
