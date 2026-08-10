"""Focused coverage for optional DyGIE-v2 architecture arms and defensive boundaries."""

from __future__ import annotations

import importlib
import json
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import pytest
import torch
from torch import nn

from radgraphit.core.errors import BackendError, CheckpointError
from radgraphit.core.models import (
    ReportPrediction,
    ScoredEntity,
    ScoredRelation,
    TokenizedReport,
)
from radgraphit.inference.backends.dygie_v2 import tokenizer_embedder
from radgraphit.inference.backends.dygie_v2.backend import DyGIEv2Backend
from radgraphit.inference.backends.dygie_v2.ner_head import NERHead
from radgraphit.inference.backends.dygie_v2.relation_head import RelationHead
from radgraphit.inference.backends.dygie_v2.span_extractor import EndpointSpanExtractor
from radgraphit.inference.backends.dygie_v2.tokenizer_embedder import MismatchedEmbedder
from radgraphit.inference.backends.dygie_v2.transformer_block import (
    TransformerBlock,
    sinusoidal_position_encoding,
)
from radgraphit.inference.backends.dygie_v2.vocab import Vocabulary, derive_dataset


def _vocab() -> Vocabulary:
    vocab = Vocabulary()
    vocab.add_token("Observation::definitely present", "sample__ner_labels")
    vocab.add_token("located_at", "sample__relation_labels")
    return vocab


def test_value_object_helpers_and_backend_protocol_module() -> None:
    report = TokenizedReport(("uno", "due"))
    assert len(report) == 2
    assert report.text == "uno due"

    entity = ScoredEntity(0, 1, "label", 1.0, 0.8)
    relation = ScoredRelation("located_at", 0, 1, 2, 2, 1.0, 0.9)
    prediction = ReportPrediction("0", ("uno", "due"), (entity,), (relation,))
    assert entity.span == (0, 1)
    assert relation.source_span == (0, 1)
    assert relation.target_span == (2, 2)
    assert prediction.text == "uno due"

    module = importlib.import_module("radgraphit.inference.backends.base")
    assert module.__all__ == ["InferenceBackend", "RawBackendOutput"]


def test_backend_device_property_success_and_error_translation() -> None:
    class SuccessfulModel:
        def predict(self, _tokens):
            return [(0, 0, "label", 1.0, 0.9)], []

    backend = DyGIEv2Backend(SuccessfulModel(), torch.device("cpu"))  # type: ignore[arg-type]
    assert backend.device == torch.device("cpu")
    assert backend.infer(("test",)).ner[0][2] == "label"

    class BrokenModel:
        def predict(self, _tokens):
            raise RuntimeError("inference exploded")

    broken = DyGIEv2Backend(BrokenModel(), torch.device("cpu"))  # type: ignore[arg-type]
    with pytest.raises(BackendError, match="inference failed"):
        broken.infer(("test",))


def test_span_extractor_mean_pooling() -> None:
    extractor = EndpointSpanExtractor(2, 4, 1, mean_pool=True)
    words = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    spans = torch.tensor([[0, 0], [0, 2]], dtype=torch.long)
    output = extractor(words, spans)
    assert output.shape == (2, extractor.get_output_dim())
    assert torch.equal(output[0, 4:6], words[0])
    assert torch.equal(output[1, 4:6], words.mean(dim=0))


def test_transformer_block_identity_projection_and_positional_encoding() -> None:
    positions = torch.tensor([0, 2], dtype=torch.long)
    assert sinusoidal_position_encoding(positions, 3).shape == (2, 3)

    common = {
        "d_model": 4,
        "nhead": 2,
        "num_layers": 1,
        "dim_feedforward": 8,
        "dropout": 0.0,
    }
    identity = TransformerBlock(4, **common).eval()
    projected = TransformerBlock(3, **common).eval()
    assert isinstance(identity.input_proj, nn.Identity)
    assert isinstance(projected.input_proj, nn.Linear)
    assert identity(torch.ones(2, 4), positions).shape == (2, 4)
    assert projected(torch.ones(2, 3), positions).shape == (2, 4)


def test_transformer_ner_and_relation_arms_with_between_context() -> None:
    vocab = _vocab()
    transformer = {
        "d_model": 4,
        "nhead": 2,
        "num_layers": 1,
        "dim_feedforward": 8,
        "dropout": 0.0,
    }
    feedforward = {"hidden_dims": [4], "dropout": 0.0}
    spans = [(0, 0), (2, 2)]
    span_embeddings = torch.tensor(
        [[1.0, 0.0, 0.5, -0.5], [0.0, 1.0, -0.5, 0.5]], dtype=torch.float32
    )

    ner = NERHead(vocab, 4, feedforward, transformer).eval()
    with torch.no_grad():
        assert isinstance(ner.predict("sample", spans, span_embeddings), list)

    relation = RelationHead(
        vocab,
        span_emb_dim=4,
        word_emb_dim=2,
        feedforward_params=feedforward,
        spans_per_word=1.0,
        transformer_params=transformer,
        use_between_context=True,
    ).eval()
    word_embeddings = torch.tensor([[1.0, 1.0], [2.0, 4.0], [3.0, 9.0]])
    with torch.no_grad():
        assert isinstance(
            relation.predict("sample", spans, span_embeddings, 2, word_embeddings),
            list,
        )

    between = RelationHead._between_context(
        torch.tensor([[1.0], [2.0], [3.0], [4.0]]), [(0, 0), (3, 3)]
    )
    assert between.shape == (2, 2, 1)
    assert between[0, 1, 0].item() == pytest.approx(2.5)
    assert between[0, 0, 0].item() == 0.0


class _BatchEncoding(dict[str, torch.Tensor]):
    def __init__(self, word_ids: list[int | None]):
        super().__init__(
            input_ids=torch.arange(len(word_ids)).unsqueeze(0),
            attention_mask=torch.ones(1, len(word_ids), dtype=torch.long),
        )
        self._word_ids = word_ids

    def word_ids(self, batch_index: int | None = None) -> list[int | None]:
        assert batch_index in {None, 0}
        return self._word_ids


class _FakeTokenizer:
    _counts: ClassVar[dict[str, int]] = {"aa": 2, "bbb": 3, "empty": 0, "long": 5}

    def num_special_tokens_to_add(self, pair: bool) -> int:
        assert pair is False
        return 2

    def __call__(self, words: list[str], *, add_special_tokens: bool, **_kwargs):
        word_ids = [index for index, word in enumerate(words) for _ in range(self._counts[word])]
        word_ids = [None, *word_ids, None] if add_special_tokens else [None, *word_ids]
        return _BatchEncoding(word_ids)


class _FakeEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))
        self.config = SimpleNamespace(hidden_size=2)

    def forward(self, *, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        assert input_ids.shape == attention_mask.shape
        positions = input_ids.to(torch.float32).unsqueeze(-1)
        hidden = torch.cat([positions, positions + self.weight], dim=-1)
        return SimpleNamespace(last_hidden_state=hidden)


def _patch_encoder_factories(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        tokenizer_embedder.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: _FakeTokenizer(),
    )
    monkeypatch.setattr(
        tokenizer_embedder.AutoConfig, "from_pretrained", lambda *args, **kwargs: object()
    )
    monkeypatch.setattr(
        tokenizer_embedder.AutoModel, "from_config", lambda *args, **kwargs: _FakeEncoder()
    )


def test_mismatched_embedder_chunks_pools_freezes_and_fills_empty_words(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _patch_encoder_factories(monkeypatch)
    embedder = MismatchedEmbedder("fake/model", 5, train_parameters=False)
    assert embedder.get_output_dim() == 2
    assert all(not parameter.requires_grad for parameter in embedder.encoder.parameters())
    assert embedder._word_subword_counts(["aa", "bbb", "empty"]) == [2, 3, 0]
    assert embedder._pack_chunks([]) == []

    with caplog.at_level(logging.WARNING, logger="radgraphit"):
        assert embedder._pack_chunks([5, 0, 2, 2]) == [[0], [1, 2], [3]]
    assert "will be truncated" in caplog.text

    output = embedder(["aa", "bbb", "empty"])
    assert output.shape == (3, 2)
    assert torch.count_nonzero(output[0]).item() > 0
    assert torch.equal(output[2], torch.zeros(2))


def test_mismatched_embedder_rejects_a_budget_without_content_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_encoder_factories(monkeypatch)
    with pytest.raises(ValueError, match="too small"):
        MismatchedEmbedder("fake/model", 2)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({}, "non-empty object"),
        ({"": {"": 0}}, "namespace"),
        ({"sample__ner_labels": {}}, "non-empty object"),
        ({"sample__ner_labels": {"": 1}}, "null label"),
        ({"sample__ner_labels": {"": 0, "label": True}}, "non-negative integers"),
    ],
)
def test_vocabulary_rejects_malformed_namespaces(
    tmp_path: Path, payload: dict, message: str
) -> None:
    path = tmp_path / "vocab.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(CheckpointError, match=message):
        Vocabulary.load(path)


def test_vocabulary_accessors_duplicate_tokens_and_io_error(tmp_path: Path) -> None:
    vocab = _vocab()
    first = vocab.add_token("located_at", "sample__relation_labels")
    second = vocab.add_token("located_at", "sample__relation_labels")
    assert first == second == 1
    assert vocab.get_token_index("located_at", "sample__relation_labels") == 1

    with pytest.raises(CheckpointError, match="valid JSON vocabulary"):
        Vocabulary.load(tmp_path / "missing.json")


def test_dataset_derivation_requires_one_matching_namespace_pair() -> None:
    with pytest.raises(CheckpointError, match="exactly one"):
        derive_dataset(Vocabulary())

    vocab = Vocabulary()
    vocab.add_namespace("ner-source__ner_labels")
    vocab.add_namespace("relation-source__relation_labels")
    with pytest.raises(CheckpointError, match="does not match"):
        derive_dataset(vocab)
