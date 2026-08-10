"""Acceptance test #4 — the checkpoint loader, with a mock encoder (no download).

Builds the real ported model topology from a synthetic config.json + vocab.json, but injects a tiny
fake word-embedder so no transformer encoder is ever downloaded. Verifies: strict-load round trip,
that a mismatched checkpoint raises a clear error, the ``{"model_state_dict": ...}`` and bare
state-dict forms, and that the loaded backend actually runs and returns well-formed raw tuples.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from radgraphit.core.errors import CheckpointError
from radgraphit.core.models import RawBackendOutput
from radgraphit.inference.backends.dygie_v2 import loader, tokenizer_embedder
from radgraphit.inference.backends.dygie_v2.vocab import Vocabulary, derive_dataset

HIDDEN = 8

VOCAB = {
    "radgraph-it__ner_labels": {
        "": 0,
        "Anatomy::definitely present": 1,
        "Observation::definitely absent": 2,
        "Observation::definitely present": 3,
    },
    "radgraph-it__relation_labels": {"": 0, "located_at": 1, "modify": 2, "suggestive_of": 3},
}

CONFIG = {
    "encoder": {
        "model_name": "injected-not-downloaded",
        "max_length": 512,
        "train_parameters": True,
    },
    "max_span_width": 4,
    "feature_size": 3,
    "feedforward_params": {"hidden_dims": [5], "dropout": 0.0},
    "loss_weights": {"ner": 0.2, "relation": 1.0},
    "relation_spans_per_word": 0.5,
}


class FakeEmbedder(nn.Module):
    """A tiny word-embedder with the ``MismatchedEmbedder`` interface but no transformer/download.

    Has real parameters under ``encoder.*`` so the state dict round-trips exactly like the real one.
    """

    def __init__(self, hidden: int = HIDDEN):
        super().__init__()
        self.encoder = nn.Linear(hidden, hidden)
        self._hidden = hidden

    def get_output_dim(self) -> int:
        return self._hidden

    def forward(self, words: list[str]) -> torch.Tensor:
        n = len(words)
        base = torch.arange(n * self._hidden, dtype=torch.float32).reshape(n, self._hidden)
        return self.encoder(base)


def _write_bundle(directory: Path, *, config: dict | None = None) -> tuple[Path, Path]:
    (directory / "vocab.json").write_text(json.dumps(VOCAB))
    (directory / "config.json").write_text(json.dumps(config if config is not None else CONFIG))
    return directory / "config.json", directory / "vocab.json"


def _build(config: dict, embedder: FakeEmbedder):
    vocab = _vocab_from_dict(VOCAB)
    dataset = derive_dataset(vocab)
    return loader.build_model(config, vocab, dataset, embedder=embedder)


def _vocab_from_dict(data: dict) -> Vocabulary:
    vocab = Vocabulary()
    for namespace, t2i in data.items():
        vocab.add_namespace(namespace)
        for token, _idx in sorted(t2i.items(), key=lambda kv: kv[1]):
            if token != "":
                vocab.add_token(token, namespace)
    return vocab


def test_derive_dataset_from_vocab() -> None:
    assert derive_dataset(_vocab_from_dict(VOCAB)) == "radgraph-it"


def test_strict_load_round_trip_and_infer(tmp_path: Path) -> None:
    config_path, vocab_path = _write_bundle(tmp_path)

    # Build a model, save its checkpoint in the trainer's {"model_state_dict": ...} form.
    source = _build(CONFIG, FakeEmbedder())
    weights_path = tmp_path / "best.pt"
    torch.save({"epoch": 3, "model_state_dict": source.state_dict()}, weights_path)

    backend = loader.load_backend(
        config_path=config_path,
        vocab_path=vocab_path,
        weights_path=weights_path,
        device="cpu",
        embedder=FakeEmbedder(),
    )

    out = backend.infer(("Nessuna", "evidenza", "di", "pneumotorace", "."))
    assert isinstance(out, RawBackendOutput)
    for entry in out.ner:
        start, end, label, raw, soft = entry
        assert isinstance(start, int) and isinstance(end, int)
        assert label in VOCAB["radgraph-it__ner_labels"]
        assert isinstance(raw, float) and isinstance(soft, float)
    for entry in out.relations:
        assert len(entry) == 7
        assert entry[4] in VOCAB["radgraph-it__relation_labels"]


def test_bare_state_dict_without_wrapper_is_accepted(tmp_path: Path) -> None:
    config_path, vocab_path = _write_bundle(tmp_path)
    source = _build(CONFIG, FakeEmbedder())
    weights_path = tmp_path / "weights.pt"
    torch.save(source.state_dict(), weights_path)  # no {"model_state_dict": ...} wrapper

    backend = loader.load_backend(
        config_path=config_path,
        vocab_path=vocab_path,
        weights_path=weights_path,
        device="cpu",
        embedder=FakeEmbedder(),
    )
    assert backend.infer(("referto", "normale")).ner is not None


def test_safetensors_round_trip_is_accepted(tmp_path: Path) -> None:
    config_path, vocab_path = _write_bundle(tmp_path)
    source = _build(CONFIG, FakeEmbedder())
    weights_path = tmp_path / "model.safetensors"
    save_file(source.state_dict(), weights_path)

    backend = loader.load_backend(
        config_path=config_path,
        vocab_path=vocab_path,
        weights_path=weights_path,
        device="cpu",
        embedder=FakeEmbedder(),
    )
    assert backend.infer(("referto", "normale")).ner is not None


def test_mismatched_checkpoint_raises_checkpoint_error(tmp_path: Path) -> None:
    config_path, vocab_path = _write_bundle(tmp_path)

    # Checkpoint from a model with a *different* feature_size -> width_embedding shape differs.
    mismatched_config = {**CONFIG, "feature_size": 7}
    other = _build(mismatched_config, FakeEmbedder())
    weights_path = tmp_path / "best.pt"
    torch.save({"model_state_dict": other.state_dict()}, weights_path)

    with pytest.raises(CheckpointError, match="strict=True"):
        loader.load_backend(
            config_path=config_path,
            vocab_path=vocab_path,
            weights_path=weights_path,
            device="cpu",
            embedder=FakeEmbedder(),
        )


def test_config_missing_required_field_raises(tmp_path: Path) -> None:
    bad_config = {"encoder": {"model_name": "x", "max_length": 512}}  # missing max_span_width, ...
    config_path, vocab_path = _write_bundle(tmp_path, config=bad_config)
    (tmp_path / "best.pt").write_bytes(b"unused")
    with pytest.raises(CheckpointError, match="missing or has an invalid required field"):
        loader.load_backend(
            config_path=config_path,
            vocab_path=vocab_path,
            weights_path=tmp_path / "best.pt",
            device="cpu",
            embedder=FakeEmbedder(),
        )


def test_malformed_config_is_wrapped_as_checkpoint_error(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text("{not-json")
    vocab_path = tmp_path / "vocab.json"
    vocab_path.write_text(json.dumps(VOCAB))
    with pytest.raises(CheckpointError, match="valid JSON config"):
        loader.load_backend(
            config_path=config_path,
            vocab_path=vocab_path,
            weights_path=tmp_path / "best.pt",
            device="cpu",
            embedder=FakeEmbedder(),
        )


def test_malformed_vocab_is_wrapped_as_checkpoint_error(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(CONFIG))
    vocab_path = tmp_path / "vocab.json"
    vocab_path.write_text("{not-json")
    with pytest.raises(CheckpointError, match="valid JSON vocabulary"):
        loader.load_backend(
            config_path=config_path,
            vocab_path=vocab_path,
            weights_path=tmp_path / "best.pt",
            device="cpu",
            embedder=FakeEmbedder(),
        )


def test_non_contiguous_vocab_indices_are_rejected(tmp_path: Path) -> None:
    bad_vocab = {
        **VOCAB,
        "radgraph-it__ner_labels": {"": 0, "Observation::uncertain": 2},
    }
    path = tmp_path / "vocab.json"
    path.write_text(json.dumps(bad_vocab))
    with pytest.raises(CheckpointError, match="contiguous"):
        Vocabulary.load(path)


def test_corrupt_checkpoint_is_wrapped_as_checkpoint_error(tmp_path: Path) -> None:
    config_path, vocab_path = _write_bundle(tmp_path)
    weights_path = tmp_path / "best.pt"
    weights_path.write_bytes(b"not a torch checkpoint")
    with pytest.raises(CheckpointError, match="safely read checkpoint"):
        loader.load_backend(
            config_path=config_path,
            vocab_path=vocab_path,
            weights_path=weights_path,
            device="cpu",
            embedder=FakeEmbedder(),
        )


def test_embedder_uses_pinned_config_without_loading_pretrained_encoder_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, object] = {}

    class FakeTokenizer:
        def num_special_tokens_to_add(self, pair: bool) -> int:
            assert pair is False
            return 2

    class FakeEncoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(1))
            self.config = type("Config", (), {"hidden_size": HIDDEN})()

    def fake_tokenizer(model_name: str, *, revision: str | None):
        calls["tokenizer"] = (model_name, revision)
        return FakeTokenizer()

    def fake_config(model_name: str, *, revision: str | None):
        calls["config"] = (model_name, revision)
        return object()

    def fake_model(config: object):
        calls["model_config"] = config
        return FakeEncoder()

    monkeypatch.setattr(tokenizer_embedder.AutoTokenizer, "from_pretrained", fake_tokenizer)
    monkeypatch.setattr(tokenizer_embedder.AutoConfig, "from_pretrained", fake_config)
    monkeypatch.setattr(tokenizer_embedder.AutoModel, "from_config", fake_model)

    revision = "0f1c91e551551869a72935962b54c1f5247333a4"
    embedder = tokenizer_embedder.MismatchedEmbedder(
        "IVN-RIN/medBIT-r3-plus",
        512,
        revision=revision,
    )

    assert calls["tokenizer"] == ("IVN-RIN/medBIT-r3-plus", revision)
    assert calls["config"] == ("IVN-RIN/medBIT-r3-plus", revision)
    assert calls["model_config"] is not None
    assert embedder.get_output_dim() == HIDDEN


def test_unexpected_model_construction_error_is_wrapped(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_model(*args, **kwargs):
        raise RuntimeError("encoder construction failed")

    monkeypatch.setattr(loader, "DyGIEModel", fail_model)
    with pytest.raises(CheckpointError, match="Could not construct"):
        loader.build_model(CONFIG, _vocab_from_dict(VOCAB), "radgraph-it", embedder=FakeEmbedder())


def test_corrupt_safetensors_and_unexpected_pt_structure_are_rejected(tmp_path: Path) -> None:
    safetensors_path = tmp_path / "model.safetensors"
    safetensors_path.write_bytes(b"not safetensors")
    with pytest.raises(CheckpointError, match="Could not read safetensors"):
        loader._read_state_dict(safetensors_path)

    pt_path = tmp_path / "weights.pt"
    torch.save({"parameter": 123}, pt_path)
    with pytest.raises(CheckpointError, match="Unexpected checkpoint structure"):
        loader._read_state_dict(pt_path)


def test_config_must_be_a_json_object(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text("[]")
    with pytest.raises(CheckpointError, match="must contain a JSON object"):
        loader.load_backend(
            config_path=config_path,
            vocab_path=tmp_path / "vocab.json",
            weights_path=tmp_path / "weights.pt",
        )


def test_device_transfer_error_is_wrapped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path, vocab_path = _write_bundle(tmp_path)

    class BrokenModel:
        def to(self, _device):
            raise RuntimeError("device transfer failed")

    monkeypatch.setattr(loader, "build_model", lambda *args, **kwargs: BrokenModel())
    monkeypatch.setattr(loader, "load_weights", lambda *args, **kwargs: None)
    with pytest.raises(CheckpointError, match="Could not move"):
        loader.load_backend(
            config_path=config_path,
            vocab_path=vocab_path,
            weights_path=tmp_path / "weights.pt",
            device="cpu",
        )
