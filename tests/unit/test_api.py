"""Acceptance test #5 — the public API facade with an injected fake backend and a mocked Hub.

Covers: single-string and batch predict, the ``__call__`` alias, lazy loading, configuration errors,
and ``from_pretrained`` forwarding repository/revision/cache/device to the resolver and backend.
No network, no torch, no real weights.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from radgraphit import RadGraphIT
from radgraphit.api import RadGraphIT as ApiRadGraphIT
from radgraphit.api import _build_backend
from radgraphit.artifacts import registry, resolver
from radgraphit.core.errors import BackendError, ModelNotConfiguredError
from radgraphit.core.models import RawBackendOutput
from radgraphit.inference.service import InferenceService


def test_predict_single_string(recording_backend) -> None:
    predictor = RadGraphIT.from_service(InferenceService(recording_backend))
    result = predictor.predict("Nessuna evidenza di pneumotorace.")
    assert list(result.keys()) == ["0"]
    assert result["0"]["entities"]["1"]["tokens"] == "Nessuna"
    assert result["0"]["data_split"] == "inference"
    assert recording_backend.calls  # backend was actually called


def test_predict_batch_preserves_order(recording_backend) -> None:
    predictor = RadGraphIT.from_service(InferenceService(recording_backend))
    result = predictor.predict(["Referto uno.", "Referto due.", "Referto tre."])
    assert list(result.keys()) == ["0", "1", "2"]
    # token 0 of each report is the entity; order is preserved.
    assert result["0"]["entities"]["1"]["tokens"] == "Referto"
    assert [c[0] for c in recording_backend.calls] == ["Referto", "Referto", "Referto"]


def test_call_is_alias_of_predict(recording_backend) -> None:
    predictor = RadGraphIT.from_service(InferenceService(recording_backend))
    text = "Nessuna evidenza di pneumotorace."
    assert predictor(text) == predictor.predict(text)


def test_predict_graphs_returns_typed_objects(recording_backend) -> None:
    predictor = RadGraphIT.from_service(InferenceService(recording_backend))
    graphs = predictor.predict_graphs("Nessuna evidenza.")
    assert graphs[0].entities[0].label == "Observation::definitely present"


def test_construction_is_lazy_and_does_no_io(monkeypatch: pytest.MonkeyPatch) -> None:
    # snapshot_download must not be called at construction time — only on first predict.
    monkeypatch.setattr(
        resolver, "snapshot_download", lambda **kw: pytest.fail("network at construction")
    )
    predictor = RadGraphIT(device="cpu")  # default (configured) model — must not resolve here
    assert predictor._service is None


def test_empty_model_id_raises_on_first_predict() -> None:
    predictor = RadGraphIT.from_pretrained("")
    with pytest.raises(ModelNotConfiguredError, match="non-empty"):
        predictor.predict("Nessuna evidenza.")


def test_from_pretrained_forwards_repo_revision_cache_and_device(
    tmp_path: Path, bundle_builder, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = bundle_builder(tmp_path)
    snapshot_kwargs: dict[str, object] = {}
    build_args: dict[str, object] = {}

    def fake_snapshot_download(**kwargs):
        snapshot_kwargs.update(kwargs)
        return str(bundle)

    def fake_hf_hub_download(**kwargs):
        path = bundle / kwargs["filename"]
        if not path.is_file():
            raise OSError("not found")
        return str(path)

    monkeypatch.setattr(resolver, "hf_hub_download", fake_hf_hub_download)

    def fake_build_backend(resolved, device):
        build_args["device"] = device
        build_args["backend_name"] = resolved.manifest.backend

        class _Fake:
            def infer(self, tokens):
                return RawBackendOutput(
                    ner=((0, 0, "Anatomy::definitely present", 1.0, 0.9),), relations=()
                )

        return _Fake()

    monkeypatch.setattr(resolver, "snapshot_download", fake_snapshot_download)
    monkeypatch.setattr("radgraphit.api._build_backend", fake_build_backend)

    predictor = ApiRadGraphIT.from_pretrained(
        "ORG/RadGraphIT-v1", revision="rev-42", cache_dir="/tmp/cache", device="cpu"
    )
    # Still lazy: nothing downloaded yet.
    assert snapshot_kwargs == {}

    result = predictor.predict("Nessuna evidenza.")

    assert snapshot_kwargs["repo_id"] == "ORG/RadGraphIT-v1"
    assert snapshot_kwargs["revision"] == "rev-42"
    assert snapshot_kwargs["cache_dir"] == "/tmp/cache"
    assert build_args == {"device": "cpu", "backend_name": "dygie_v2"}
    assert result["0"]["entities"]["1"]["label"] == "Anatomy::definitely present"


def test_from_local_verifies_and_loads(
    tmp_path: Path, bundle_builder, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = bundle_builder(tmp_path)

    def fake_build_backend(resolved, device):
        assert resolved.directory == bundle

        class _Fake:
            def infer(self, tokens):
                return RawBackendOutput(ner=(), relations=())

        return _Fake()

    monkeypatch.setattr("radgraphit.api._build_backend", fake_build_backend)
    predictor = RadGraphIT.from_local(str(bundle), device="cpu")
    result = predictor.predict("Referto normale.")
    assert result == {
        "0": {
            "text": "Referto normale .",
            "entities": {},
            "data_source": None,
            "data_split": "inference",
        }
    }


def test_registry_defaults_are_pinned() -> None:
    assert registry.DEFAULT_MODEL_ID == "radgraphIT/Radgraph-IT"
    assert registry.DEFAULT_REVISION == "11f691cd0e42985b87330e197507c91b43b9bfd8"


def test_load_is_chainable_for_an_already_constructed_service(recording_backend) -> None:
    predictor = RadGraphIT.from_service(InferenceService(recording_backend))
    assert predictor.load() is predictor


def test_backend_factory_forwards_all_verified_bundle_paths(
    tmp_path: Path, bundle_builder, monkeypatch: pytest.MonkeyPatch
) -> None:
    resolved = resolver.resolve_local(bundle_builder(tmp_path))
    calls: dict[str, object] = {}
    sentinel = object()

    def fake_load_backend(**kwargs):
        calls.update(kwargs)
        return sentinel

    monkeypatch.setattr("radgraphit.inference.backends.dygie_v2.load_backend", fake_load_backend)
    assert _build_backend(resolved, "cpu") is sentinel
    assert calls == {
        "config_path": resolved.config_path,
        "vocab_path": resolved.vocab_path,
        "weights_path": resolved.weights_path,
        "device": "cpu",
        "encoder_revision": resolved.manifest.encoder_revision,
    }


def test_backend_factory_rejects_an_unknown_backend(tmp_path: Path, bundle_builder) -> None:
    resolved = resolver.resolve_local(bundle_builder(tmp_path))
    object.__setattr__(resolved.manifest, "backend", "unknown")
    with pytest.raises(BackendError, match="does not support"):
        _build_backend(resolved, "cpu")
