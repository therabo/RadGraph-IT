"""Shared, torch-free test fixtures.

Nothing here imports torch: the fake backend speaks the same plain-tuple contract the real backend
does (:class:`RawBackendOutput`), so the whole tokenizer -> backend -> decoder -> serializer -> API
path can be exercised without a model. torch-dependent helpers live inside the individual tests
that need them (``test_loader.py``, ``test_device.py``).

Helpers are exposed as fixtures (not importable module attributes) so tests never need fragile
cross-directory imports.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path

import pytest

from radgraphit.core.models import RawBackendOutput

BackendOutput = RawBackendOutput | Callable[[tuple[str, ...]], RawBackendOutput]


class RecordingBackend:
    """A fake ``InferenceBackend`` that records every ``infer`` call and returns a preset output.

    ``output`` is either a fixed :class:`RawBackendOutput` or a callable ``tokens -> output``.
    """

    def __init__(self, output: BackendOutput | None = None):
        self.calls: list[tuple[str, ...]] = []
        self._output: BackendOutput = output if output is not None else RawBackendOutput((), ())

    def infer(self, tokens: tuple[str, ...]) -> RawBackendOutput:
        self.calls.append(tokens)
        if callable(self._output):
            return self._output(tokens)
        return self._output


@pytest.fixture
def backend_factory() -> Callable[[BackendOutput], RecordingBackend]:
    """Return ``make(output) -> RecordingBackend`` for tests that need a custom backend."""

    def make(output: BackendOutput) -> RecordingBackend:
        return RecordingBackend(output)

    return make


@pytest.fixture
def recording_backend() -> RecordingBackend:
    """A backend that marks token 0 as an entity, so serialized output is easy to assert on."""

    def mark_first_token(tokens: tuple[str, ...]) -> RawBackendOutput:
        return RawBackendOutput(
            ner=((0, 0, "Observation::definitely present", 1.0, 0.9),),
            relations=(),
        )

    return RecordingBackend(mark_first_token)


@pytest.fixture
def empty_backend() -> RecordingBackend:
    """A backend that predicts nothing (empty graph) for every report."""
    return RecordingBackend(RawBackendOutput((), ()))


_DEFAULT_BUNDLE_FILES = {
    "config.json": b'{"encoder": {"model_name": "IVN-RIN/medBIT-r3-plus", "max_length": 512}}',
    "vocab.json": b'{"radgraph-it__ner_labels": {"": 0}, "radgraph-it__relation_labels": {"": 0}}',
    "model.safetensors": b"\x00\x01\x02 dummy-weights-not-loaded-by-manifest-or-resolver-tests",
}


@pytest.fixture
def bundle_builder() -> Callable[..., Path]:
    """Return ``build(directory, *, files=None, with_manifest=True, subdir=None, **overrides)``.

    Writes a torch-free bundle: the listed files, and (when ``with_manifest``) a
    ``model_manifest.json`` whose SHA-256s match them. ``subdir`` nests the bundle files in a
    subdirectory of ``directory`` (e.g. ``"model"``) to mimic the published repo layout. The weights
    file is dummy bytes — good enough for manifest/resolver/API tests, which never load the model
    (they mock the backend build). Always returns ``directory`` (the root pointed at the resolver).
    """

    def build(
        directory: Path,
        *,
        files: dict[str, bytes] | None = None,
        with_manifest: bool = True,
        subdir: str | None = None,
        **manifest_overrides: object,
    ) -> Path:
        target = directory / subdir if subdir else directory
        target.mkdir(parents=True, exist_ok=True)
        payload = files if files is not None else _DEFAULT_BUNDLE_FILES
        for name, content in payload.items():
            (target / name).write_bytes(content)
        if with_manifest:
            manifest: dict[str, object] = {
                "manifest_version": 1,
                "package": "radgraphit",
                "min_package_version": "1.0.0",
                "schema": "radgraph-xl",
                "backend": "dygie_v2",
                "encoder": {
                    "model_name": "IVN-RIN/medBIT-r3-plus",
                    "revision": "0f1c91e551551869a72935962b54c1f5247333a4",
                    "max_length": 512,
                },
                "revision": "test-revision",
                "weights_file": "model.safetensors",
                "files": {
                    name: {"sha256": hashlib.sha256(content).hexdigest()}
                    for name, content in payload.items()
                },
            }
            manifest.update(manifest_overrides)
            (target / "model_manifest.json").write_text(json.dumps(manifest))
        return directory

    return build
