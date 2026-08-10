"""Hub/local resolver tests with all network access replaced by local bundle fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

from radgraphit.artifacts import registry, resolver
from radgraphit.core.errors import (
    ManifestError,
    ModelNotConfiguredError,
    ModelNotFoundError,
)


def _mock_hub(
    monkeypatch: pytest.MonkeyPatch,
    root: Path,
    *,
    recorded: dict[str, object] | None = None,
) -> None:
    def fake_hf_hub_download(**kwargs):
        candidate = root / kwargs["filename"]
        if not candidate.is_file():
            raise OSError(f"{kwargs['filename']} not found")
        return str(candidate)

    def fake_snapshot_download(**kwargs):
        if recorded is not None:
            recorded.update(kwargs)
        return str(root)

    monkeypatch.setattr(resolver, "hf_hub_download", fake_hf_hub_download)
    monkeypatch.setattr(resolver, "snapshot_download", fake_snapshot_download)


@pytest.mark.parametrize(
    ("model_id", "revision"),
    [
        ("", "abc123"),
        ("ORG/RadGraphIT-v1", ""),
    ],
)
def test_empty_hub_configuration_raises_before_download(
    model_id: str,
    revision: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        resolver,
        "hf_hub_download",
        lambda **kwargs: pytest.fail("must not download"),
    )
    with pytest.raises(ModelNotConfiguredError, match="non-empty"):
        resolver.resolve_pretrained(model_id, revision=revision)


def test_default_model_and_encoder_revisions_are_immutable_commit_shas() -> None:
    assert len(registry.DEFAULT_REVISION) == 40
    assert all(char in "0123456789abcdef" for char in registry.DEFAULT_REVISION)
    assert registry.ENCODER_REVISIONS
    for revision in registry.ENCODER_REVISIONS.values():
        assert len(revision) == 40
        assert all(char in "0123456789abcdef" for char in revision)


def test_resolve_pretrained_downloads_only_manifest_declared_files(
    tmp_path: Path,
    bundle_builder,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = bundle_builder(tmp_path)
    recorded: dict[str, object] = {}
    _mock_hub(monkeypatch, bundle, recorded=recorded)

    resolved = resolver.resolve_pretrained(
        "ORG/RadGraphIT-v1",
        revision="abc123",
        cache_dir="/tmp/radgraphit-cache",
    )

    assert recorded["repo_id"] == "ORG/RadGraphIT-v1"
    assert recorded["revision"] == "abc123"
    assert recorded["cache_dir"] == "/tmp/radgraphit-cache"
    assert recorded["local_files_only"] is False
    assert sorted(recorded["allow_patterns"]) == [
        "config.json",
        "model.safetensors",
        "model_manifest.json",
        "vocab.json",
    ]
    assert resolved.manifest.backend == "dygie_v2"
    assert resolved.directory == bundle
    assert resolved.config_path == bundle / "config.json"
    assert resolved.weights_path == bundle / "model.safetensors"


def test_resolve_pretrained_verifies_checksums(
    tmp_path: Path,
    bundle_builder,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = bundle_builder(tmp_path)
    (bundle / "model.safetensors").write_bytes(b"tampered")
    _mock_hub(monkeypatch, bundle)
    with pytest.raises(ManifestError, match="Checksum mismatch"):
        resolver.resolve_pretrained("ORG/RadGraphIT-v1", revision="abc123")


def test_missing_remote_manifest_maps_to_model_not_found(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_hub(monkeypatch, tmp_path)
    with pytest.raises(ModelNotFoundError, match="manifest is mandatory"):
        resolver.resolve_pretrained("ORG/RadGraphIT-v1", revision="abc123")


def test_snapshot_failure_maps_to_model_not_found(
    tmp_path: Path,
    bundle_builder,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = bundle_builder(tmp_path)
    _mock_hub(monkeypatch, bundle)
    monkeypatch.setattr(
        resolver,
        "snapshot_download",
        lambda **kwargs: (_ for _ in ()).throw(OSError("network is unreachable")),
    )
    with pytest.raises(ModelNotFoundError, match="Could not download"):
        resolver.resolve_pretrained("ORG/RadGraphIT-v1", revision="abc123")


def test_resolve_local_missing_directory_raises(tmp_path: Path) -> None:
    with pytest.raises(ModelNotFoundError, match="does not exist"):
        resolver.resolve_local(tmp_path / "nope")


def test_resolve_local_verifies_bundle(tmp_path: Path, bundle_builder) -> None:
    bundle = bundle_builder(tmp_path)
    resolved = resolver.resolve_local(bundle)
    assert resolved.manifest.revision == "test-revision"
    assert resolved.directory == bundle


def test_resolve_local_rejects_bundle_without_manifest(tmp_path: Path, bundle_builder) -> None:
    bundle = bundle_builder(tmp_path, with_manifest=False)
    with pytest.raises(ManifestError, match="manifest not found"):
        resolver.resolve_local(bundle)


def test_resolve_local_bundle_in_model_subdir(tmp_path: Path, bundle_builder) -> None:
    root = bundle_builder(tmp_path, subdir="model")
    resolved = resolver.resolve_local(root)
    assert resolved.directory == tmp_path / "model"
    assert resolved.manifest.backend == "dygie_v2"
    assert resolved.config_path == tmp_path / "model" / "config.json"
    assert resolved.weights_path == tmp_path / "model" / "model.safetensors"


def test_resolve_pretrained_finds_manifest_bundle_in_subdir(
    tmp_path: Path,
    bundle_builder,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = bundle_builder(tmp_path, subdir="model")
    recorded: dict[str, object] = {}
    _mock_hub(monkeypatch, root, recorded=recorded)
    resolved = resolver.resolve_pretrained("ORG/RadGraphIT-v1", revision="abc123")
    assert resolved.directory == tmp_path / "model"
    assert resolved.manifest.backend == "dygie_v2"
    assert sorted(recorded["allow_patterns"]) == [
        "model/config.json",
        "model/model.safetensors",
        "model/model_manifest.json",
        "model/vocab.json",
    ]


def test_missing_bundle_raises(tmp_path: Path) -> None:
    (tmp_path / "unrelated.txt").write_text("not a bundle")
    with pytest.raises(ModelNotFoundError, match="No RadGraphIT model bundle"):
        resolver.resolve_local(tmp_path)


def test_package_version_uses_metadata_and_source_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(resolver, "version", lambda _name: "7.8.9")
    assert resolver._package_version() == "7.8.9"

    def missing(_name: str) -> str:
        raise resolver.PackageNotFoundError

    monkeypatch.setattr(resolver, "version", missing)
    from radgraphit import __version__

    assert resolver._package_version() == __version__


def test_bundle_shape_requires_weights_and_valid_encoder_config(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text('{"encoder": {}}')
    (tmp_path / "vocab.json").write_text("{}")
    assert resolver._looks_like_bundle(tmp_path) is False

    (tmp_path / "model.safetensors").write_bytes(b"weights")
    (tmp_path / "config.json").write_text("{not-json")
    assert resolver._looks_like_bundle(tmp_path) is False

    (tmp_path / "config.json").write_text("{}")
    assert resolver._looks_like_bundle(tmp_path) is False
