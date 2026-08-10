"""Acceptance test #3b — manifest parsing, compatibility, and SHA-256 verification.

No torch and no network: the weight file is dummy bytes, because the manifest layer only ever
hashes bytes and checks metadata — it never loads the model.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from radgraphit.artifacts import manifest as manifest_module
from radgraphit.artifacts.manifest import ModelManifest, sha256_file
from radgraphit.core.errors import ManifestError

GOOD_FILES = {
    "config.json": (b'{"encoder": {"model_name": "IVN-RIN/medBIT-r3-plus", "max_length": 512}}'),
    "vocab.json": b'{"radgraph-it__ner_labels": {"": 0}}',
    "best.pt": b"\x00\x01\x02 not-real-weights",
}


def write_bundle(directory: Path, *, files: dict[str, bytes] | None = None, **manifest_overrides):
    """Write a bundle (files + a manifest with matching checksums) and return the directory."""
    files = files if files is not None else GOOD_FILES
    for name, content in files.items():
        (directory / name).write_bytes(content)
    manifest = {
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
        "weights_file": "best.pt",
        "files": {
            name: {"sha256": hashlib.sha256(content).hexdigest()} for name, content in files.items()
        },
    }
    manifest.update(manifest_overrides)
    (directory / "model_manifest.json").write_text(json.dumps(manifest))
    return directory


def test_parse_and_verify_good_bundle(tmp_path: Path) -> None:
    write_bundle(tmp_path)
    manifest = ModelManifest.from_path(tmp_path / "model_manifest.json")
    assert manifest.backend == "dygie_v2"
    assert manifest.encoder_model_name == "IVN-RIN/medBIT-r3-plus"
    assert manifest.encoder_revision == "0f1c91e551551869a72935962b54c1f5247333a4"
    assert manifest.weights_file == "best.pt"
    manifest.check_compatibility("1.0.0")
    manifest.verify_files(tmp_path)  # does not raise


def test_missing_manifest_raises(tmp_path: Path) -> None:
    with pytest.raises(ManifestError, match="not found"):
        ModelManifest.from_path(tmp_path / "model_manifest.json")


def test_malformed_manifest_missing_field(tmp_path: Path) -> None:
    (tmp_path / "model_manifest.json").write_text('{"package": "radgraphit"}')
    with pytest.raises(ManifestError, match="missing or has a malformed field"):
        ModelManifest.from_path(tmp_path / "model_manifest.json")


def test_checksum_mismatch_is_detected(tmp_path: Path) -> None:
    write_bundle(tmp_path)
    (tmp_path / "best.pt").write_bytes(b"corrupted-after-manifest")
    manifest = ModelManifest.from_path(tmp_path / "model_manifest.json")
    with pytest.raises(ManifestError, match="Checksum mismatch"):
        manifest.verify_files(tmp_path)


def test_missing_listed_file_is_detected(tmp_path: Path) -> None:
    write_bundle(tmp_path)
    (tmp_path / "best.pt").unlink()
    manifest = ModelManifest.from_path(tmp_path / "model_manifest.json")
    with pytest.raises(ManifestError, match="missing"):
        manifest.verify_files(tmp_path)


def test_unsupported_backend_rejected(tmp_path: Path) -> None:
    write_bundle(tmp_path, backend="some_other_backend")
    manifest = ModelManifest.from_path(tmp_path / "model_manifest.json")
    with pytest.raises(ManifestError, match="backend"):
        manifest.check_compatibility("1.0.0")


def test_unsupported_schema_rejected(tmp_path: Path) -> None:
    write_bundle(tmp_path, schema="radgraph-v1")
    manifest = ModelManifest.from_path(tmp_path / "model_manifest.json")
    with pytest.raises(ManifestError, match="schema"):
        manifest.check_compatibility("1.0.0")


def test_package_version_too_old_rejected(tmp_path: Path) -> None:
    write_bundle(tmp_path, min_package_version="9.9.9")
    manifest = ModelManifest.from_path(tmp_path / "model_manifest.json")
    with pytest.raises(ManifestError, match="requires radgraphit >="):
        manifest.check_compatibility("1.0.0")


def test_newer_manifest_version_rejected(tmp_path: Path) -> None:
    write_bundle(tmp_path, manifest_version=999)
    manifest = ModelManifest.from_path(tmp_path / "model_manifest.json")
    with pytest.raises(ManifestError, match="newer than this package supports"):
        manifest.check_compatibility("1.0.0")


def test_older_manifest_version_rejected(tmp_path: Path) -> None:
    write_bundle(tmp_path, manifest_version=0)
    manifest = ModelManifest.from_path(tmp_path / "model_manifest.json")
    with pytest.raises(ManifestError, match="unsupported"):
        manifest.check_compatibility("1.0.0")


def test_pep440_prerelease_does_not_satisfy_final_requirement(tmp_path: Path) -> None:
    write_bundle(tmp_path, min_package_version="1.0.0")
    manifest = ModelManifest.from_path(tmp_path / "model_manifest.json")
    with pytest.raises(ManifestError, match="requires radgraphit >="):
        manifest.check_compatibility("1.0.0rc1")


def test_manifest_requires_checksums_for_every_bundle_file(tmp_path: Path) -> None:
    write_bundle(tmp_path)
    path = tmp_path / "model_manifest.json"
    payload = json.loads(path.read_text())
    del payload["files"]["best.pt"]
    path.write_text(json.dumps(payload))
    with pytest.raises(ManifestError, match=r"missing: best\.pt"):
        ModelManifest.from_path(path)


@pytest.mark.parametrize("unsafe", ["../best.pt", "/tmp/best.pt", r"..\best.pt"])
def test_manifest_rejects_unsafe_filenames(tmp_path: Path, unsafe: str) -> None:
    write_bundle(tmp_path)
    path = tmp_path / "model_manifest.json"
    payload = json.loads(path.read_text())
    payload["weights_file"] = unsafe
    with pytest.raises(ManifestError, match="plain filename"):
        path.write_text(json.dumps(payload))
        ModelManifest.from_path(path)


def test_manifest_rejects_invalid_checksum_shape(tmp_path: Path) -> None:
    write_bundle(tmp_path)
    path = tmp_path / "model_manifest.json"
    payload = json.loads(path.read_text())
    payload["files"]["best.pt"]["sha256"] = "not-a-sha"
    path.write_text(json.dumps(payload))
    with pytest.raises(ManifestError, match="64 hex digits"):
        ModelManifest.from_path(path)


def test_manifest_detects_encoder_config_mismatch(tmp_path: Path) -> None:
    write_bundle(tmp_path)
    config_path = tmp_path / "config.json"
    config_path.write_text('{"encoder": {"model_name": "different/model", "max_length": 512}}')
    manifest_path = tmp_path / "model_manifest.json"
    payload = json.loads(manifest_path.read_text())
    payload["files"]["config.json"]["sha256"] = sha256_file(config_path)
    manifest_path.write_text(json.dumps(payload))
    manifest = ModelManifest.from_path(manifest_path)
    with pytest.raises(ManifestError, match=r"does not match config\.json"):
        manifest.verify_files(tmp_path)


def test_sha256_file_streaming_matches_hashlib(tmp_path: Path) -> None:
    payload = b"x" * (3 * (1 << 20) + 7)  # spans multiple 1 MiB chunks
    (tmp_path / "blob").write_bytes(payload)
    assert sha256_file(tmp_path / "blob") == hashlib.sha256(payload).hexdigest()


def test_manifest_json_must_be_readable_and_an_object(tmp_path: Path) -> None:
    path = tmp_path / "model_manifest.json"
    path.write_text("{not-json")
    with pytest.raises(ManifestError, match="valid JSON manifest"):
        ModelManifest.from_path(path)

    path.write_text("[]")
    with pytest.raises(ManifestError, match="must be a JSON object"):
        ModelManifest.from_path(path)


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("encoder-not-object", "encoder.*object"),
        ("files-empty", "files.*non-empty object"),
        ("metadata-not-object", "metadata"),
        ("unsupported-weights", "weights_file must be one of"),
        ("empty-model-name", "encoder.model_name"),
        ("unknown-encoder-without-revision", "encoder.revision"),
        ("boolean-manifest-version", "manifest_version"),
        ("blank-package", "package"),
        ("non-integer-max-length", "encoder.max_length"),
        ("zero-max-length", "encoder.max_length"),
    ],
)
def test_manifest_rejects_each_malformed_field(tmp_path: Path, case: str, message: str) -> None:
    write_bundle(tmp_path)
    path = tmp_path / "model_manifest.json"
    payload = json.loads(path.read_text())
    if case == "encoder-not-object":
        payload["encoder"] = []
    elif case == "files-empty":
        payload["files"] = {}
    elif case == "metadata-not-object":
        payload["files"]["best.pt"] = "not-an-object"
    elif case == "unsupported-weights":
        payload["weights_file"] = "weights.bin"
    elif case == "empty-model-name":
        payload["encoder"]["model_name"] = " "
    elif case == "unknown-encoder-without-revision":
        payload["encoder"] = {"model_name": "unknown/model", "max_length": 512}
    elif case == "boolean-manifest-version":
        payload["manifest_version"] = True
    elif case == "blank-package":
        payload["package"] = " "
    elif case == "non-integer-max-length":
        payload["encoder"]["max_length"] = "512"
    elif case == "zero-max-length":
        payload["encoder"]["max_length"] = 0
    path.write_text(json.dumps(payload))
    with pytest.raises(ManifestError, match=message):
        ModelManifest.from_path(path)


def test_manifest_uses_registered_encoder_revision_when_omitted(tmp_path: Path) -> None:
    write_bundle(tmp_path)
    path = tmp_path / "model_manifest.json"
    payload = json.loads(path.read_text())
    del payload["encoder"]["revision"]
    path.write_text(json.dumps(payload))
    manifest = ModelManifest.from_path(path)
    assert manifest.encoder_revision == "0f1c91e551551869a72935962b54c1f5247333a4"


def test_scalar_manifest_validators_reject_non_filename_and_non_string() -> None:
    with pytest.raises(ManifestError, match="non-empty filename"):
        manifest_module._safe_filename(None, field="test")
    with pytest.raises(ManifestError, match="non-empty string"):
        manifest_module._non_empty_string(None, field="test")


def test_package_and_invalid_versions_are_rejected(tmp_path: Path) -> None:
    write_bundle(tmp_path, package="other-package")
    manifest = ModelManifest.from_path(tmp_path / "model_manifest.json")
    with pytest.raises(ManifestError, match="targets package"):
        manifest.check_compatibility("1.0.0")

    write_bundle(tmp_path, min_package_version="not-a-version")
    manifest = ModelManifest.from_path(tmp_path / "model_manifest.json")
    with pytest.raises(ManifestError, match="invalid version"):
        manifest.check_compatibility("1.0.0")


def test_hashing_io_error_is_wrapped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    write_bundle(tmp_path)
    manifest = ModelManifest.from_path(tmp_path / "model_manifest.json")

    def fail_hash(_path: Path) -> str:
        raise OSError("disk failure")

    monkeypatch.setattr(manifest_module, "sha256_file", fail_hash)
    with pytest.raises(ManifestError, match="Could not hash"):
        manifest.verify_files(tmp_path)


@pytest.mark.parametrize(
    ("config", "message"),
    [
        (b"{not-json", "valid JSON config"),
        (b"[]", "must contain an 'encoder' object"),
        (b'{"encoder": []}', "must contain an 'encoder' object"),
        (
            b'{"encoder": {"model_name": "IVN-RIN/medBIT-r3-plus", "max_length": 0}}',
            "Invalid encoder configuration",
        ),
        (
            b'{"encoder": {"model_name": "IVN-RIN/medBIT-r3-plus", "max_length": 256}}',
            "max_length does not match",
        ),
    ],
)
def test_verified_config_metadata_must_be_valid_and_consistent(
    tmp_path: Path, config: bytes, message: str
) -> None:
    files = {**GOOD_FILES, "config.json": config}
    write_bundle(tmp_path, files=files)
    manifest = ModelManifest.from_path(tmp_path / "model_manifest.json")
    with pytest.raises(ManifestError, match=message):
        manifest.verify_files(tmp_path)
