"""``model_manifest.json``: schema, parsing, and SHA-256 verification.

The manifest is versioned *inside the Hugging Face model repository*, alongside ``config.json``,
``vocab.json`` and the weights. It records what package/schema/backend the bundle targets, which
encoder it needs, the pinned revision, and a SHA-256 for every required file. The resolver verifies
it against the downloaded bytes **before** the model is built, so a corrupted or tampered download
fails loudly and early rather than as an obscure state-dict error.

Example ``model_manifest.json`` (see also ``scripts/publish_model.py``, which generates it):

.. code-block:: json

    {
      "manifest_version": 1,
      "package": "radgraphit",
      "min_package_version": "1.0.0",
      "schema": "radgraph-xl",
      "backend": "dygie_v2",
      "encoder": {
        "model_name": "IVN-RIN/medBIT-r3-plus",
        "revision": "<immutable encoder commit sha>",
        "max_length": 512
      },
      "revision": "<git-sha-or-tag of this bundle>",
      "weights_file": "model.safetensors",
      "files": {
        "config.json": {"sha256": "…"},
        "vocab.json": {"sha256": "…"},
        "model.safetensors": {"sha256": "…"}
      }
    }
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath

from packaging.version import InvalidVersion, Version

from ..core.errors import ManifestError
from . import registry

MANIFEST_VERSION = 1
_CHUNK = 1 << 20  # 1 MiB streaming reads, so large weight files don't load into memory to hash
_SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")


def sha256_file(path: Path) -> str:
    """Return the hex SHA-256 of a file, read in streaming chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_filename(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ManifestError(f"Manifest field {field} must be a non-empty filename.")
    posix = PurePosixPath(value)
    windows = PureWindowsPath(value)
    if (
        posix.is_absolute()
        or windows.is_absolute()
        or len(posix.parts) != 1
        or len(windows.parts) != 1
        or value in {".", ".."}
    ):
        raise ManifestError(
            f"Manifest field {field} must be a plain filename inside the bundle, got {value!r}."
        )
    return value


def _positive_int(value: object, *, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ManifestError(f"Manifest field {field} must be a positive integer.")
    if value <= 0:
        raise ManifestError(f"Manifest field {field} must be a positive integer.")
    return value


def _non_empty_string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ManifestError(f"Manifest field {field} must be a non-empty string.")
    return value


@dataclass(frozen=True, slots=True)
class ModelManifest:
    """Parsed, validated model manifest."""

    manifest_version: int
    package: str
    min_package_version: str
    schema: str
    backend: str
    encoder_model_name: str
    encoder_revision: str
    encoder_max_length: int
    revision: str
    weights_file: str
    files: dict[str, str]  # filename -> expected sha256

    @classmethod
    def from_path(cls, path: Path) -> ModelManifest:
        """Parse and structurally validate a ``model_manifest.json`` file."""
        if not path.is_file():
            raise ManifestError(
                f"Model manifest not found at {path}. A valid RadGraphIT bundle must contain "
                f"{registry.MANIFEST_FILENAME} alongside its config, vocabulary, and weights."
            )
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ManifestError(f"Could not read a valid JSON manifest at {path}: {exc}") from exc
        if not isinstance(raw, dict):
            raise ManifestError(f"Model manifest at {path} must be a JSON object.")

        try:
            encoder = raw["encoder"]
            files_raw = raw["files"]
            if not isinstance(encoder, dict):
                raise TypeError("'encoder' must be an object")
            if not isinstance(files_raw, dict) or not files_raw:
                raise TypeError("'files' must be a non-empty object")
            files: dict[str, str] = {}
            for name, metadata in files_raw.items():
                filename = _safe_filename(name, field="files key")
                if not isinstance(metadata, dict):
                    raise TypeError(f"metadata for {filename!r} must be an object")
                expected = metadata["sha256"]
                if not isinstance(expected, str) or not _SHA256.fullmatch(expected):
                    raise ValueError(f"sha256 for {filename!r} must contain exactly 64 hex digits")
                files[filename] = expected.lower()

            weights_file = _safe_filename(raw["weights_file"], field="weights_file")
            if weights_file not in registry.WEIGHT_FILENAMES:
                raise ValueError(
                    f"weights_file must be one of {', '.join(registry.WEIGHT_FILENAMES)}"
                )
            encoder_model_name = encoder["model_name"]
            if not isinstance(encoder_model_name, str) or not encoder_model_name.strip():
                raise ValueError("encoder.model_name must be a non-empty string")
            if "revision" in encoder:
                encoder_revision = encoder["revision"]
            else:
                encoder_revision = registry.ENCODER_REVISIONS.get(encoder_model_name)
            if not isinstance(encoder_revision, str) or not encoder_revision.strip():
                raise ValueError(
                    "encoder.revision must contain an immutable commit SHA or release tag"
                )
            manifest_version = raw["manifest_version"]
            if not isinstance(manifest_version, int) or isinstance(manifest_version, bool):
                raise ValueError("manifest_version must be an integer")

            manifest = cls(
                manifest_version=manifest_version,
                package=_non_empty_string(raw["package"], field="package"),
                min_package_version=_non_empty_string(
                    raw["min_package_version"], field="min_package_version"
                ),
                schema=_non_empty_string(raw["schema"], field="schema"),
                backend=_non_empty_string(raw["backend"], field="backend"),
                encoder_model_name=encoder_model_name,
                encoder_revision=encoder_revision,
                encoder_max_length=_positive_int(encoder["max_length"], field="encoder.max_length"),
                revision=_non_empty_string(raw["revision"], field="revision"),
                weights_file=weights_file,
                files=files,
            )
        except ManifestError:
            raise
        except (KeyError, TypeError, ValueError) as exc:
            raise ManifestError(
                f"Model manifest at {path} is missing or has a malformed field: {exc}"
            ) from exc
        required = {
            registry.CONFIG_FILENAME,
            registry.VOCAB_FILENAME,
            manifest.weights_file,
        }
        missing_checksums = required.difference(manifest.files)
        if missing_checksums:
            raise ManifestError(
                "Model manifest must provide SHA-256 checksums for every required bundle file; "
                f"missing: {', '.join(sorted(missing_checksums))}."
            )
        return manifest

    def check_compatibility(self, package_version: str) -> None:
        """Validate manifest/backend/schema/version compatibility with this package.

        Raises :class:`ManifestError` on any incompatibility, with a message naming the mismatch.
        """
        if self.manifest_version != MANIFEST_VERSION:
            if self.manifest_version > MANIFEST_VERSION:
                raise ManifestError(
                    f"Manifest version {self.manifest_version} is newer than this package supports "
                    f"({MANIFEST_VERSION}). Upgrade radgraphit."
                )
            raise ManifestError(
                f"Manifest version {self.manifest_version} is unsupported; expected "
                f"{MANIFEST_VERSION}."
            )
        if self.package != "radgraphit":
            raise ManifestError(f"Manifest targets package {self.package!r}, not 'radgraphit'.")
        if self.backend not in registry.SUPPORTED_BACKENDS:
            raise ManifestError(
                f"Manifest backend {self.backend!r} is not supported by this package "
                f"(supported: {sorted(registry.SUPPORTED_BACKENDS)})."
            )
        if self.schema not in registry.SUPPORTED_SCHEMAS:
            raise ManifestError(
                f"Manifest schema {self.schema!r} is not supported by this package "
                f"(supported: {sorted(registry.SUPPORTED_SCHEMAS)})."
            )
        try:
            installed = Version(package_version)
            required = Version(self.min_package_version)
        except InvalidVersion as exc:
            raise ManifestError(f"Manifest or package contains an invalid version: {exc}") from exc
        if installed < required:
            raise ManifestError(
                f"This model requires radgraphit >= {self.min_package_version}, but "
                f"{package_version} is installed. Upgrade radgraphit."
            )

    def verify_files(self, directory: Path) -> None:
        """Verify every file listed in the manifest exists in ``directory`` with the right SHA-256.

        Also checks that the manifest's encoder metadata agrees with ``config.json``.
        Raises :class:`ManifestError` naming the first inconsistency.
        """
        for filename, expected in self.files.items():
            file_path = directory / filename
            if not file_path.is_file():
                raise ManifestError(
                    f"Manifest lists required file {filename!r} but it is missing from {directory}."
                )
            try:
                actual = sha256_file(file_path)
            except OSError as exc:
                raise ManifestError(f"Could not hash manifest file {file_path}: {exc}") from exc
            if actual != expected:
                raise ManifestError(
                    f"Checksum mismatch for {filename!r} in {directory}: manifest expects "
                    f"{expected}, downloaded file is {actual}. The bundle is corrupted or has been "
                    "tampered with; delete the cache directory and re-download."
                )
        config_path = directory / registry.CONFIG_FILENAME
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ManifestError(
                f"Could not read a valid JSON config at {config_path}: {exc}"
            ) from exc
        if not isinstance(config, dict) or not isinstance(config.get("encoder"), dict):
            raise ManifestError(f"{config_path} must contain an 'encoder' object.")
        encoder = config["encoder"]
        if encoder.get("model_name") != self.encoder_model_name:
            raise ManifestError(
                "Manifest encoder.model_name does not match config.json: "
                f"{self.encoder_model_name!r} != {encoder.get('model_name')!r}."
            )
        try:
            config_max_length = _positive_int(
                encoder.get("max_length"), field="config encoder.max_length"
            )
        except ManifestError as exc:
            raise ManifestError(f"Invalid encoder configuration in {config_path}: {exc}") from exc
        if config_max_length != self.encoder_max_length:
            raise ManifestError(
                "Manifest encoder.max_length does not match config.json: "
                f"{self.encoder_max_length} != {config_max_length}."
            )

    def required_filenames(self) -> tuple[str, ...]:
        """All filenames the manifest declares (used to drive selective downloads)."""
        names = set(self.files) | {
            registry.MANIFEST_FILENAME,
            registry.CONFIG_FILENAME,
            registry.VOCAB_FILENAME,
            self.weights_file,
        }
        return tuple(sorted(names))
