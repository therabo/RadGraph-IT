"""Resolve a model bundle to a verified local directory.

Two entry points:

* :func:`resolve_pretrained` — the standard path. Downloads only the files declared by the bundle's
  manifest from an explicitly pinned Hugging Face Hub revision, caches them, and verifies
  compatibility + SHA-256 before returning.
* :func:`resolve_local` — for development, offline testing, and diagnostics. Points at an existing
  local directory and requires the exact same manifest verification.

Both return a :class:`ResolvedModel` (a verified directory + parsed manifest). Neither ever falls
back to unverified loading: a missing manifest, an unreachable repository, or a bad checksum raises
a specific, actionable error.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path, PurePosixPath

from huggingface_hub import hf_hub_download, snapshot_download

from ..core.errors import ModelNotConfiguredError, ModelNotFoundError
from . import registry
from .manifest import ModelManifest

logger = logging.getLogger("radgraphit")


@dataclass(frozen=True, slots=True)
class ResolvedModel:
    """A verified, on-disk model bundle."""

    directory: Path
    manifest: ModelManifest

    @property
    def config_path(self) -> Path:
        return self.directory / registry.CONFIG_FILENAME

    @property
    def vocab_path(self) -> Path:
        return self.directory / registry.VOCAB_FILENAME

    @property
    def weights_path(self) -> Path:
        return self.directory / self.manifest.weights_file


def _package_version() -> str:
    try:
        return version("radgraphit")
    except PackageNotFoundError:  # not installed (e.g. running from a source checkout)
        from .. import __version__

        return __version__


_BUNDLE_SUBDIRS = ("", "model")  # search the snapshot root first, then a conventional model/ subdir


def _looks_like_bundle(directory: Path) -> bool:
    """Whether ``directory`` holds a dygie training bundle (config + vocab + weights).

    The ``config.json`` must carry an ``"encoder"`` block, which distinguishes the training config
    from an unrelated Hugging Face ``config.json`` that may sit at the repo root next to custom
    modeling code (as in the published RadGraphIT repo, whose bundle lives under ``model/``).
    """
    if not directory.is_dir():
        return False
    config_path = directory / registry.CONFIG_FILENAME
    if not config_path.is_file() or not (directory / registry.VOCAB_FILENAME).is_file():
        return False
    if not any((directory / weights).is_file() for weights in registry.WEIGHT_FILENAMES):
        return False
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, UnicodeError):
        return False
    return isinstance(config, dict) and "encoder" in config


def _locate_bundle(root: Path) -> Path:
    """Return the directory holding the dygie bundle (``root`` or its ``model/`` subdir)."""
    for sub in _BUNDLE_SUBDIRS:
        candidate = root / sub if sub else root
        if _looks_like_bundle(candidate):
            return candidate
    raise ModelNotFoundError(
        f"No RadGraphIT model bundle found under {root} (looked in {root} and {root / 'model'}). "
        f"A bundle must contain {registry.CONFIG_FILENAME} (with an 'encoder' block), "
        f"{registry.VOCAB_FILENAME}, and a weights file ({' or '.join(registry.WEIGHT_FILENAMES)})."
    )


def _verify(directory: Path) -> ResolvedModel:
    """Locate a manifest-backed bundle within ``directory`` and validate it completely."""
    bundle = _locate_bundle(directory)
    manifest_path = bundle / registry.MANIFEST_FILENAME
    manifest = ModelManifest.from_path(manifest_path)
    manifest.check_compatibility(_package_version())
    manifest.verify_files(bundle)
    logger.debug("Verified model bundle at %s (revision %s)", bundle, manifest.revision)
    return ResolvedModel(directory=bundle, manifest=manifest)


def resolve_local(path: str | Path) -> ResolvedModel:
    """Resolve and verify a model bundle in an existing local directory.

    Raises :class:`ModelNotFoundError` if the directory does not exist, and :class:`ManifestError`
    (via :func:`_verify`) if its required manifest is missing, incompatible, or fails verification.
    """
    directory = Path(path).expanduser()
    if not directory.is_dir():
        raise ModelNotFoundError(
            f"Local model directory {directory} does not exist or is not a directory. Point "
            "RadGraphIT.from_local(...) at a directory containing model_manifest.json, "
            "config.json, vocab.json and the declared weights file, or at a parent holding them "
            "under model/."
        )
    return _verify(directory)


def resolve_pretrained(
    model_id: str,
    *,
    revision: str,
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
) -> ResolvedModel:
    """Download (or reuse cached) the model bundle from the Hub and verify it.

    ``model_id`` and ``revision`` must be non-empty. Revisions should be immutable commit SHAs or
    tags; the package default is pinned to an exact commit.

    The default RadGraphIT model is a **public, ungated** Hub repository, so no authentication is
    required — anyone can download it with a plain ``pip install``.

    Raises :class:`ModelNotFoundError` if the download fails for any reason (unknown repo, network,
    offline-with-no-cache, or a repository that is private/gated instead of public).
    """
    if not isinstance(model_id, str) or not model_id.strip():
        raise ModelNotConfiguredError("model_id must be a non-empty Hugging Face repository id.")
    if not isinstance(revision, str) or not revision.strip():
        raise ModelNotConfiguredError(
            "revision must be a non-empty immutable commit SHA or release tag."
        )

    manifest_locations = (
        registry.MANIFEST_FILENAME,
        f"model/{registry.MANIFEST_FILENAME}",
    )
    manifest_path: Path | None = None
    manifest_location: str | None = None
    manifest_errors: list[str] = []
    for candidate in manifest_locations:
        try:
            downloaded = hf_hub_download(
                repo_id=model_id,
                filename=candidate,
                revision=revision,
                cache_dir=str(cache_dir) if cache_dir is not None else None,
                local_files_only=local_files_only,
            )
        except Exception as exc:
            manifest_errors.append(f"{candidate}: {exc}")
            continue
        manifest_path = Path(downloaded)
        manifest_location = candidate
        break
    if manifest_path is None or manifest_location is None:
        details = "; ".join(manifest_errors)
        raise ModelNotFoundError(
            f"Could not find {registry.MANIFEST_FILENAME} for model {model_id!r} at revision "
            f"{revision!r}. A verified manifest is mandatory. Details: {details}"
        )

    manifest = ModelManifest.from_path(manifest_path)
    manifest.check_compatibility(_package_version())
    bundle_prefix = PurePosixPath(manifest_location).parent
    prefix = "" if str(bundle_prefix) == "." else f"{bundle_prefix.as_posix()}/"
    allow_patterns = [f"{prefix}{name}" for name in manifest.required_filenames()]
    try:
        local_dir = snapshot_download(
            repo_id=model_id,
            revision=revision,
            cache_dir=str(cache_dir) if cache_dir is not None else None,
            local_files_only=local_files_only,
            allow_patterns=allow_patterns,
        )
    except Exception as exc:
        raise ModelNotFoundError(
            f"Could not download model {model_id!r} at revision {revision!r} from the Hugging "
            f"Face Hub: {exc}. Check the repository id, the revision, your network connection, and "
            "that the repository is public and ungated."
        ) from exc
    return _verify(Path(local_dir))
