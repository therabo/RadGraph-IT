#!/usr/bin/env python3
"""Prepare (and optionally publish) a RadGraphIT model bundle for the Hugging Face Hub.

This script is for the **maintainer**, to run once the trained v2 weights are ready. It does two
things:

1. **Generate ``model_manifest.json``** next to a bundle directory that already contains
   ``config.json``, ``vocab.json`` and the weights (``best.pt`` or ``model.safetensors``). The
   manifest records the package/schema/backend compatibility, the encoder, the revision label, and a
   SHA-256 for every required file — exactly what :mod:`radgraphit.artifacts.manifest` verifies
   before building the model.

2. **Optionally upload** the bundle to the Hub — but only when ``--push`` is passed explicitly. By
   default the script writes the manifest and prints the upload commands without performing any
   network action.

It is intentionally torch-free: it hashes bytes and reads ``config.json``; it never loads the model.

Usage
-----
Generate the manifest only (no network)::

    python scripts/publish_model.py /path/to/bundle \
        --min-package-version 1.0.0 --revision v1

Then, when ready, create the repo and upload (requires ``huggingface-cli login``)::

    python scripts/publish_model.py /path/to/bundle --repo-id ORGANIZATION/Radgraph-IT --push

After the upload, edit **src/radgraphit/artifacts/registry.py**:

    DEFAULT_MODEL_ID = "ORGANIZATION/Radgraph-IT"
    DEFAULT_REVISION = "<the commit sha printed by --push>"
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

MANIFEST_FILENAME = "model_manifest.json"
CONFIG_FILENAME = "config.json"
VOCAB_FILENAME = "vocab.json"
WEIGHT_CANDIDATES = ("model.safetensors", "best.pt")  # preference order (safetensors preferred)
ENCODER_REVISIONS = {
    "IVN-RIN/medBIT-r3-plus": "0f1c91e551551869a72935962b54c1f5247333a4",
}


def _lfs_pointer_oid(path: Path) -> str | None:
    """If ``path`` is a git-LFS pointer file, return the SHA-256 oid it declares, else ``None``.

    A Hugging Face repo cloned without ``git lfs pull`` leaves large files (the weights) as small
    pointer files whose declared ``oid sha256:`` *is* the SHA-256 of the real content — so the
    manifest can be generated correctly without materializing the weights.
    """
    try:
        if path.stat().st_size > 300:  # pointers are tiny; real weight files are not
            return None
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    if not text.startswith("version https://git-lfs.github.com/spec/v1"):
        return None
    for line in text.splitlines():
        if line.startswith("oid sha256:"):
            return line.split("oid sha256:", 1)[1].strip()
    return None


def _sha256(path: Path) -> str:
    """SHA-256 of a file's real content, transparently using a git-LFS pointer's declared oid when
    the file has not been materialized (see :func:`_lfs_pointer_oid`)."""
    oid = _lfs_pointer_oid(path)
    if oid is not None:
        return oid
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _find_weights(bundle: Path) -> str:
    for candidate in WEIGHT_CANDIDATES:
        if (bundle / candidate).is_file():
            return candidate
    raise SystemExit(
        f"error: no weights file found in {bundle} (looked for {', '.join(WEIGHT_CANDIDATES)})."
    )


def build_manifest(
    bundle: Path,
    *,
    revision: str,
    min_package_version: str,
    encoder_revision: str | None = None,
) -> dict:
    config_path = bundle / CONFIG_FILENAME
    vocab_path = bundle / VOCAB_FILENAME
    for required in (config_path, vocab_path):
        if not required.is_file():
            raise SystemExit(f"error: required file missing: {required}")

    weights_file = _find_weights(bundle)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    encoder = config.get("encoder", {})
    encoder_model_name = encoder.get("model_name")
    resolved_encoder_revision = encoder_revision or ENCODER_REVISIONS.get(encoder_model_name)
    if not resolved_encoder_revision:
        raise SystemExit(
            "error: no immutable encoder revision configured; pass --encoder-revision."
        )

    files = [CONFIG_FILENAME, VOCAB_FILENAME, weights_file]
    manifest = {
        "manifest_version": 1,
        "package": "radgraphit",
        "min_package_version": min_package_version,
        "schema": "radgraph-xl",
        "backend": "dygie_v2",
        "encoder": {
            "model_name": encoder_model_name,
            "revision": resolved_encoder_revision,
            "max_length": encoder.get("max_length"),
        },
        "revision": revision,
        "weights_file": weights_file,
        "files": {name: {"sha256": _sha256(bundle / name)} for name in files},
    }
    return manifest


def _print_upload_instructions(bundle: Path, repo_id: str | None) -> None:
    target = repo_id or "ORGANIZATION/Radgraph-IT"
    print("\nManifest written. To publish (only when ready), run:")
    print(f"    python scripts/publish_model.py {bundle} --repo-id {target} --push")
    print("or manually with the Hugging Face CLI:")
    print(f"    huggingface-cli repo create {target} --type model")
    print(f"    huggingface-cli upload {target} {bundle} .")
    print(
        "\nThen set DEFAULT_MODEL_ID / DEFAULT_REVISION in "
        "src/radgraphit/artifacts/registry.py to the published id and commit sha."
    )


def _push(bundle: Path, repo_id: str, private: bool) -> None:
    from huggingface_hub import HfApi  # imported only when actually pushing

    api = HfApi()
    print(f"Creating repo {repo_id} (private={private}) if needed…")
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    print(f"Uploading {bundle} → {repo_id}…")
    commit = api.upload_folder(repo_id=repo_id, repo_type="model", folder_path=str(bundle))
    print(f"Uploaded. Commit: {commit}")
    print(
        "\nNow pin src/radgraphit/artifacts/registry.py:\n"
        f'    DEFAULT_MODEL_ID = "{repo_id}"\n'
        '    DEFAULT_REVISION = "<the commit sha above>"'
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "bundle", type=Path, help="Directory with config.json, vocab.json, weights."
    )
    parser.add_argument(
        "--revision", default="v1", help="Revision label recorded in the manifest (default: v1)."
    )
    parser.add_argument(
        "--min-package-version",
        default="1.0.0",
        help="Minimum radgraphit version compatible with this bundle (default: 1.0.0).",
    )
    parser.add_argument(
        "--encoder-revision",
        help="Immutable commit SHA or release tag for the encoder repository.",
    )
    parser.add_argument("--repo-id", help="Target Hugging Face repo id (for --push).")
    parser.add_argument(
        "--push",
        action="store_true",
        help="Actually create the repo and upload. Off by default (no network).",
    )
    parser.add_argument("--private", action="store_true", help="Create the Hub repo as private.")
    args = parser.parse_args(argv)

    bundle: Path = args.bundle
    if not bundle.is_dir():
        raise SystemExit(f"error: {bundle} is not a directory.")

    manifest = build_manifest(
        bundle,
        revision=args.revision,
        min_package_version=args.min_package_version,
        encoder_revision=args.encoder_revision,
    )
    (bundle / MANIFEST_FILENAME).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {bundle / MANIFEST_FILENAME}")
    print(json.dumps(manifest, indent=2))

    if args.push:
        if not args.repo_id:
            raise SystemExit("error: --push requires --repo-id.")
        _push(bundle, args.repo_id, args.private)
    else:
        _print_upload_instructions(bundle, args.repo_id)
    return 0


if __name__ == "__main__":
    sys.exit(main())
