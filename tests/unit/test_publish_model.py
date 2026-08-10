"""Unit tests for scripts/publish_model.py — manifest generation, incl. git-LFS pointer handling.

The maintainer script is not part of the importable package, so it is loaded by path. The git-LFS
logic matters because a Hugging Face repo cloned without ``git lfs pull`` leaves the weights as
small pointer files whose declared ``oid`` is the real content's SHA-256.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

from radgraphit.artifacts import registry

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "publish_model.py"
_OID = "987fc75edf3ac78d1a0f05b8260f832c16ccd0cb6d6612432b2c0dbdf00a757a"
_LFS_POINTER = f"version https://git-lfs.github.com/spec/v1\noid sha256:{_OID}\nsize 444754251\n"


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("publish_model", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


publish_model = _load_script()


def test_publisher_encoder_pins_match_runtime_registry() -> None:
    assert publish_model.ENCODER_REVISIONS == registry.ENCODER_REVISIONS


def test_lfs_pointer_oid_reads_declared_hash(tmp_path: Path) -> None:
    pointer = tmp_path / "best.pt"
    pointer.write_text(_LFS_POINTER)
    assert publish_model._lfs_pointer_oid(pointer) == _OID


def test_lfs_pointer_oid_none_for_real_file(tmp_path: Path) -> None:
    real = tmp_path / "config.json"
    real.write_text('{"encoder": {}}')
    assert publish_model._lfs_pointer_oid(real) is None


def test_sha256_uses_pointer_oid_not_pointer_bytes(tmp_path: Path) -> None:
    pointer = tmp_path / "best.pt"
    pointer.write_text(_LFS_POINTER)
    assert publish_model._sha256(pointer) == _OID


def test_build_manifest_from_unpulled_lfs_clone(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(
        '{"encoder": {"model_name": "IVN-RIN/medBIT-r3-plus", "max_length": 512}}'
    )
    (tmp_path / "vocab.json").write_text('{"radgraph-it__ner_labels": {"": 0}}')
    (tmp_path / "best.pt").write_text(_LFS_POINTER)  # un-pulled LFS pointer, not real weights

    manifest = publish_model.build_manifest(tmp_path, revision="v1", min_package_version="1.0.0")

    assert manifest["backend"] == "dygie_v2"
    assert manifest["schema"] == "radgraph-xl"
    assert manifest["weights_file"] == "best.pt"
    assert manifest["files"]["best.pt"]["sha256"] == _OID
    assert manifest["encoder"]["model_name"] == "IVN-RIN/medBIT-r3-plus"
    assert manifest["encoder"]["revision"] == "0f1c91e551551869a72935962b54c1f5247333a4"
