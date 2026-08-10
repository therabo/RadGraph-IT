"""CLI smoke tests for ``radgraphit predict`` (backend build mocked; no torch, no network)."""

from __future__ import annotations

import json
import runpy
import sys
from argparse import Namespace
from pathlib import Path

import pytest

from radgraphit import cli
from radgraphit.cli import _build_predictor, main
from radgraphit.core.models import RawBackendOutput


def _fake_backend_builder(resolved, device):
    class _Fake:
        def infer(self, tokens):
            return RawBackendOutput(
                ner=((0, 0, "Observation::definitely present", 1.0, 0.9),), relations=()
            )

    return _Fake()


def test_predict_from_argument(
    tmp_path: Path, bundle_builder, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    bundle = bundle_builder(tmp_path)
    monkeypatch.setattr("radgraphit.api._build_backend", _fake_backend_builder)

    exit_code = main(
        [
            "predict",
            "--model-dir",
            str(bundle),
            "--device",
            "cpu",
            "Nessuna evidenza di pneumotorace.",
        ]
    )
    assert exit_code == 0

    payload = json.loads(capsys.readouterr().out)
    assert list(payload.keys()) == ["0"]
    assert payload["0"]["entities"]["1"]["tokens"] == "Nessuna"


def test_predict_from_stdin(
    tmp_path: Path, bundle_builder, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    bundle = bundle_builder(tmp_path)
    monkeypatch.setattr("radgraphit.api._build_backend", _fake_backend_builder)
    monkeypatch.setattr("sys.stdin", _StringStdin("Referto uno.\nReferto due.\n"))

    exit_code = main(["predict", "--model-dir", str(bundle), "--device", "cpu"])
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert list(payload.keys()) == ["0", "1"]


def test_predict_from_input_file(
    tmp_path: Path, bundle_builder, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    bundle = bundle_builder(tmp_path / "bundle")
    reports = tmp_path / "reports.txt"
    reports.write_text("Referto uno.\nReferto due.\n", encoding="utf-8")
    monkeypatch.setattr("radgraphit.api._build_backend", _fake_backend_builder)
    exit_code = main(
        [
            "predict",
            "--model-dir",
            str(bundle),
            "--input-file",
            str(reports),
        ]
    )
    assert exit_code == 0
    assert list(json.loads(capsys.readouterr().out)) == ["0", "1"]


def test_invalid_input_returns_error_exit_code(
    tmp_path: Path, bundle_builder, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    bundle = bundle_builder(tmp_path)
    monkeypatch.setattr("radgraphit.api._build_backend", _fake_backend_builder)
    # A whitespace-only report is rejected by the domain validation -> exit code 1.
    exit_code = main(["predict", "--model-dir", str(bundle), "--device", "cpu", "   "])
    assert exit_code == 1
    assert "radgraphit:" in capsys.readouterr().err


def test_positional_reports_and_input_file_are_rejected(tmp_path: Path, capsys) -> None:
    reports = tmp_path / "reports.txt"
    reports.write_text("Referto da file.\n", encoding="utf-8")
    exit_code = main(["predict", "--input-file", str(reports), "Referto posizionale."])
    assert exit_code == 2
    assert "mutually exclusive" in capsys.readouterr().err


@pytest.mark.parametrize("option", ["--revision", "--cache-dir"])
def test_hub_options_with_local_model_are_rejected(
    option: str,
    tmp_path: Path,
    capsys,
) -> None:
    exit_code = main(["predict", "--model-dir", str(tmp_path), option, "value", "Referto."])
    assert exit_code == 2
    assert "cannot be used with --model-dir" in capsys.readouterr().err


class _StringStdin:
    def __init__(self, text: str):
        self._text = text

    def read(self) -> str:
        return self._text


def test_hub_predictor_forwards_default_and_explicit_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []
    sentinel = object()

    def fake_from_pretrained(**kwargs):
        calls.append(kwargs)
        return sentinel

    monkeypatch.setattr(cli.RadGraphIT, "from_pretrained", fake_from_pretrained)
    base = {
        "model_dir": None,
        "device": "cpu",
        "cache_dir": None,
        "model_id": None,
        "revision": None,
    }
    assert _build_predictor(Namespace(**base)) is sentinel
    assert (
        _build_predictor(
            Namespace(**{**base, "model_id": "org/model", "revision": "rev", "cache_dir": "/cache"})
        )
        is sentinel
    )
    assert calls == [
        {"device": "cpu", "cache_dir": None},
        {"device": "cpu", "cache_dir": "/cache", "model_id": "org/model", "revision": "rev"},
    ]


def test_no_report_and_unreadable_input_file_return_errors(
    capsys, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("sys.stdin", _StringStdin("\n  \n"))
    assert main(["predict"]) == 2
    assert "no report provided" in capsys.readouterr().err

    assert main(["predict", "--input-file", str(tmp_path / "missing.txt")]) == 1
    assert "radgraphit:" in capsys.readouterr().err


def test_unknown_command_defensive_branch(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeParser:
        def parse_args(self, _argv):
            return Namespace(command="unknown")

        def error(self, message: str) -> None:
            assert "unknown command" in message

    monkeypatch.setattr(cli, "_build_parser", FakeParser)
    assert main([]) == 2


def test_module_entrypoint_calls_main(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["radgraphit", "--version"])
    with (
        pytest.warns(RuntimeWarning, match="found in sys.modules"),
        pytest.raises(SystemExit, match="0"),
    ):
        runpy.run_module("radgraphit.cli", run_name="__main__")
