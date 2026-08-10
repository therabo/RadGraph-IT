"""Acceptance test #3c — explicit device selection.

CPU must always work; an explicit CUDA request on a CUDA-less machine is an error, not a silent
fallback. CUDA-present branches are exercised by monkeypatching torch's availability so the test
runs on any machine (no GPU required).
"""

from __future__ import annotations

import pytest
import torch

from radgraphit.core.errors import DeviceError
from radgraphit.inference.backends.dygie_v2.device import resolve_device


def test_cpu_always_resolves() -> None:
    assert resolve_device("cpu") == torch.device("cpu")


def test_auto_without_cuda_falls_back_to_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device("auto") == torch.device("cpu")
    assert resolve_device(None) == torch.device("cpu")


def test_auto_with_cuda_prefers_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_device("auto") == torch.device("cuda")


def test_explicit_cuda_without_cuda_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(DeviceError, match="CUDA is not available"):
        resolve_device("cuda")
    with pytest.raises(DeviceError, match="CUDA is not available"):
        resolve_device("cuda:0")


def test_cuda_index_parsed_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 4)
    assert resolve_device("cuda:2") == torch.device("cuda:2")


def test_unindexed_cuda_defaults_to_zero_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_device("cuda") == torch.device("cuda:0")


def test_cuda_index_out_of_range_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    with pytest.raises(DeviceError, match="out of range"):
        resolve_device("cuda:3")


def test_non_numeric_cuda_index_raises_when_cuda_is_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    with pytest.raises(DeviceError, match="Invalid CUDA device index"):
        resolve_device("cuda:x")


@pytest.mark.parametrize("bad", ["gpu", "cuda:x", "mps ", "xpu:0"])
def test_unknown_device_string_raises(bad: str) -> None:
    with pytest.raises(DeviceError):
        resolve_device(bad)


def test_non_string_device_raises_domain_error() -> None:
    with pytest.raises(DeviceError, match="must be a string"):
        resolve_device(42)  # type: ignore[arg-type]
