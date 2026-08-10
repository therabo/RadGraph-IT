"""Explicit device selection with clear errors and a working CPU default.

Accepts ``"auto"`` (default), ``"cpu"``, ``"cuda"``, or ``"cuda:<index>"``. ``"auto"`` picks CUDA
when available and otherwise falls back to CPU. An explicit CUDA request when CUDA is unavailable
is an error (no silent fallback) — but CPU always works.
"""

from __future__ import annotations

import torch

from ....core.errors import DeviceError


def resolve_device(device: str | None) -> torch.device:
    """Resolve a device string to a concrete :class:`torch.device`.

    Raises :class:`DeviceError` for an unknown string or a CUDA request on a machine without CUDA.
    """
    if device is None or device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not isinstance(device, str):
        raise DeviceError(f"device must be a string or None, got {type(device).__name__}.")
    if device == "cpu":
        return torch.device("cpu")
    if device == "cuda" or device.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise DeviceError(
                f"Device {device!r} was requested but CUDA is not available. Use device='cpu' or "
                "device='auto' (which falls back to CPU)."
            )
        index = 0
        if ":" in device:
            suffix = device.split(":", 1)[1]
            if not suffix.isdigit():
                raise DeviceError(f"Invalid CUDA device index in {device!r}.")
            index = int(suffix)
            if index >= torch.cuda.device_count():
                raise DeviceError(
                    f"CUDA device index {index} out of range: only {torch.cuda.device_count()} "
                    "CUDA device(s) visible."
                )
        return torch.device(f"cuda:{index}")
    raise DeviceError(
        f"Unknown device {device!r}. Expected one of: 'auto', 'cpu', 'cuda', 'cuda:<index>'."
    )
