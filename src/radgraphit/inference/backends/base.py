"""The backend-facing surface of the inference protocol.

A backend is anything that can turn one report's tokens into raw scored predictions. The protocol
itself is defined in :mod:`radgraphit.core.models` (so it stays torch-free and importable
everywhere); it is re-exported here as the canonical import for backend authors and for the
service. Concrete backends (e.g. ``dygie_v2``) live in sibling packages and may depend on torch.
"""

from __future__ import annotations

from ...core.models import InferenceBackend, RawBackendOutput

__all__ = ["InferenceBackend", "RawBackendOutput"]
