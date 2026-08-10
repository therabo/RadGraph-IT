"""The ``dygie_v2`` inference backend: a thin, torch-free-facing wrapper over the ported model.

Implements the :class:`~radgraphit.core.models.InferenceBackend` protocol: tokens in, raw scored
tuples out. All torch lives inside; nothing torch-shaped crosses ``infer``'s boundary.
"""

from __future__ import annotations

import torch

from ....core.errors import BackendError
from ....core.models import RawBackendOutput
from .model import DyGIEModel


class DyGIEv2Backend:
    """Runs a loaded :class:`DyGIEModel` on one report at a time."""

    def __init__(self, model: DyGIEModel, device: torch.device):
        self._model = model
        self._device = device

    @property
    def device(self) -> torch.device:
        return self._device

    def infer(self, tokens: tuple[str, ...]) -> RawBackendOutput:
        try:
            ner, relations = self._model.predict(tokens)
        except Exception as exc:
            raise BackendError(f"dygie_v2 inference failed: {exc}") from exc
        return RawBackendOutput(ner=tuple(ner), relations=tuple(relations))
