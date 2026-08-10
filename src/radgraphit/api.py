"""The public facade: :class:`RadGraphIT`.

Small, typed, stable, and free of import-time side effects. Constructing a ``RadGraphIT`` performs
no network I/O; the model artifact is resolved and loaded lazily on first ``predict`` (or an
explicit :meth:`load`). Importing this module does not import torch or huggingface_hub.

Usage::

    from radgraphit import RadGraphIT

    predictor = RadGraphIT.from_local("/path/to/model", device="auto")
    annotations = predictor.predict([
        "Non si evidenziano segni di pneumotorace dopo la rimozione del drenaggio toracico."
    ])

The standard Hub-backed path is::

    predictor = RadGraphIT(device="auto")  # or RadGraphIT.from_pretrained(...)
    annotations = predictor.predict("Nessuna evidenza di pneumotorace.")
"""

from __future__ import annotations

import functools
import logging
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

from .artifacts import registry
from .core.errors import BackendError
from .core.models import InferenceBackend, ReportPrediction
from .inference.service import InferenceService

if TYPE_CHECKING:
    from .artifacts.resolver import ResolvedModel

logger = logging.getLogger("radgraphit")


def _resolve_pretrained(
    model_id: str, revision: str, cache_dir: str | None, local_files_only: bool
) -> ResolvedModel:
    # Imported lazily so `import radgraphit` never pulls in huggingface_hub.
    from .artifacts import resolver

    return resolver.resolve_pretrained(
        model_id, revision=revision, cache_dir=cache_dir, local_files_only=local_files_only
    )


def _resolve_local(path: str) -> ResolvedModel:
    from .artifacts import resolver

    return resolver.resolve_local(path)


def _build_backend(resolved: ResolvedModel, device: str) -> InferenceBackend:
    backend_name = resolved.manifest.backend
    if backend_name == "dygie_v2":
        # Imported lazily so torch/transformers load only when a model is actually built.
        from .inference.backends.dygie_v2 import load_backend

        return load_backend(
            config_path=resolved.config_path,
            vocab_path=resolved.vocab_path,
            weights_path=resolved.weights_path,
            device=device,
            encoder_revision=resolved.manifest.encoder_revision,
        )
    raise BackendError(
        f"Model manifest requests backend {backend_name!r}, which this version of radgraphit does "
        f"not support (supported: {sorted(registry.SUPPORTED_BACKENDS)})."
    )


class RadGraphIT:
    """Predict RadGraph-XL graphs for Italian radiology reports."""

    def __init__(
        self,
        *,
        device: str = "auto",
        model_id: str | None = None,
        revision: str | None = None,
        cache_dir: str | None = None,
        local_files_only: bool = False,
    ):
        """Configure a predictor backed by the default (Hugging Face Hub) model.

        No network I/O happens here — the artifact is downloaded and the model built lazily on the
        first :meth:`predict`. Passing ``model_id``/``revision`` overrides the registry defaults.
        The default model is a public, ungated Hub repository, so no authentication is required.
        """
        self._device = device
        self._service: InferenceService | None = None
        resolved_model_id = model_id if model_id is not None else registry.DEFAULT_MODEL_ID
        resolved_revision = revision if revision is not None else registry.DEFAULT_REVISION
        self._resolve: Callable[[], ResolvedModel] = functools.partial(
            _resolve_pretrained, resolved_model_id, resolved_revision, cache_dir, local_files_only
        )

    @classmethod
    def from_pretrained(
        cls,
        model_id: str = registry.DEFAULT_MODEL_ID,
        *,
        revision: str = registry.DEFAULT_REVISION,
        cache_dir: str | None = None,
        device: str = "auto",
        local_files_only: bool = False,
    ) -> RadGraphIT:
        """Standard path: resolve the model from the Hugging Face Hub (lazily, on first predict).

        ``revision`` is pinned explicitly (commit sha or tag). The default model is a public,
        ungated Hub repository, so no authentication is required.
        """
        return cls(
            device=device,
            model_id=model_id,
            revision=revision,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )

    @classmethod
    def from_local(cls, path: str, *, device: str = "auto") -> RadGraphIT:
        """Development / offline path: load a manifest-verified bundle from a local directory."""
        instance = cls(device=device)
        instance._resolve = functools.partial(_resolve_local, path)
        return instance

    @classmethod
    def from_service(cls, service: InferenceService, *, device: str = "auto") -> RadGraphIT:
        """Advanced/testing: build a predictor around a pre-constructed service (e.g. a custom or
        fake backend). Skips all artifact resolution and loading."""
        instance = cls(device=device)
        instance._service = service
        return instance

    def load(self) -> RadGraphIT:
        """Eagerly resolve and load the model. Returns ``self`` for chaining. Optional — the model
        is otherwise loaded on first :meth:`predict`."""
        self._ensure_loaded()
        return self

    def _ensure_loaded(self) -> InferenceService:
        if self._service is None:
            resolved = self._resolve()
            backend = _build_backend(resolved, self._device)
            self._service = InferenceService(backend)
            logger.debug("RadGraphIT model loaded (backend=%s)", resolved.manifest.backend)
        return self._service

    def predict(self, reports: str | Sequence[str]) -> dict[str, object]:
        """Annotate one report (``str``) or several (``Sequence[str]``).

        Returns the canonical RadGraph-XL dictionary, keyed by input order (``"0"``, ``"1"``, …).
        See the README for the exact output shape.
        """
        return self._ensure_loaded().predict(reports)

    def predict_graphs(self, reports: str | Sequence[str]) -> list[ReportPrediction]:
        """Advanced: return the typed internal graphs instead of the RadGraph-XL dict."""
        return self._ensure_loaded().predict_graphs(reports)

    def __call__(self, reports: str | Sequence[str]) -> dict[str, object]:
        """Alias for :meth:`predict`, for compatibility with the upstream callable interface."""
        return self.predict(reports)
