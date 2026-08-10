"""Inference orchestration, with no global I/O.

Ties the pure pieces together — validate input, tokenize, run the (injected) backend, decode to
domain objects, serialize to RadGraph-XL — around any :class:`InferenceBackend`. The backend is a
constructor argument, which is exactly where a fake backend is injected in tests; the service
itself never imports torch.
"""

from __future__ import annotations

from collections.abc import Sequence

from ..core.errors import InvalidReportError
from ..core.models import InferenceBackend, ReportPrediction
from ..serialization.radgraph_xl import (
    SerializationDiagnostics,
    to_radgraph_xl,
    to_radgraph_xl_with_diagnostics,
)
from .decoder import decode_report
from .tokenizer import tokenize_report


def normalize_reports(reports: str | Sequence[str]) -> list[str]:
    """Validate and normalize the public input into a non-empty list of non-empty strings.

    Accepts a single ``str`` or a ``Sequence[str]``. Rejects — deterministically, with
    :class:`InvalidReportError` — anything else: the wrong type, an empty container, a non-string
    element, or a string that is empty or whitespace-only. Order is preserved.

    Unlike the upstream package, an empty report is never silently coerced to the string
    ``"None"``; it is an error the caller must see.
    """
    if isinstance(reports, str):
        items = [reports]
    elif isinstance(reports, Sequence) and not isinstance(reports, bytes | bytearray):
        items = list(reports)
    else:
        raise InvalidReportError(
            f"reports must be a str or a sequence of str, got {type(reports).__name__}."
        )

    if not items:
        raise InvalidReportError("reports is empty: provide at least one report.")

    for index, item in enumerate(items):
        if not isinstance(item, str):
            raise InvalidReportError(
                f"report at index {index} must be a str, got {type(item).__name__}."
            )
        if not item.strip():
            raise InvalidReportError(f"report at index {index} is empty or whitespace-only.")
    return items


class InferenceService:
    """Stateless orchestration around a single backend. One report is processed at a time."""

    def __init__(self, backend: InferenceBackend):
        self._backend = backend

    def predict_graphs(self, reports: str | Sequence[str]) -> list[ReportPrediction]:
        """Validate, tokenize, infer, and decode — returning the typed domain graphs (in order)."""
        items = normalize_reports(reports)
        predictions: list[ReportPrediction] = []
        for index, text in enumerate(items):
            tokenized = tokenize_report(text)
            raw = self._backend.infer(tokenized.tokens)
            predictions.append(decode_report(str(index), tokenized.tokens, raw))
        return predictions

    def predict(self, reports: str | Sequence[str]) -> dict[str, object]:
        """Return the canonical RadGraph-XL dict (the public contract)."""
        return to_radgraph_xl(self.predict_graphs(reports))

    def predict_with_diagnostics(
        self, reports: str | Sequence[str]
    ) -> tuple[dict[str, object], SerializationDiagnostics]:
        """Return the RadGraph-XL dict plus internal serialization diagnostics."""
        return to_radgraph_xl_with_diagnostics(self.predict_graphs(reports))
