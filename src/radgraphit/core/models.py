"""Immutable, typed value objects that flow through the inference pipeline.

These are plain ``@dataclass(frozen=True)`` objects (no torch, no I/O) so the pure boundary —
tokenizer -> backend -> decoder -> serializer — can be reasoned about and unit-tested without a
model. Token indices everywhere are **token-level, zero-based, and inclusive** (``end_ix`` is the
last token *in* the span), exactly like RadGraph-XL; there are no character offsets.

The typed protocols (:class:`InferenceBackend`) live here too, kept intentionally small: the only
thing the service needs from a backend is "given tokens, give me raw scored predictions".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class TokenizedReport:
    """A report after RadGraph-XL-compatible normalization + tokenization.

    ``tokens`` are the normalized word tokens handed verbatim to the backend; ``text`` is their
    space-joined form (the ``"text"`` field of the RadGraph-XL output).
    """

    tokens: tuple[str, ...]

    @property
    def text(self) -> str:
        return " ".join(self.tokens)

    def __len__(self) -> int:
        return len(self.tokens)


@dataclass(frozen=True, slots=True)
class ScoredEntity:
    """One raw NER prediction: an inclusive ``[start_ix, end_ix]`` span with a label and scores."""

    start_ix: int
    end_ix: int
    label: str
    raw_score: float
    softmax_score: float

    @property
    def span(self) -> tuple[int, int]:
        return (self.start_ix, self.end_ix)


@dataclass(frozen=True, slots=True)
class ScoredRelation:
    """One raw relation prediction, **directed** source -> target.

    Endpoints are referenced by span (not entity id); the serializer resolves them to entity ids.
    """

    label: str
    source_start: int
    source_end: int
    target_start: int
    target_end: int
    raw_score: float
    softmax_score: float

    @property
    def source_span(self) -> tuple[int, int]:
        return (self.source_start, self.source_end)

    @property
    def target_span(self) -> tuple[int, int]:
        return (self.target_start, self.target_end)


@dataclass(frozen=True, slots=True)
class ReportPrediction:
    """The full raw graph predicted for a single report, before RadGraph-XL serialization.

    This is the internal, typed result. The public API serializes it to the RadGraph-XL dict;
    advanced callers can request these objects directly (see ``RadGraphIT.predict_graphs``).
    """

    doc_key: str
    tokens: tuple[str, ...]
    entities: tuple[ScoredEntity, ...]
    relations: tuple[ScoredRelation, ...]

    @property
    def text(self) -> str:
        return " ".join(self.tokens)


@dataclass(frozen=True, slots=True)
class RawBackendOutput:
    """The minimal, torch-free contract a backend returns for one report.

    NER tuples are ``(start, end, label, raw_score, softmax_score)`` and relation tuples are
    ``(s1, e1, s2, e2, label, raw_score, softmax_score)`` — the same shape upstream DyGIE decode
    produced, kept as plain Python so nothing torch-shaped leaks past the backend boundary.
    """

    ner: tuple[tuple[int, int, str, float, float], ...]
    relations: tuple[tuple[int, int, int, int, str, float, float], ...]


@runtime_checkable
class InferenceBackend(Protocol):
    """What :class:`~radgraphit.inference.service.InferenceService` needs from any backend.

    Deliberately tiny: one report's tokens in, one :class:`RawBackendOutput` out. This is the seam
    a fake backend is injected at in tests, and the boundary that keeps torch out of the service.
    """

    def infer(self, tokens: tuple[str, ...]) -> RawBackendOutput:
        """Run inference on a single report's tokens and return raw scored predictions."""
        ...
