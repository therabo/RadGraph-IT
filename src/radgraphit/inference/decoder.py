"""Decode raw DyGIE backend output into immutable domain value objects.

The backend speaks in plain tuples (the shape upstream DyGIE ``decode`` produced); this module is
the single place that lifts those into typed :class:`ScoredEntity` / :class:`ScoredRelation`
objects, producing one :class:`ReportPrediction` per report. Keeping this separate from both the
torch backend and the serializer means the whole "raw predictions -> domain graph" step is pure
and trivially testable with synthetic tuples.
"""

from __future__ import annotations

from ..core.models import (
    RawBackendOutput,
    ReportPrediction,
    ScoredEntity,
    ScoredRelation,
)


def decode_report(
    doc_key: str,
    tokens: tuple[str, ...],
    raw: RawBackendOutput,
) -> ReportPrediction:
    """Lift one report's raw backend tuples into a typed :class:`ReportPrediction`."""
    entities = tuple(
        ScoredEntity(
            start_ix=int(start),
            end_ix=int(end),
            label=str(label),
            raw_score=float(raw_score),
            softmax_score=float(soft_score),
        )
        for (start, end, label, raw_score, soft_score) in raw.ner
    )
    relations = tuple(
        ScoredRelation(
            label=str(label),
            source_start=int(s1),
            source_end=int(e1),
            target_start=int(s2),
            target_end=int(e2),
            raw_score=float(raw_score),
            softmax_score=float(soft_score),
        )
        for (s1, e1, s2, e2, label, raw_score, soft_score) in raw.relations
    )
    return ReportPrediction(
        doc_key=doc_key,
        tokens=tokens,
        entities=entities,
        relations=relations,
    )
