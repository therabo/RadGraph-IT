"""The single owner of the external RadGraph-XL schema.

Everything about how a predicted graph becomes the canonical RadGraph-XL dictionary lives here and
nowhere else — no serialization logic is scattered through the neural network or the backend. The
schema reproduced is the one emitted by the upstream RadGraph package
(``radgraph/utils.py::postprocess_reports`` / ``get_entity``), with three deliberate, documented
guarantees added on top so the output is reproducible and never self-inconsistent:

1. **Stable entity ordering.** Entities are sorted by ``(start_ix, end_ix, label)`` *before* ids
   are assigned, so the same graph always serializes to the same ``"1"``, ``"2"``, … numbering
   (upstream numbered them in raw model-output order, which is not reproducible run to run).
2. **No dangling edges.** A predicted relation is emitted only if *both* endpoints correspond to
   an emitted NER node. Relations that don't are dropped from the contract and counted in the
   internal :class:`SerializationDiagnostics` instead — the XL contract itself is never altered.
3. **Deterministic relation ordering.** Each entity's ``relations`` list is sorted by
   ``(label, target_entity_id)``.

The per-report dictionary keeps exactly the upstream keys — ``text``, ``entities``,
``data_source`` (``None``), ``data_split`` (``"inference"``) — and adds no others. Relation entries
are ``[label, target_entity_id]`` with no score (the default contract carries no scores).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from ..core.models import ReportPrediction

DATA_SPLIT = "inference"

# The four keys of a per-report RadGraph-XL object, in upstream order. Kept as a constant so tests
# can assert the contract has neither gained nor lost a key.
REPORT_KEYS = ("text", "entities", "data_source", "data_split")


@dataclass(frozen=True, slots=True)
class SerializationDiagnostics:
    """Non-contract, internal-only counters gathered while serializing.

    ``dropped_dangling_relations`` counts predicted relations discarded because at least one
    endpoint had no emitted NER node. ``per_report`` maps the top-level report key to that report's
    own dropped count.
    """

    dropped_dangling_relations: int = 0
    per_report: dict[str, int] = field(default_factory=dict)


def _entities_to_dict(pred: ReportPrediction) -> tuple[dict[str, object], int]:
    """Serialize one report's entities (with their relations) and return ``(entities, dropped)``.

    ``dropped`` is the number of predicted relations not emitted because an endpoint span was not
    an emitted NER node.
    """
    # Stable, reproducible numbering: sort by (start, end, label) then assign 1-based string ids.
    ordered = sorted(pred.entities, key=lambda e: (e.start_ix, e.end_ix, e.label))

    span_to_id: dict[tuple[int, int], str] = {}
    for position, entity in enumerate(ordered, start=1):
        # First entity at a given span wins (spans are unique in NER output; this is defensive and
        # matches upstream's first-match resolution for duplicate spans).
        span_to_id.setdefault(entity.span, str(position))

    entities: dict[str, object] = {}
    emitted_relations = 0
    for position, entity in enumerate(ordered, start=1):
        relations: list[list[str]] = []
        for relation in pred.relations:
            if relation.source_span != entity.span:
                continue
            target_id = span_to_id.get(relation.target_span)
            if target_id is None:
                continue  # dangling: target span has no emitted node — dropped (counted below)
            relations.append([relation.label, target_id])
            emitted_relations += 1

        relations.sort(key=lambda label_target: (label_target[0], int(label_target[1])))
        entities[str(position)] = {
            "tokens": " ".join(pred.tokens[entity.start_ix : entity.end_ix + 1]),
            "label": entity.label,
            "start_ix": entity.start_ix,
            "end_ix": entity.end_ix,
            "relations": relations,
        }

    dropped = len(pred.relations) - emitted_relations
    return entities, dropped


def report_to_radgraph_xl(pred: ReportPrediction) -> tuple[dict[str, object], int]:
    """Serialize a single report to its RadGraph-XL object; return ``(object, dropped_count)``."""
    entities, dropped = _entities_to_dict(pred)
    report: dict[str, object] = {
        "text": pred.text,
        "entities": entities,
        "data_source": None,
        "data_split": DATA_SPLIT,
    }
    return report, dropped


def to_radgraph_xl(predictions: Sequence[ReportPrediction]) -> dict[str, object]:
    """Serialize a batch to the canonical RadGraph-XL dict, keyed by input order (``"0"``, ``"1"``).

    This is the public contract returned by ``RadGraphIT.predict`` / ``__call__``. Report order is
    preserved: the key is the positional index in ``predictions``, regardless of each prediction's
    own ``doc_key``.
    """
    result, _ = to_radgraph_xl_with_diagnostics(predictions)
    return result


def to_radgraph_xl_with_diagnostics(
    predictions: Sequence[ReportPrediction],
) -> tuple[dict[str, object], SerializationDiagnostics]:
    """Like :func:`to_radgraph_xl` but also return :class:`SerializationDiagnostics`.

    Used internally / by advanced callers that want the dropped-dangling-relation count without
    changing the contract shape.
    """
    result: dict[str, object] = {}
    per_report: dict[str, int] = {}
    total_dropped = 0
    for index, pred in enumerate(predictions):
        key = str(index)
        report, dropped = report_to_radgraph_xl(pred)
        result[key] = report
        per_report[key] = dropped
        total_dropped += dropped
    diagnostics = SerializationDiagnostics(
        dropped_dangling_relations=total_dropped,
        per_report=per_report,
    )
    return result, diagnostics
