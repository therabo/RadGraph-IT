"""Acceptance test #2 — pure decoder + serializer with synthetic raw predictions.

Exercises, on a hand-built raw prediction (no model): multiple entities, relation direction, a
dropped dangling relation, stable entity ordering, deterministic relation ordering, multi-token
span text, batch keying, and the exact RadGraph-XL dictionary.
"""

from __future__ import annotations

from radgraphit.core.models import RawBackendOutput, ScoredEntity, ScoredRelation
from radgraphit.inference.decoder import decode_report
from radgraphit.serialization.radgraph_xl import (
    REPORT_KEYS,
    to_radgraph_xl,
    to_radgraph_xl_with_diagnostics,
)

# "Nessuna evidenza di pneumotorace nel torace ." — token indices 0..6.
TOKENS = ("Nessuna", "evidenza", "di", "pneumotorace", "nel", "torace", ".")

# Entities are supplied out of (start, end, label) order on purpose, to prove the serializer's
# stable re-ordering. torace (5,5) is given before pneumotorace (3,3).
NER = (
    (5, 5, "Anatomy::definitely present", 3.1, 0.98),
    (3, 3, "Observation::definitely absent", 2.4, 0.91),
)
# Relations, also out of order, including one dangling edge (target span (10,10) has no node).
RELATIONS = (
    (3, 3, 5, 5, "modify", 1.0, 0.7),  # pneumotorace -> torace
    (3, 3, 5, 5, "located_at", 1.5, 0.8),  # pneumotorace -> torace (same source, sorts first)
    (3, 3, 10, 10, "modify", 0.9, 0.6),  # dangling: no entity at (10,10) -> dropped
    (5, 5, 3, 3, "suggestive_of", 1.2, 0.75),  # torace -> pneumotorace (reverse direction)
)


def _raw() -> RawBackendOutput:
    return RawBackendOutput(ner=NER, relations=RELATIONS)


def test_decoder_lifts_raw_tuples_into_typed_objects() -> None:
    pred = decode_report("0", TOKENS, _raw())
    assert pred.doc_key == "0"
    assert pred.tokens == TOKENS
    assert pred.entities == (
        ScoredEntity(5, 5, "Anatomy::definitely present", 3.1, 0.98),
        ScoredEntity(3, 3, "Observation::definitely absent", 2.4, 0.91),
    )
    assert pred.relations[0] == ScoredRelation("modify", 3, 3, 5, 5, 1.0, 0.7)
    assert pred.relations[0].source_span == (3, 3)
    assert pred.relations[0].target_span == (5, 5)


def test_serializer_exact_radgraph_xl_dictionary() -> None:
    pred = decode_report("0", TOKENS, _raw())
    result, diagnostics = to_radgraph_xl_with_diagnostics([pred])

    assert result == {
        "0": {
            "text": "Nessuna evidenza di pneumotorace nel torace .",
            "entities": {
                # Stable id assignment: (3,3) sorts before (5,5) -> pneumotorace is entity "1".
                "1": {
                    "tokens": "pneumotorace",
                    "label": "Observation::definitely absent",
                    "start_ix": 3,
                    "end_ix": 3,
                    # Sorted by (label, target_id); the dangling modify->(10,10) is gone.
                    "relations": [["located_at", "2"], ["modify", "2"]],
                },
                "2": {
                    "tokens": "torace",
                    "label": "Anatomy::definitely present",
                    "start_ix": 5,
                    "end_ix": 5,
                    "relations": [["suggestive_of", "1"]],
                },
            },
            "data_source": None,
            "data_split": "inference",
        }
    }
    # The dangling edge is counted only in the internal diagnostics, never in the contract.
    assert diagnostics.dropped_dangling_relations == 1
    assert diagnostics.per_report == {"0": 1}


def test_report_object_has_exactly_the_contract_keys() -> None:
    result = to_radgraph_xl([decode_report("0", TOKENS, _raw())])
    assert tuple(result["0"].keys()) == REPORT_KEYS
    assert result["0"]["data_source"] is None
    assert result["0"]["data_split"] == "inference"


def test_multi_token_span_tokens_are_the_inclusive_slice() -> None:
    raw = RawBackendOutput(ner=((0, 1, "Observation::uncertain", 1.0, 0.5),), relations=())
    result = to_radgraph_xl([decode_report("0", TOKENS, raw)])
    assert result["0"]["entities"]["1"]["tokens"] == "Nessuna evidenza"
    assert result["0"]["entities"]["1"]["start_ix"] == 0
    assert result["0"]["entities"]["1"]["end_ix"] == 1


def test_batch_is_keyed_by_input_order() -> None:
    report0 = decode_report("0", TOKENS, _raw())
    report1 = decode_report(
        "1",
        ("Referto", "normale", "."),
        RawBackendOutput(ner=((0, 0, "Observation::definitely present", 1.0, 0.9),), relations=()),
    )
    result = to_radgraph_xl([report0, report1])
    assert list(result.keys()) == ["0", "1"]
    assert result["1"]["entities"]["1"]["tokens"] == "Referto"


def test_relation_to_span_without_a_source_node_is_dropped_and_counted() -> None:
    # A relation whose *source* span has no emitted entity is also dangling.
    raw = RawBackendOutput(
        ner=((5, 5, "Anatomy::definitely present", 1.0, 0.9),),
        relations=((3, 3, 5, 5, "located_at", 1.0, 0.8),),  # source (3,3) not an entity
    )
    result, diagnostics = to_radgraph_xl_with_diagnostics([decode_report("0", TOKENS, raw)])
    assert result["0"]["entities"]["1"]["relations"] == []
    assert diagnostics.dropped_dangling_relations == 1
