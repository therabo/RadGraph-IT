"""Service-level orchestration: the RadGraph-XL dict, the typed graphs, and the diagnostics path."""

from __future__ import annotations

from radgraphit.core.models import RawBackendOutput
from radgraphit.inference.service import InferenceService


def test_predict_with_diagnostics_reports_dropped_edges(backend_factory) -> None:
    def one_dangling(tokens: tuple[str, ...]) -> RawBackendOutput:
        return RawBackendOutput(
            ner=((0, 0, "Observation::definitely present", 1.0, 0.9),),
            relations=((0, 0, 99, 99, "located_at", 1.0, 0.8),),  # target (99,99) has no node
        )

    service = InferenceService(backend_factory(one_dangling))
    result, diagnostics = service.predict_with_diagnostics("Referto di prova.")
    assert result["0"]["entities"]["1"]["relations"] == []
    assert diagnostics.dropped_dangling_relations == 1


def test_predict_graphs_and_dict_agree_on_order(recording_backend) -> None:
    service = InferenceService(recording_backend)
    reports = ["uno", "due", "tre"]
    graphs = service.predict_graphs(reports)
    result = service.predict(reports)
    assert [g.doc_key for g in graphs] == ["0", "1", "2"]
    assert list(result.keys()) == ["0", "1", "2"]
