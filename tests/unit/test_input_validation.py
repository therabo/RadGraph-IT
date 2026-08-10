"""Acceptance test #3a — deterministic input validation.

A non-empty ``str`` or non-empty ``Sequence[str]`` is accepted and order-preserving; everything else
is rejected with :class:`InvalidReportError`. An empty report is an error, never silently coerced.
"""

from __future__ import annotations

import pytest

from radgraphit.core.errors import InvalidReportError
from radgraphit.inference.service import InferenceService, normalize_reports


def test_single_string_becomes_one_item_list() -> None:
    assert normalize_reports("referto") == ["referto"]


def test_sequence_preserves_order() -> None:
    assert normalize_reports(["a", "b", "c"]) == ["a", "b", "c"]
    assert normalize_reports(("x", "y")) == ["x", "y"]


@pytest.mark.parametrize("bad", ["", "   ", "\n\t "])
def test_empty_or_whitespace_string_is_rejected(bad: str) -> None:
    with pytest.raises(InvalidReportError):
        normalize_reports(bad)


def test_empty_sequence_is_rejected() -> None:
    with pytest.raises(InvalidReportError):
        normalize_reports([])


@pytest.mark.parametrize("bad", [None, 123, 4.5, {"a": 1}, b"bytes"])
def test_non_string_non_sequence_is_rejected(bad: object) -> None:
    with pytest.raises(InvalidReportError):
        normalize_reports(bad)  # type: ignore[arg-type]


def test_sequence_with_non_string_element_is_rejected() -> None:
    with pytest.raises(InvalidReportError):
        normalize_reports(["ok", 42])  # type: ignore[list-item]


def test_sequence_with_whitespace_element_is_rejected() -> None:
    with pytest.raises(InvalidReportError):
        normalize_reports(["ok", "   "])


def test_service_predict_rejects_bad_input_before_touching_backend(empty_backend) -> None:
    service = InferenceService(empty_backend)
    with pytest.raises(InvalidReportError):
        service.predict("")
    assert empty_backend.calls == []  # validation happens before any backend call
