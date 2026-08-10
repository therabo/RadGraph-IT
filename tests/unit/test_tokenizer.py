"""Acceptance test #1 — the RadGraph-XL-compatible Italian tokenizer.

Verifies the token stream for Italian sentences with punctuation, the upstream normalization
fixups, and — crucially — that the token indices the model sees are exactly the positions in the
token list handed to the backend (no hidden re-tokenization, no character offsets).
"""

from __future__ import annotations

import pytest

from radgraphit.core.models import RawBackendOutput
from radgraphit.inference.service import InferenceService
from radgraphit.inference.tokenizer import normalize_report, tokenize_report


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "Non si evidenziano segni di pneumotorace dopo la rimozione del drenaggio toracico.",
            [
                "Non",
                "si",
                "evidenziano",
                "segni",
                "di",
                "pneumotorace",
                "dopo",
                "la",
                "rimozione",
                "del",
                "drenaggio",
                "toracico",
                ".",
            ],
        ),
        # wordpunct splits numeric dots; the ")." fixup inserts a space before the period.
        (
            "Lesione di 3.5 cm (verosimilmente benigna).",
            ["Lesione", "di", "3", ".", "5", "cm", "(", "verosimilmente", "benigna", ")", "."],
        ),
        # the "%." fixup splits the percent sign from the period.
        ("Aumento del 50%.", ["Aumento", "del", "50", "%", "."]),
        # apostrophe is its own token (wordpunct).
        (
            "Riferito dolore all'emitorace destro.",
            ["Riferito", "dolore", "all", "'", "emitorace", "destro", "."],
        ),
    ],
)
def test_tokenize_italian_with_punctuation(text: str, expected: list[str]) -> None:
    tokenized = tokenize_report(text)
    assert list(tokenized.tokens) == expected
    # text is the normalized tokens joined by a single space (the RadGraph-XL "text" field).
    assert tokenized.text == " ".join(expected)


def test_normalize_collapses_whitespace_and_newlines() -> None:
    assert normalize_report("a\n\n  b\t c") == "a b c"


def test_non_whitespace_report_yields_at_least_one_token() -> None:
    assert len(tokenize_report(".").tokens) >= 1


def test_indices_are_positions_in_the_tokens_passed_to_the_backend(backend_factory) -> None:
    """The start_ix/end_ix the model emits must index the exact tuple given to ``backend.infer``."""
    text = "Non si evidenzia pneumotorace nel torace destro."
    expected_tokens = tuple(tokenize_report(text).tokens)

    # A backend that labels the span (start=3, end=3) — i.e. the 4th token — as an entity.
    def label_index_three(tokens: tuple[str, ...]) -> RawBackendOutput:
        assert tokens == expected_tokens  # the service tokenizes and passes exactly these tokens
        return RawBackendOutput(
            ner=((3, 3, "Observation::definitely absent", 2.0, 0.95),), relations=()
        )

    backend = backend_factory(label_index_three)
    service = InferenceService(backend)

    graphs = service.predict_graphs(text)
    entity = graphs[0].entities[0]

    # The entity's inclusive token slice resolves against the same token tuple the backend received.
    assert backend.calls == [expected_tokens]
    assert expected_tokens[entity.start_ix : entity.end_ix + 1] == ("pneumotorace",)
    assert expected_tokens[entity.start_ix] == "pneumotorace"
