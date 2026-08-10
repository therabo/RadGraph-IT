"""Acceptance test #6 — opt-in integration test against a real v2 checkpoint.

Runs against ``RADGRAPHIT_TEST_MODEL_DIR`` when supplied, or against the pinned public Hub model
when ``RADGRAPHIT_RUN_HUB_INTEGRATION=1``. The latter path is used by CI and exercises selective
download, manifest verification, strict checkpoint loading, tokenization, and inference together.

What it verifies:
  * the bundle loads with ``strict=True`` (via the normal ``from_local`` path, incl. manifest
    verification);
  * inference on a **synthetic, non-clinical** report returns a well-formed RadGraph-XL dictionary
    whose entity labels/relations belong to the checkpoint's own vocab schema.

To also assert an exact golden dictionary, set ``RADGRAPHIT_TEST_GOLDEN`` to a JSON file the
maintainer captured once from a trusted run on ``SYNTHETIC_REPORT`` below.

Run it with::

    RADGRAPHIT_TEST_MODEL_DIR=/path/to/bundle pytest -m integration
    RADGRAPHIT_TEST_MODEL_DIR=/path/to/bundle RADGRAPHIT_TEST_GOLDEN=/path/to/golden.json \
        pytest -m integration
    RADGRAPHIT_RUN_HUB_INTEGRATION=1 \
        RADGRAPHIT_TEST_GOLDEN=tests/fixtures/real_checkpoint_golden.json pytest -m integration
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from radgraphit import RadGraphIT

pytestmark = pytest.mark.integration

MODEL_DIR = os.environ.get("RADGRAPHIT_TEST_MODEL_DIR")
RUN_HUB = os.environ.get("RADGRAPHIT_RUN_HUB_INTEGRATION") == "1"
GOLDEN = os.environ.get("RADGRAPHIT_TEST_GOLDEN")

# A synthetic, invented report — not derived from any patient/clinical source.
SYNTHETIC_REPORT = "Non si evidenzia pneumotorace nel torace destro."


requires_model = pytest.mark.skipif(
    not MODEL_DIR and not RUN_HUB,
    reason="set RADGRAPHIT_TEST_MODEL_DIR or RADGRAPHIT_RUN_HUB_INTEGRATION=1",
)


@pytest.fixture(scope="module")
def predictor() -> RadGraphIT:
    if MODEL_DIR:
        return RadGraphIT.from_local(MODEL_DIR, device="cpu")
    return RadGraphIT(device="cpu")


@requires_model
def test_real_checkpoint_loads_strict_and_predicts(predictor: RadGraphIT) -> None:
    result = predictor.predict(SYNTHETIC_REPORT)

    # Contract shape.
    assert list(result.keys()) == ["0"]
    report = result["0"]
    assert set(report.keys()) == {"text", "entities", "data_source", "data_split"}
    assert report["data_source"] is None
    assert report["data_split"] == "inference"

    tokens = report["text"].split()
    for entity in report["entities"].values():
        assert 0 <= entity["start_ix"] <= entity["end_ix"] < len(tokens)
        assert entity["tokens"] == " ".join(tokens[entity["start_ix"] : entity["end_ix"] + 1])
        assert "::" in entity["label"]  # Anatomy::… / Observation::…
        for relation in entity["relations"]:
            label, target_id = relation
            assert isinstance(label, str)
            assert target_id in report["entities"]  # no dangling edges in the contract


@requires_model
def test_real_checkpoint_matches_golden_if_provided(predictor: RadGraphIT) -> None:
    if not GOLDEN:
        pytest.skip("set RADGRAPHIT_TEST_GOLDEN to compare against a captured golden output")
    result = predictor.predict(SYNTHETIC_REPORT)
    expected = json.loads(Path(GOLDEN).read_text(encoding="utf-8"))
    assert result == expected
