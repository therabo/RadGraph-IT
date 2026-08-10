"""Candidate span enumeration — the inference-relevant half of ``training_v2/src/dygie/dataset.py``.

One document is a single "sentence" spanning the whole report (v2 invariant), so enumeration is the
only piece of the dataset reader inference needs: all inclusive ``(start, end)`` spans of width
1..``max_span_width``, ordered by ``(start, end)``.
"""

from __future__ import annotations


def enumerate_spans(
    words: tuple[str, ...] | list[str], max_span_width: int
) -> list[tuple[int, int]]:
    """All inclusive (start, end) spans of width 1..max_span_width, ordered by (start, end)."""
    n = len(words)
    spans: list[tuple[int, int]] = []
    for start in range(n):
        for end in range(start, min(start + max_span_width, n)):
            spans.append((start, end))
    return spans
