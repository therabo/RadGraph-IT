"""RadGraph-XL-compatible report tokenizer for Italian reports.

This is the exact equivalent of ``radgraph_xl_preprocess_report`` from the upstream RadGraph
package (``radgraph/utils.py``) — the same normalization substitutions followed by
``nltk.tokenize.wordpunct_tokenize`` and the same handful of post-tokenization fixups. The v2
training pipeline uses byte-for-byte this function for its own tokenization
(``training_v2/src/check_tokenization_raw_it.py`` documents it as an "EXACT COPY … (inference
code)"), so reproducing it here is what keeps train-time and inference-time token indices aligned.

``nltk.tokenize.wordpunct_tokenize`` is backed by a pure-regex ``WordPunctTokenizer`` (pattern
``\\w+|[^\\w\\s]+``); it needs no downloaded NLTK data and performs no I/O, so importing and calling
it never touches the network.

Indices produced downstream from these tokens are **token-level, zero-based, and inclusive**.
This module never emits character offsets.
"""

from __future__ import annotations

import re

from nltk.tokenize import wordpunct_tokenize

from ..core.models import TokenizedReport

# Collapse any run of whitespace to a single space (applied after the literal substitutions).
_WHITESPACE = re.compile(r"\s+")


def normalize_report(text: str) -> str:
    """Apply the RadGraph-XL normalization + tokenization, returning a space-joined token string.

    Kept as a separate function (mirroring upstream) so the exact normalized string can be
    inspected or tested independently of the ``.split()`` that yields the token list.
    """
    text = text.replace("\\n", "  ")
    text = text.replace("\\f", "  ")
    text = text.replace("\\u2122", "      ")
    text = text.replace("\n", " ")
    text = text.replace('\\"', "``")

    text_sub = _WHITESPACE.sub(" ", text)
    tokenized_text = " ".join(wordpunct_tokenize(text_sub))
    tokenized_text = tokenized_text.replace(").", ") .")
    tokenized_text = tokenized_text.replace("%.", "% .")
    tokenized_text = tokenized_text.replace(".'", ". '")
    tokenized_text = tokenized_text.replace("%,", "% ,")
    tokenized_text = tokenized_text.replace("%)", "% )")
    return tokenized_text


def tokenize_report(text: str) -> TokenizedReport:
    """Normalize and tokenize one report into a :class:`TokenizedReport`.

    The caller is responsible for having validated that ``text`` is a non-empty, non-whitespace
    string (see :func:`radgraphit.inference.service.normalize_reports`); any such string yields at
    least one token, because ``wordpunct_tokenize`` emits a token for every non-whitespace run.
    """
    tokens = normalize_report(text).split()
    return TokenizedReport(tokens=tuple(tokens))
