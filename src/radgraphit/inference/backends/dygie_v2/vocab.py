"""Minimal label vocabulary — inference-only port of ``training_v2/src/dygie/vocab.py``.

Per-namespace string<->int maps with the null label ``""`` pinned to index 0 (so "argmax == 0"
means "no label"). Namespaces are ``"{dataset}__ner_labels"`` / ``"{dataset}__relation_labels"``;
the dataset prefix is *derived from the saved vocab*, never hard-coded, so the label schema is
whatever the checkpoint was trained with (the current v2 artifact: 6 NER classes, 3 relations).
"""

from __future__ import annotations

import json
from pathlib import Path

from ....core.errors import CheckpointError

NULL_LABEL = ""
NER_SUFFIX = "__ner_labels"
RELATION_SUFFIX = "__relation_labels"


class Vocabulary:
    """String<->index maps per namespace, loaded from the checkpoint's ``vocab.json``."""

    def __init__(self) -> None:
        self._token_to_index: dict[str, dict[str, int]] = {}
        self._index_to_token: dict[str, dict[int, str]] = {}

    def add_namespace(self, namespace: str) -> None:
        if namespace not in self._token_to_index:
            self._token_to_index[namespace] = {NULL_LABEL: 0}
            self._index_to_token[namespace] = {0: NULL_LABEL}

    def add_token(self, token: str, namespace: str) -> int:
        self.add_namespace(namespace)
        t2i = self._token_to_index[namespace]
        if token not in t2i:
            idx = len(t2i)
            t2i[token] = idx
            self._index_to_token[namespace][idx] = token
        return t2i[token]

    def get_token_index(self, token: str, namespace: str) -> int:
        return self._token_to_index[namespace][token]

    def get_token_from_index(self, index: int, namespace: str) -> str:
        return self._index_to_token[namespace][index]

    def get_vocab_size(self, namespace: str) -> int:
        return len(self._token_to_index[namespace])

    def get_namespaces(self) -> list[str]:
        return list(self._token_to_index.keys())

    @classmethod
    def load(cls, path: str | Path) -> Vocabulary:
        """Load the exact label<->index mapping saved next to the checkpoint."""
        vocab_path = Path(path)
        try:
            data = json.loads(vocab_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise CheckpointError(
                f"Could not read a valid JSON vocabulary at {vocab_path}: {exc}"
            ) from exc
        if not isinstance(data, dict) or not data:
            raise CheckpointError(f"vocab.json at {vocab_path} must contain a non-empty object.")

        vocab = cls()
        for namespace, t2i in data.items():
            if not isinstance(namespace, str) or not namespace:
                raise CheckpointError("Every vocab namespace must be a non-empty string.")
            if not isinstance(t2i, dict) or not t2i:
                raise CheckpointError(f"Vocab namespace {namespace!r} must be a non-empty object.")
            if t2i.get(NULL_LABEL) != 0:
                raise CheckpointError(
                    f"Vocab namespace {namespace!r} must map the null label '' to index 0."
                )
            if not all(
                isinstance(token, str)
                and isinstance(index, int)
                and not isinstance(index, bool)
                and index >= 0
                for token, index in t2i.items()
            ):
                raise CheckpointError(
                    f"Vocab namespace {namespace!r} must map strings to non-negative integers."
                )
            indices = list(t2i.values())
            if len(indices) != len(set(indices)) or sorted(indices) != list(range(len(indices))):
                raise CheckpointError(
                    f"Vocab namespace {namespace!r} indices must be unique and contiguous from 0."
                )
            vocab.add_namespace(namespace)
            for token, _idx in sorted(t2i.items(), key=lambda kv: kv[1]):
                if token != NULL_LABEL:
                    vocab.add_token(token, namespace)
        return vocab


def derive_dataset(vocab: Vocabulary) -> str:
    """Derive the single ``dataset`` prefix from the vocab's namespaces.

    The model was trained with one ``dataset`` constant (``"radgraph-it"`` for the current
    artifact), which prefixes both label namespaces. We recover it from the vocab rather than
    hard-coding it, and validate that the NER and relation prefixes agree, so the head lookups
    ``f"{dataset}__ner_labels"`` / ``f"{dataset}__relation_labels"`` are guaranteed to hit.
    """
    ner_prefixes = [
        ns[: -len(NER_SUFFIX)] for ns in vocab.get_namespaces() if ns.endswith(NER_SUFFIX)
    ]
    rel_prefixes = [
        ns[: -len(RELATION_SUFFIX)] for ns in vocab.get_namespaces() if ns.endswith(RELATION_SUFFIX)
    ]
    if len(ner_prefixes) != 1 or len(rel_prefixes) != 1:
        raise CheckpointError(
            "vocab.json must contain exactly one '*__ner_labels' and one '*__relation_labels' "
            f"namespace; found NER={ner_prefixes}, relation={rel_prefixes}."
        )
    if ner_prefixes[0] != rel_prefixes[0]:
        raise CheckpointError(
            f"vocab.json NER prefix {ner_prefixes[0]!r} does not match relation prefix "
            f"{rel_prefixes[0]!r}; the bundle is inconsistent."
        )
    return ner_prefixes[0]
