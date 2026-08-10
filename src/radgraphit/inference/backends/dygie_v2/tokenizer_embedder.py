"""Word-level embeddings from a subword encoder — faithful port of
``training_v2/src/dygie/tokenizer_embedder.py``.

Tokenize each word into subwords, run the transformer, mean-pool subwords back to one vector per
word. Long documents are split into non-overlapping, word-aligned chunks of at most ``max_length``
transformer positions ("context folding"), so a word's subwords are always encoded together.

The ``encoder`` submodule is built from the pinned encoder configuration without downloading its
pretrained weights: every encoder parameter is subsequently populated by the verified RadGraphIT
checkpoint through ``load_state_dict(strict=True)``.
"""

from __future__ import annotations

import logging

import torch
from torch import nn
from transformers import AutoConfig, AutoModel, AutoTokenizer

logger = logging.getLogger("radgraphit")


class MismatchedEmbedder(nn.Module):
    def __init__(
        self,
        model_name: str,
        max_length: int,
        train_parameters: bool = True,
        *,
        revision: str | None = None,
    ):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
        encoder_config = AutoConfig.from_pretrained(model_name, revision=revision)
        self.encoder = AutoModel.from_config(encoder_config)  # type: ignore[no-untyped-call]
        self.max_length = max_length
        self.output_dim = self.encoder.config.hidden_size
        if not train_parameters:
            for p in self.encoder.parameters():
                p.requires_grad = False

        num_specials = self.tokenizer.num_special_tokens_to_add(pair=False)
        budget = max_length - num_specials
        if budget <= 0:
            raise ValueError(
                f"max_length={max_length} too small for {model_name}'s "
                f"{num_specials} special tokens"
            )
        self._budget = budget

    def get_output_dim(self) -> int:
        return self.output_dim

    def _word_subword_counts(self, words: list[str]) -> list[int]:
        enc = self.tokenizer(words, is_split_into_words=True, add_special_tokens=False)
        counts = [0] * len(words)
        for w in enc.word_ids():
            if w is not None:
                counts[w] += 1
        return counts

    def _pack_chunks(self, counts: list[int]) -> list[list[int]]:
        """Greedily group word indices into chunks whose total subword count fits the budget."""
        chunks: list[list[int]] = []
        current: list[int] = []
        current_len = 0
        for w, n in enumerate(counts):
            n = max(n, 1)
            if n > self._budget:
                logger.warning(
                    "word %d has %d subwords > budget %d; it will be truncated by the tokenizer",
                    w,
                    n,
                    self._budget,
                )
            if current and current_len + n > self._budget:
                chunks.append(current)
                current, current_len = [], 0
            current.append(w)
            current_len += n
        if current:
            chunks.append(current)
        return chunks

    def forward(self, words: list[str]) -> torch.Tensor:
        """Returns (num_words, hidden_size)."""
        device = next(self.encoder.parameters()).device
        counts = self._word_subword_counts(words)
        chunks = self._pack_chunks(counts)

        word_embeddings: list[torch.Tensor | None] = [None] * len(words)
        hidden: torch.Tensor | None = None
        for chunk_word_ixs in chunks:
            chunk_words = [words[i] for i in chunk_word_ixs]
            enc = self.tokenizer(
                chunk_words,
                is_split_into_words=True,
                add_special_tokens=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)
            hidden = self.encoder(
                input_ids=input_ids, attention_mask=attention_mask
            ).last_hidden_state[0]

            word_ids = enc.word_ids(batch_index=0)
            positions_by_word: dict[int, list[int]] = {}
            for pos, w in enumerate(word_ids):
                if w is not None:
                    positions_by_word.setdefault(w, []).append(pos)
            for local_w, positions in positions_by_word.items():
                word_embeddings[chunk_word_ixs[local_w]] = hidden[positions].mean(dim=0)

        for i, emb in enumerate(word_embeddings):
            if emb is None:  # word tokenized to zero subwords (rare; e.g. an unknown char)
                assert hidden is not None  # at least one chunk always runs for a non-empty report
                word_embeddings[i] = hidden.new_zeros(self.output_dim)

        return torch.stack(word_embeddings, dim=0)  # type: ignore[arg-type]
