from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

from regrag.types import Chunk


def _tok(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


class BM25Retriever:
    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        self._bm25 = BM25Okapi([_tok(c.text) for c in chunks])

    def search(self, query: str, k: int = 20) -> list[tuple[Chunk, float]]:
        scores = self._bm25.get_scores(_tok(query))
        top = sorted(range(len(scores)), key=scores.__getitem__, reverse=True)[:k]
        return [(self.chunks[i], float(scores[i])) for i in top]
