from __future__ import annotations

from regrag.types import Chunk


class HybridRetriever:
    """Reciprocal rank fusion of several retrievers: score = sum 1 / (rrf_k + rank)."""

    def __init__(self, retrievers: list, rrf_k: int = 60, pool: int = 50) -> None:
        self.retrievers, self.rrf_k, self.pool = retrievers, rrf_k, pool

    def search(self, query: str, k: int = 20) -> list[tuple[Chunk, float]]:
        scores: dict[str, float] = {}
        chunks: dict[str, Chunk] = {}
        for r in self.retrievers:
            for rank, (c, _) in enumerate(r.search(query, max(self.pool, k)), start=1):
                chunks[c.id] = c
                scores[c.id] = scores.get(c.id, 0.0) + 1.0 / (self.rrf_k + rank)
        top = sorted(scores, key=scores.__getitem__, reverse=True)[:k]
        return [(chunks[i], scores[i]) for i in top]
