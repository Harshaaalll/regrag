from __future__ import annotations

from sentence_transformers import CrossEncoder

from regrag.types import Chunk


class Reranker:
    def __init__(self, model: str = "BAAI/bge-reranker-v2-m3", device: str = "cpu", max_length: int = 512) -> None:
        self.model = CrossEncoder(model, device=device, max_length=max_length)

    def rerank(self, query: str, hits: list[tuple[Chunk, float]], k: int = 5) -> list[tuple[Chunk, float]]:
        if not hits:
            return []
        scores = self.model.predict([(query, c.text) for c, _ in hits], batch_size=4, show_progress_bar=False)
        order = sorted(range(len(hits)), key=lambda i: scores[i], reverse=True)[:k]
        return [(hits[i][0], float(scores[i])) for i in order]
