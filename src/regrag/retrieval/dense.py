from __future__ import annotations

from regrag.embeddings.bge import Embedder
from regrag.types import Chunk
from regrag.vectordb.chroma_store import ChromaStore


class DenseRetriever:
    def __init__(self, embedder: Embedder, store: ChromaStore, doc_ids: list[str] | None = None) -> None:
        self.embedder, self.store, self.doc_ids = embedder, store, doc_ids

    def search(self, query: str, k: int = 20) -> list[tuple[Chunk, float]]:
        return self.store.search(self.embedder.encode([query])[0], k, self.doc_ids)
