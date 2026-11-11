from __future__ import annotations

from typing import Protocol

from regrag.types import Chunk


class VectorStore(Protocol):
    """What the retrievers need from a vector store. Embeddings are L2-normalised, so score = cosine similarity."""

    def count(self) -> int: ...

    def ids(self) -> set[str]: ...

    def delete_doc(self, doc_id: str) -> None: ...

    def doc_counts(self) -> dict[str, int]: ...

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None: ...

    def search(self, embedding: list[float], k: int = 20, doc_ids: list[str] | None = None) -> list[tuple[Chunk, float]]: ...
