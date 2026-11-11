from __future__ import annotations

from pathlib import Path

import chromadb

from regrag.types import Chunk


class ChromaStore:
    def __init__(self, path: str | Path, name: str = "regrag") -> None:
        self._col = chromadb.PersistentClient(path=str(path)).get_or_create_collection(
            name, metadata={"hnsw:space": "cosine"}
        )

    def count(self) -> int:
        return self._col.count()

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        self._col.upsert(
            ids=[c.id for c in chunks],
            embeddings=embeddings,
            documents=[c.text for c in chunks],
            metadatas=[{"doc_id": c.doc_id, "page": c.page or 0, "section": c.section or ""} for c in chunks],
        )

    def ids(self) -> set[str]:
        return set(self._col.get(include=[])["ids"])

    def delete_doc(self, doc_id: str) -> None:
        self._col.delete(where={"doc_id": doc_id})

    def doc_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for m in self._col.get(include=["metadatas"])["metadatas"]:
            counts[m["doc_id"]] = counts.get(m["doc_id"], 0) + 1
        return counts

    def search(self, embedding: list[float], k: int = 20, doc_ids: list[str] | None = None) -> list[tuple[Chunk, float]]:
        r = self._col.query(query_embeddings=[embedding], n_results=k,
                            where={"doc_id": {"$in": doc_ids}} if doc_ids else None)
        out = []
        for id_, doc, meta, dist in zip(r["ids"][0], r["documents"][0], r["metadatas"][0], r["distances"][0]):
            chunk = Chunk(id=id_, doc_id=meta["doc_id"], text=doc, page=meta["page"] or None,
                          section=meta["section"] or None)
            out.append((chunk, 1.0 - dist))  # cosine similarity
        return out
