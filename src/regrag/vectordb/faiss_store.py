from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from regrag.types import Chunk


class FaissStore:
    """Faiss index plus a JSON sidecar for chunk text and metadata (Faiss itself stores only vectors).

    kind: "flat" = exact search (IndexFlatIP), "hnsw" = approximate graph search (IndexHNSWFlat).
    Same add / search / count / doc_counts interface as ChromaStore. add() with an existing id replaces that chunk.
    """

    def __init__(self, path: str | Path, kind: str = "flat", dim: int | None = None) -> None:
        import faiss  # pip install -e ".[faiss]"

        if kind not in ("flat", "hnsw"):
            raise ValueError(f"unknown kind {kind!r}")
        self._faiss, self.kind, self.path = faiss, kind, Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self._idx_file, self._meta_file = self.path / f"{kind}.faiss", self.path / f"{kind}.json"
        self.chunks: list[Chunk] = []
        self.vectors = np.zeros((0, dim or 0), dtype="float32")
        if self._meta_file.exists():
            self.chunks = [Chunk(**c) for c in json.loads(self._meta_file.read_text(encoding="utf-8"))]
            self.vectors = np.load(self.path / f"{kind}.npy")
        self._index = None
        self._rebuild()

    def _rebuild(self) -> None:
        if not len(self.vectors):
            self._index = None
            return
        d = self.vectors.shape[1]
        self._index = self._faiss.IndexFlatIP(d) if self.kind == "flat" else self._faiss.IndexHNSWFlat(d, 32, self._faiss.METRIC_INNER_PRODUCT)
        self._index.add(self.vectors)

    def count(self) -> int:
        return len(self.chunks)

    def ids(self) -> set[str]:
        return {c.id for c in self.chunks}

    def delete_doc(self, doc_id: str) -> None:
        keep = [i for i, c in enumerate(self.chunks) if c.doc_id != doc_id]
        if len(keep) == len(self.chunks):
            return
        self.chunks, self.vectors = [self.chunks[i] for i in keep], self.vectors[keep]
        self._rebuild()
        self._meta_file.write_text(json.dumps([c.model_dump() for c in self.chunks], ensure_ascii=False), encoding="utf-8")
        np.save(self.path / f"{self.kind}.npy", self.vectors)
        if self._index is not None:
            self._faiss.write_index(self._index, str(self._idx_file))

    def doc_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for c in self.chunks:
            counts[c.doc_id] = counts.get(c.doc_id, 0) + 1
        return counts

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        new = np.asarray(embeddings, dtype="float32")
        keep = [i for i, c in enumerate(self.chunks) if c.id not in {n.id for n in chunks}]
        old_vecs = self.vectors[keep] if len(self.vectors) else np.zeros((0, new.shape[1]), dtype="float32")
        self.chunks = [self.chunks[i] for i in keep] + list(chunks)
        self.vectors = np.vstack([old_vecs, new])
        self._rebuild()
        self._meta_file.write_text(json.dumps([c.model_dump() for c in self.chunks], ensure_ascii=False), encoding="utf-8")
        np.save(self.path / f"{self.kind}.npy", self.vectors)
        self._faiss.write_index(self._index, str(self._idx_file))

    def search(self, embedding: list[float], k: int = 20, doc_ids: list[str] | None = None) -> list[tuple[Chunk, float]]:
        if self._index is None:
            return []
        allowed = set(doc_ids) if doc_ids else None
        want = len(self.chunks) if allowed else min(k, len(self.chunks))  # filter after search, so look at everything
        scores, ids = self._index.search(np.asarray([embedding], dtype="float32"), want)
        out = []
        for s, i in zip(scores[0], ids[0]):
            if i >= 0 and (allowed is None or self.chunks[i].doc_id in allowed):
                out.append((self.chunks[i], float(s)))
            if len(out) == k:
                break
        return out
