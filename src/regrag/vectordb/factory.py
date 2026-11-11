from __future__ import annotations

import os
from pathlib import Path

from regrag.vectordb.base import VectorStore


def make_store(index_dir: Path, kind: str | None = None) -> VectorStore:
    """kind: chroma (default) | faiss-flat | faiss-hnsw. Override with env REGRAG_VECTORDB.

    Faiss indexes live next to the Chroma one in <index_dir>-faiss, so the two never mix.
    """
    kind = kind or os.environ.get("REGRAG_VECTORDB", "chroma")
    if kind == "chroma":
        from regrag.vectordb.chroma_store import ChromaStore

        return ChromaStore(index_dir)
    if kind in ("faiss-flat", "faiss-hnsw"):
        from regrag.vectordb.faiss_store import FaissStore

        return FaissStore(Path(f"{index_dir}-faiss"), kind.split("-")[1])
    raise ValueError(f"unknown vector store {kind!r}")
