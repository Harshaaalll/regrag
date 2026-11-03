"""Chunk every PDF in data/raw and embed it into the Chroma index at data/index.

Resumable: chunks already in the index are skipped.
Run: python scripts/build_index.py [--skip 151MD ...]   (skipped docs are also removed from the index)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

from regrag.chunking.heading_aware import chunk_pages
from regrag.config import ROOT, load_config
from regrag.embeddings.bge import Embedder
from regrag.ingestion.loader import load_pdf
from regrag.vectordb.chroma_store import ChromaStore

BATCH = 32  # chunks embedded and written per step


def main() -> None:
    skip = set(sys.argv[sys.argv.index("--skip") + 1 :]) if "--skip" in sys.argv else set()
    cfg = load_config()
    ch = cfg["chunking"]
    chunks = []
    for pdf in sorted((ROOT / cfg["paths"]["raw"]).glob("*.pdf")):
        if pdf.stem in skip:
            print(f"{pdf.name}: skipped", flush=True)
            continue
        pieces = chunk_pages(load_pdf(pdf), ch["max_tokens"], ch["overlap_tokens"])
        print(f"{pdf.name}: {len(pieces)} chunks", flush=True)
        chunks.extend(pieces)

    store = ChromaStore(ROOT / cfg["paths"]["index"])
    for doc in skip:  # keep the index consistent with the corpus
        store._col.delete(where={"doc_id": doc})
    done = set(store._col.get(include=[])["ids"])
    todo = [c for c in chunks if c.id not in done]
    print(f"{len(chunks)} chunks total, {len(todo)} to embed", flush=True)
    if not todo:
        return

    emb = Embedder(cfg["embeddings"]["model"], cfg["embeddings"]["device"])
    t0 = time.time()
    for i in range(0, len(todo), BATCH):
        batch = todo[i : i + BATCH]
        store.add(batch, emb.encode([c.text for c in batch]))
        n = i + len(batch)
        print(f"embedded {n}/{len(todo)}  {time.time() - t0:.0f}s", flush=True)
    print("index size:", store.count())


if __name__ == "__main__":
    sys.exit(main())
