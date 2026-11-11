"""Compare Chroma, Faiss-flat and Faiss-HNSW on the real index, without loading any embedding model.

Reads the vectors already stored in Chroma, builds Faiss copies in a temp folder, then times search and checks how often
the top-5 matches Chroma's. Queries are the stored chunk vectors themselves (easy queries, so agreement will look high).
--scale N adds a SYNTHETIC timing test (jittered copies of the real vectors) for flat vs HNSW only: timing, not accuracy.
  python scripts/compare_vectorstores.py [--queries 100] [--scale 100000]
"""
from __future__ import annotations

import argparse
import json
import tempfile
import time
from statistics import median

import numpy as np

from regrag.config import ROOT, load_config
from regrag.types import Chunk
from regrag.vectordb.chroma_store import ChromaStore
from regrag.vectordb.faiss_store import FaissStore


def timed(fn, n: int) -> float:
    t = []
    for i in range(n):
        s = time.perf_counter()
        fn(i)
        t.append((time.perf_counter() - s) * 1000)
    return median(t)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", type=int, default=100)
    ap.add_argument("--scale", type=int, default=0)
    a = ap.parse_args()

    chroma = ChromaStore(ROOT / load_config()["paths"]["index"])
    got = chroma._col.get(include=["embeddings", "documents", "metadatas"])
    vecs = np.asarray(got["embeddings"], dtype="float32")
    chunks = [Chunk(id=i, doc_id=m["doc_id"], text=d, page=m["page"] or None) for i, d, m in zip(got["ids"], got["documents"], got["metadatas"])]
    n, q = len(chunks), min(a.queries, len(chunks))
    rng = np.random.default_rng(0)
    qi = rng.choice(n, q, replace=False)
    result = {"chunks": n, "queries": q, "dim": int(vecs.shape[1]), "stores": {}}
    ref = [[c.id for c, _ in chroma.search(vecs[i].tolist(), 5)] for i in qi]
    result["stores"]["chroma"] = {"search_ms_median": round(timed(lambda j: chroma.search(vecs[qi[j]].tolist(), 5), q), 2)}
    with tempfile.TemporaryDirectory() as tmp:
        for kind in ("flat", "hnsw"):
            t0 = time.perf_counter()
            store = FaissStore(f"{tmp}/{kind}", kind)
            store.add(chunks, vecs.tolist())
            build = time.perf_counter() - t0
            hits = [[c.id for c, _ in store.search(vecs[i].tolist(), 5)] for i in qi]
            overlap = float(np.mean([len(set(h) & set(r)) / 5 for h, r in zip(hits, ref)]))
            result["stores"][f"faiss-{kind}"] = {
                "build_s": round(build, 2), "search_ms_median": round(timed(lambda j: store.search(vecs[qi[j]].tolist(), 5), q), 2),
                "top5_overlap_with_chroma": round(overlap, 3)}
    if a.scale:
        import faiss

        big = vecs[rng.integers(0, n, a.scale)] + rng.normal(0, 0.02, (a.scale, vecs.shape[1])).astype("float32")
        big /= np.linalg.norm(big, axis=1, keepdims=True)
        qs = big[rng.choice(a.scale, min(q, 100), replace=False)]
        syn = {}
        for kind, idx in (("flat", faiss.IndexFlatIP(big.shape[1])), ("hnsw", faiss.IndexHNSWFlat(big.shape[1], 32, faiss.METRIC_INNER_PRODUCT))):
            t0 = time.perf_counter()
            idx.add(big)
            syn[kind] = {"build_s": round(time.perf_counter() - t0, 2),
                         "search_ms_median": round(timed(lambda j: idx.search(qs[j:j + 1], 5), len(qs)), 3)}
        result["synthetic_scale"] = {"vectors": a.scale, "note": "jittered copies of real vectors; timing only, not accuracy", **syn}
    print(json.dumps(result, indent=1))
    out = ROOT / "evals" / "results"
    out.mkdir(parents=True, exist_ok=True)
    (out / "vectorstore_compare.json").write_text(json.dumps(result, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
