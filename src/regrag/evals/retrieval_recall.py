from __future__ import annotations

from pathlib import Path

import yaml

from regrag.types import Chunk


def load_items(path: Path) -> list[dict]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))["items"]


def page_spans(chunks: list[Chunk]) -> dict[str, tuple[int, int]]:
    """A chunk only stores its start page; assume it runs up to the next chunk's start page."""
    spans = {}
    for i, c in enumerate(chunks):
        same_doc = i + 1 < len(chunks) and chunks[i + 1].doc_id == c.doc_id
        nxt = chunks[i + 1].page if same_doc else c.page
        spans[c.id] = (c.page, max(c.page, nxt))
    return spans


def recall_at_k(items: list[dict], chunks: list[Chunk], search, ks=(1, 3, 5, 10, 20), doc_id: str | None = None) -> dict:
    """Share of answerable questions where a top-k chunk covers at least one gold page.

    `strict` counts only the chunk's start page; `loose` counts its whole assumed span.
    """
    spans = page_spans(chunks)
    rows = [it for it in items if it["tag"] != "unanswerable" and it["pages"]]
    out = {k: {"strict": 0, "loose": 0} for k in ks}
    per_item = {}
    for it in rows:
        gold = set(it["pages"])
        hits = search(it["q"], max(ks))
        first = {"strict": None, "loose": None}
        for rank, (c, _) in enumerate(hits, start=1):
            if doc_id and c.doc_id != doc_id:
                continue  # a hit in the wrong document never counts
            lo, hi = spans[c.id]
            if first["strict"] is None and c.page in gold:
                first["strict"] = rank
            if first["loose"] is None and gold & set(range(lo, hi + 1)):
                first["loose"] = rank
        per_item[it["id"]] = (it["tag"], first["strict"], first["loose"])
        for k in ks:
            for m in ("strict", "loose"):
                if first[m] is not None and first[m] <= k:
                    out[k][m] += 1
    n = len(rows)
    return {"n": n, "recall": {k: {m: v / n for m, v in d.items()} for k, d in out.items()}, "per_item": per_item}
