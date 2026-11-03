from __future__ import annotations

from pathlib import Path

import fitz  # pymupdf

from regrag.types import Chunk


def load_pdf(path: Path) -> list[Chunk]:
    """Page-level text blocks. Tables/figures are handled by the Docling path in a later step."""
    doc_id = path.stem
    chunks: list[Chunk] = []
    with fitz.open(path) as pdf:
        for i, page in enumerate(pdf, start=1):
            text = page.get_text("text").strip()
            if text:
                chunks.append(Chunk(id=f"{doc_id}:p{i}", doc_id=doc_id, text=text, page=i))
    return chunks


def load_folder(folder: Path) -> list[Chunk]:
    out: list[Chunk] = []
    for p in sorted(folder.glob("*.pdf")):
        out.extend(load_pdf(p))
    return out
