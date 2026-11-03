from __future__ import annotations

import re

from regrag.types import Chunk

# "Chapter IV - Title" or bare "Chapter IV"; rejects TOC lines (dot leaders) and body sentences.
CHAPTER = re.compile(
    r"^\s*((chapter\s+[ivxlc\d]+)|(annex(ure)?\s*[-–—]?\s*[ivxlc\d]+))\s*([-–—:]\s*[^.\n]{0,100})?$",
    re.I,
)
NOISE = re.compile(r".*\.{4,}.*|\s*\d{1,3}\s*")  # TOC dot-leader lines, bare page numbers
MIN_WORDS = 20  # don't split a clause off a chunk shorter than this
CLAUSE = re.compile(r"^\s*(\d{1,3})\.\s+\S")  # "12. Text..." starts a new clause


def _words(text: str) -> int:
    return len(text.split())


def chunk_pages(pages: list[Chunk], max_tokens: int = 400, overlap_tokens: int = 50) -> list[Chunk]:
    """Split page chunks at chapter/clause starts; cap size by words (rough token proxy).

    Each chunk keeps its starting page and the latest chapter heading as `section`.
    """
    out: list[Chunk] = []
    section: str | None = None
    buf: list[str] = []
    start_page: int | None = None
    doc_id = pages[0].doc_id if pages else ""

    def flush() -> None:
        nonlocal buf, start_page
        text = "\n".join(buf).strip()
        if text:
            out.append(Chunk(id=f"{doc_id}:c{len(out)}", doc_id=doc_id, text=text,
                             page=start_page, section=section))
        tail = text.split()[-overlap_tokens:] if overlap_tokens and text else []
        buf = [" ".join(tail)] if tail else []
        start_page = None

    for page in pages:
        doc_id = page.doc_id
        for line in page.text.splitlines():
            if NOISE.fullmatch(line):
                continue
            if CHAPTER.match(line) and "..." not in line:
                flush()
                buf = []
                section = line.strip()
            elif CLAUSE.match(line) and _words("\n".join(buf)) > max(overlap_tokens, MIN_WORDS):
                flush()
            elif _words("\n".join(buf)) + _words(line) > max_tokens:
                flush()
            if start_page is None:
                start_page = page.page
            buf.append(line)
    flush()
    return out
