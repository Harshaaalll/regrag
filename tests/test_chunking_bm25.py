from regrag.chunking.heading_aware import chunk_pages
from regrag.retrieval.bm25 import BM25Retriever
from regrag.types import Chunk

PAGES = [
    Chunk(id="d:p1", doc_id="d", page=1,
          text="CHAPTER I\n1. Short title and commencement of these directions.\n"
               "2. Applicability to banks."),
    Chunk(id="d:p2", doc_id="d", page=2,
          text="CHAPTER II\n3. Customer due diligence requires identity proof for every account holder."),
]


def test_chapter_sets_section_and_page():
    chunks = chunk_pages(PAGES, max_tokens=50, overlap_tokens=0)
    assert chunks[-1].section == "CHAPTER II"
    assert chunks[-1].page == 2
    assert all(c.doc_id == "d" for c in chunks)


def test_max_tokens_respected():
    long = Chunk(id="d:p1", doc_id="d", page=1, text="\n".join(f"word{i} " * 10 for i in range(20)))
    chunks = chunk_pages([long], max_tokens=60, overlap_tokens=0)
    assert len(chunks) > 1
    assert all(len(c.text.split()) <= 70 for c in chunks)


def test_bm25_finds_relevant_chunk():
    filler = [
        Chunk(id=f"d:p{i}", doc_id="d", page=i, text=f"CHAPTER {'I' * i}\n{i}. Unrelated reporting duty number {i}.")
        for i in range(3, 8)
    ]  # BM25 idf is 0 on a 2-chunk corpus, so use more chunks
    chunks = chunk_pages(PAGES + filler, max_tokens=50, overlap_tokens=0)
    top = BM25Retriever(chunks).search("identity proof account holder", k=1)[0][0]
    assert "identity proof" in top.text
