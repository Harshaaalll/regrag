from regrag.retrieval.hybrid import HybridRetriever
from regrag.types import Chunk


def _c(i: str) -> Chunk:
    return Chunk(id=i, doc_id="d", text=i, page=1)


class Fixed:
    def __init__(self, ids: list[str]) -> None:
        self.ids = ids

    def search(self, query: str, k: int = 20):
        return [(_c(i), 0.0) for i in self.ids[:k]]


def test_rrf_prefers_chunk_ranked_well_by_both():
    h = HybridRetriever([Fixed(["a", "b", "c"]), Fixed(["c", "b", "d"])])
    top2 = {c.id for c, _ in h.search("q", k=4)[:2]}
    assert top2 == {"b", "c"}  # both appear in both lists; a and d appear in only one


def test_rrf_respects_k_and_dedupes():
    h = HybridRetriever([Fixed(["a", "b"]), Fixed(["a", "b"])])
    out = h.search("q", k=1)
    assert [c.id for c, _ in out] == ["a"]
