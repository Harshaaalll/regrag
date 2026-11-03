from regrag.retrieval.rewrite import QueryRewriter, RewritingRetriever
from regrag.types import Chunk


def _c(i: str) -> Chunk:
    return Chunk(id=i, doc_id="d", text=i, page=1)


class FakeLLM:
    def __init__(self, reply: str) -> None:
        self.reply = reply

    def generate(self, system: str, user: str) -> str:
        return self.reply


class QueryAware:
    def __init__(self) -> None:
        self.seen: list[str] = []

    def search(self, query: str, k: int = 20):
        self.seen.append(query)
        return [(_c("orig"), 0.0)] if query == "limits" else [(_c("formal"), 0.0), (_c("orig"), 0.0)]


def test_rewrite_takes_first_line_and_strips_quotes():
    assert QueryRewriter(FakeLLM('"Small Account limits"\nextra')).rewrite("limits") == "Small Account limits"


def test_empty_rewrite_falls_back_to_original():
    assert QueryRewriter(FakeLLM("  ")).rewrite("limits") == "limits"


def test_both_queries_are_searched_and_fused():
    base = QueryAware()
    r = RewritingRetriever(base, QueryRewriter(FakeLLM("Small Account limits")))
    out = [c.id for c, _ in r.search("limits", k=5)]
    assert set(base.seen) == {"limits", "Small Account limits"} and out[0] == "orig" and "formal" in out
    assert r.last_rewrite == "Small Account limits"
