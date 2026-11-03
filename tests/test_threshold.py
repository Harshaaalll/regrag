import pytest

from regrag.gate.threshold import RetrievalGate, best_threshold
from regrag.pipelines.rag import RagPipeline
from regrag.retrieval.rerank import RerankedRetriever
from regrag.types import Chunk


def _c(i: str) -> Chunk:
    return Chunk(id=i, doc_id="d", text=i, page=1)


class FakeReranker:
    def rerank(self, query, hits, k=5):
        return sorted(((c, float(c.id[1:])) for c, _ in hits), key=lambda x: -x[1])[:k]


class Base:
    def search(self, q, k=5):
        return [(_c(f"c{i}"), 0.0) for i in range(k)]


class Retr:
    def __init__(self, score: float) -> None:
        self.score = score

    def search(self, q, k=5):
        return [(_c("c1"), self.score)]


class CountingLLM:
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, system, user):
        self.calls += 1
        return "Customer due diligence is required [1]."


def test_gate_blocks_low_best_score():
    g = RetrievalGate(0.5)
    assert g.allows([(_c("a"), 0.2), (_c("b"), 0.6)]) and not g.allows([(_c("a"), 0.2)]) and not g.allows([])


def test_best_threshold_separates():
    t, bal = best_threshold([0.9, 0.8, 0.7], [0.2, 0.3])
    assert bal == 1.0 and 0.3 < t <= 0.7
    with pytest.raises(ValueError):
        best_threshold([0.9], [])


def test_reranked_retriever_uses_pool_then_keeps_k():
    out = RerankedRetriever(Base(), FakeReranker(), pool=6).search("q", 2)
    assert [c.id for c, _ in out] == ["c5", "c4"] and out[0][1] == 5.0


def test_pipeline_abstains_before_llm_when_score_low():
    llm = CountingLLM()
    p = RagPipeline(Retr(0.1), llm, retrieval_gate=RetrievalGate(0.5))
    a = p.answer("What does the KYC directive require for customer due diligence?")
    assert a.abstained and llm.calls == 0
    p2 = RagPipeline(Retr(0.9), llm, retrieval_gate=RetrievalGate(0.5))
    assert not p2.answer("What does the KYC directive require for customer due diligence?").abstained and llm.calls == 1
