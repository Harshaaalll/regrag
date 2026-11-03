from __future__ import annotations

from regrag.retrieval.hybrid import HybridRetriever
from regrag.types import Chunk

REWRITE_SYSTEM = (
    "Rewrite the user's question as a short search query for finding it in RBI Master Directions "
    "(banking KYC, AML, credit and capital rules). Use the formal terms the Directions would use "
    "(e.g. 'Customer Due Diligence', 'Officially Valid Document', 'Periodic Updation'). "
    "Keep numbers and names. Output only the query on one line, no explanation."
)


class QueryRewriter:
    def __init__(self, llm) -> None:
        self.llm = llm

    def rewrite(self, question: str) -> str:
        out = self.llm.generate(REWRITE_SYSTEM, question).strip().splitlines()
        return out[0].strip(" \"'") if out and out[0].strip() else question  # fall back to the original


class _Fixed:
    """Retriever pinned to one query, so the fusion step can combine several queries."""

    def __init__(self, base, query: str) -> None:
        self.base, self.query = base, query

    def search(self, _query: str, k: int = 20) -> list[tuple[Chunk, float]]:
        return self.base.search(self.query, k)


class RewritingRetriever:
    """Search with the original and the rewritten query, then fuse both rankings (RRF)."""

    def __init__(self, base, rewriter: QueryRewriter, rrf_k: int = 60) -> None:
        self.base, self.rewriter, self.rrf_k = base, rewriter, rrf_k
        self.last_rewrite: str | None = None

    def search(self, query: str, k: int = 20) -> list[tuple[Chunk, float]]:
        rewritten = self.rewriter.rewrite(query)
        self.last_rewrite = rewritten
        if rewritten.strip().lower() == query.strip().lower():
            return self.base.search(query, k)
        return HybridRetriever([_Fixed(self.base, query), _Fixed(self.base, rewritten)], self.rrf_k).search(query, k)
