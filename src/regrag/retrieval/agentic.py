from __future__ import annotations

from regrag.retrieval.rewrite import QueryRewriter
from regrag.types import Chunk

DECOMPOSE_SYSTEM = (
    "If answering the question needs facts from more than one place (a comparison, a list across sections, "
    "or several conditions), split it into at most 3 short, self-contained search questions, one per line. "
    "Otherwise output the question unchanged on one line. Output only the lines, nothing else."
)
ENOUGH_SYSTEM = "Do the sources contain the information needed to answer the question? Reply with one word: YES or NO."


def merge(lists: list[list[tuple[Chunk, float]]], k: int) -> list[tuple[Chunk, float]]:
    """Interleave ranked lists (best of each first), drop duplicates, keep k."""
    seen, out = set(), []
    for rank in range(max((len(x) for x in lists), default=0)):
        for lst in lists:
            if rank < len(lst) and lst[rank][0].id not in seen:
                seen.add(lst[rank][0].id)
                out.append(lst[rank])
    return out[:k]


class AgenticRetriever:
    """route (decompose) -> retrieve per sub-question -> evaluate (enough evidence?) -> refine (rewrite and retry)."""

    def __init__(self, base, llm, max_rounds: int = 2) -> None:
        self.base, self.llm, self.max_rounds = base, llm, max_rounds
        self.rewriter = QueryRewriter(llm)
        self.trace: list[dict] = []  # per-question record of what the agent did

    def decompose(self, question: str) -> list[str]:
        lines = [x.strip(" -*•\t0123456789.)") for x in self.llm.generate(DECOMPOSE_SYSTEM, question).splitlines()]
        subs = [x for x in lines if x][:3]
        return subs or [question]

    def enough(self, question: str, hits: list[tuple[Chunk, float]]) -> bool:
        sources = "\n\n".join(f"[{i}] {c.text[:600]}" for i, (c, _) in enumerate(hits, 1))
        return self.llm.generate(ENOUGH_SYSTEM, f"Sources:\n{sources}\n\nQuestion: {question}").strip().upper().startswith("YES")

    def search(self, question: str, k: int = 5) -> list[tuple[Chunk, float]]:
        subs = self.decompose(question)
        hits = merge([self.base.search(s, k) for s in subs], k)
        rounds = 1
        while rounds < self.max_rounds and not self.enough(question, hits):
            hits = merge([hits, self.base.search(self.rewriter.rewrite(question), k)], k)
            rounds += 1
        self.trace.append({"question": question, "sub_questions": subs, "rounds": rounds})
        return hits
