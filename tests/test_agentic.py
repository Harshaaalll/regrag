from regrag.retrieval.agentic import AgenticRetriever, merge
from regrag.types import Chunk


def _c(i: str) -> Chunk:
    return Chunk(id=i, doc_id="d", text=i, page=1)


class Base:
    def __init__(self) -> None:
        self.seen: list[str] = []

    def search(self, q: str, k: int = 5):
        self.seen.append(q)
        return {"A": [(_c("a1"), 0), (_c("a2"), 0)], "B": [(_c("b1"), 0)], "formal": [(_c("f1"), 0)]}.get(q, [(_c("z"), 0)])


class ScriptedLLM:
    """Replies by the system prompt: decompose / enough / rewrite."""

    def __init__(self, subs: str, enough: str, rewrite: str = "formal") -> None:
        self.subs, self.enough, self.rewrite = subs, enough, rewrite

    def generate(self, system: str, user: str) -> str:
        if "split it into" in system:
            return self.subs
        if "Reply with one word" in system:
            return self.enough
        return self.rewrite


def test_merge_interleaves_and_dedupes():
    out = merge([[(_c("a1"), 0), (_c("a2"), 0)], [(_c("b1"), 0), (_c("a1"), 0)]], 5)
    assert [c.id for c, _ in out] == ["a1", "b1", "a2"]


def test_decomposed_question_retrieves_for_each_part():
    base = Base()
    r = AgenticRetriever(base, ScriptedLLM("A\nB", "YES"))
    ids = [c.id for c, _ in r.search("compare A and B", 5)]
    assert base.seen == ["A", "B"] and ids == ["a1", "b1", "a2"] and r.trace[0]["rounds"] == 1


def test_not_enough_triggers_one_refinement():
    base = Base()
    r = AgenticRetriever(base, ScriptedLLM("A", "NO"), max_rounds=2)
    ids = [c.id for c, _ in r.search("what about A", 5)]
    assert "formal" in base.seen and "f1" in ids and r.trace[0]["rounds"] == 2


def test_refinement_is_capped():
    base = Base()
    r = AgenticRetriever(base, ScriptedLLM("A", "NO"), max_rounds=3)
    r.search("what about A", 5)
    assert r.trace[0]["rounds"] == 3
