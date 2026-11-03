from regrag.pipelines.rag import RagPipeline
from regrag.types import Chunk

CH = [
    Chunk(id="d:c0", doc_id="d", text="A small account has a balance limit of Rs 50,000.", page=35),
    Chunk(id="d:c1", doc_id="d", text="Records are kept for five years.", page=52),
]
ABSTAIN = "Not found in the indexed RBI documents."


class FakeRetriever:
    def search(self, query, k=20):
        return [(c, 1.0) for c in CH][:k]


class FakeLLM:
    def __init__(self, reply: str) -> None:
        self.reply, self.last_user = reply, ""

    def generate(self, system: str, user: str) -> str:
        self.last_user = user
        return self.reply


def _run(reply: str):
    llm = FakeLLM(reply)
    return RagPipeline(FakeRetriever(), llm, abstain_message=ABSTAIN).answer("q"), llm


def test_cited_answer_keeps_citations():
    a, llm = _run("The balance limit is Rs 50,000 [1].")
    assert not a.abstained and a.citations[0].page == 35 and "[1] (d, page 35)" in llm.last_user


def test_answer_without_citation_abstains():
    a, _ = _run("The balance limit is Rs 50,000.")
    assert a.abstained and a.text == ABSTAIN and not a.citations


def test_invalid_citation_number_abstains():
    a, _ = _run("It is Rs 50,000 [7].")
    assert a.abstained


def test_model_abstain_message_abstains():
    a, _ = _run(ABSTAIN)
    assert a.abstained
