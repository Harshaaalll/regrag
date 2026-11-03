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
    return RagPipeline(FakeRetriever(), llm, abstain_message=ABSTAIN).answer("What is the small account limit?"), llm


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


def test_off_topic_refused_without_calling_llm():
    llm = FakeLLM("should not be used")
    a = RagPipeline(FakeRetriever(), llm, abstain_message=ABSTAIN).answer("Who won the cricket world cup?")
    assert a.abstained and "only answer questions" in a.text and llm.last_user == ""


def test_pii_is_masked_everywhere():
    a, llm = _run("The limit is Rs 50,000 [1].")
    llm2 = FakeLLM("The limit is Rs 50,000 [1].")
    a = RagPipeline(FakeRetriever(), llm2, abstain_message=ABSTAIN).answer("My PAN is ABCDE1234F, what is the small account limit?")
    assert "ABCDE1234F" not in a.question and "ABCDE1234F" not in llm2.last_user and "[PAN]" in a.question


def test_advice_question_gets_disclaimer():
    llm = FakeLLM("Paragraph 28 sets the limits [1].")
    a = RagPipeline(FakeRetriever(), llm, abstain_message=ABSTAIN).answer("Should my bank open a small account?")
    assert not a.abstained and "not legal, compliance or financial advice" in a.text


class RejectGate:
    def check(self, answer, premises):
        from regrag.types import GateDecision
        return GateDecision(supported=False, score=0.1, method="fake")


def test_gate_rejection_turns_answer_into_abstention():
    p = RagPipeline(FakeRetriever(), FakeLLM("The limit is Rs 99 [1]."), abstain_message=ABSTAIN, gate=RejectGate())
    a = p.answer("What is the small account limit?")
    assert a.abstained and a.text == ABSTAIN and a.gate.method == "fake"
