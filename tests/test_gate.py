import pytest

from regrag.gate.gate import EvidenceGate, split_claims


class FakeNLI:
    def __init__(self, scores: dict[str, float]) -> None:
        self.scores = scores

    def entail(self, premise: str, hypothesis: str) -> float:
        return next((v for k, v in self.scores.items() if k in hypothesis), 0.0)


class FakeJudge:
    def __init__(self, reply: str) -> None:
        self.reply, self.calls = reply, 0

    def generate(self, system: str, user: str) -> str:
        self.calls += 1
        return self.reply


def test_split_claims_drops_citations_and_footer():
    ans = "The limit is Rs 50,000 [1]. It applies monthly [2].\nUpdated as on: October 1, 2026"
    assert split_claims(ans) == ["The limit is Rs 50,000.", "It applies monthly."]


def test_high_nli_passes_without_judge():
    j = FakeJudge("NO")
    g = EvidenceGate("nli_plus_judge", FakeNLI({"limit": 0.95}), j)
    d = g.check("The limit is Rs 50,000 [1].", ["p"])
    assert d.supported and j.calls == 0


def test_low_nli_fails_without_judge():
    j = FakeJudge("YES")
    g = EvidenceGate("nli_plus_judge", FakeNLI({"limit": 0.1}), j)
    assert not g.check("The limit is Rs 50,000 [1].", ["p"]).supported and j.calls == 0


def test_borderline_goes_to_judge():
    yes, no = FakeJudge("YES"), FakeJudge("NO")
    nli = FakeNLI({"limit": 0.6})  # threshold 0.7, band 0.15 -> 0.55..0.7 is borderline
    assert EvidenceGate("nli_plus_judge", nli, yes).check("The limit is Rs 50,000 [1].", ["p"]).supported
    assert not EvidenceGate("nli_plus_judge", nli, no).check("The limit is Rs 50,000 [1].", ["p"]).supported


def test_one_bad_claim_blocks_answer():
    g = EvidenceGate("nli_plus_judge", FakeNLI({"limit": 0.95, "monthly": 0.05}), FakeJudge("NO"))
    assert not g.check("The limit is Rs 50,000 [1]. It applies monthly always [2].", ["p"]).supported


def test_modes_and_validation():
    assert EvidenceGate("none").check("anything here at all", []).supported
    assert EvidenceGate("llm_judge", judge=FakeJudge("YES")).check("The limit is Rs 50,000 [1].", ["p"]).supported
    with pytest.raises(NotImplementedError):
        EvidenceGate("reranker_threshold")
    with pytest.raises(ValueError):
        EvidenceGate("nli_plus_judge", judge=FakeJudge("YES"))
