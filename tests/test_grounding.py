from regrag.gate.gate import EvidenceGate
from regrag.gate.grounding import numbers_in, support_rate, ungrounded_numbers
from tests.test_gate import FakeJudge, FakeNLI


def test_numbers_normalised():
    assert numbers_in("limit Rs 50,000 within ten days, 2.5 percent") == {"50000", "10", "2.5"}


def test_invented_figure_is_found():
    assert ungrounded_numbers("The limit is Rs 60,000.", "The limit is Rs 50,000 per year.") == {"60000"}
    assert ungrounded_numbers("The limit is Rs 50,000.", "Limit: 50000 rupees") == set()


def test_support_rate():
    assert support_rate([True, False, True, True]) == 0.75 and support_rate([]) == 1.0


def test_gate_number_check_blocks_even_when_nli_agrees():
    nli = FakeNLI({"limit": 0.95})
    claim = "The limit is Rs 60,000 [1]."
    assert EvidenceGate("nli_plus_judge", nli, FakeJudge("YES")).check(claim, ["limit Rs 50,000"]).supported
    g = EvidenceGate("nli_plus_judge", nli, FakeJudge("YES"), check_numbers=True)
    assert not g.check(claim, ["limit Rs 50,000"]).supported
