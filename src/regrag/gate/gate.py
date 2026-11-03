from __future__ import annotations

import re

from regrag.gate.grounding import ungrounded_numbers
from regrag.types import GateDecision

JUDGE_SYSTEM = (
    "You check whether a statement is supported by a source text from an RBI directive. "
    "Answer YES only if the source text clearly states or directly implies the statement. Otherwise answer NO. "
    "Reply with one word: YES or NO."
)
_CITE = re.compile(r"\s*\[\d+\]")


def split_claims(answer: str) -> list[str]:
    """Split an answer into short claims; drop citation markers, bullets and the 'Updated as on' footer."""
    out = []
    for line in re.split(r"(?<=[.!?])\s+|\n+", answer):
        line = _CITE.sub("", line).strip(" -*•\t")
        if len(line.split()) >= 3 and not line.lower().startswith(("updated as on", "note:")):
            out.append(line)
    return out


class EvidenceGate:
    """Does the cited evidence support each claim? modes: none | llm_judge | nli_plus_judge."""

    def __init__(self, mode: str = "nli_plus_judge", nli=None, judge=None, entail_threshold: float = 0.7, band: float = 0.15,
                 check_numbers: bool = False) -> None:
        if mode == "reranker_threshold":
            raise NotImplementedError("reranker_threshold runs before generation: pass RetrievalGate to the pipeline (retrieval_gate=)")
        if mode not in ("none", "llm_judge", "nli_plus_judge"):
            raise ValueError(f"unknown gate mode {mode!r}")
        if mode == "nli_plus_judge" and nli is None:
            raise ValueError("nli_plus_judge needs an NLI model")
        if mode in ("llm_judge", "nli_plus_judge") and judge is None:
            raise ValueError(f"{mode} needs a judge LLM")
        self.mode, self.nli, self.judge = mode, nli, judge
        self.threshold, self.band = entail_threshold, band
        self.check_numbers = check_numbers

    def _judge_says_yes(self, claim: str, premise: str) -> bool:
        reply = self.judge.generate(JUDGE_SYSTEM, f"Source text:\n{premise[:3000]}\n\nStatement: {claim}\n\nSupported?")
        return reply.strip().upper().startswith("YES")

    def check(self, answer: str, premises: list[str]) -> GateDecision:
        if self.mode == "none":
            return GateDecision(supported=True, score=1.0, method="none")
        claims = split_claims(answer)
        if not claims:
            return GateDecision(supported=False, score=0.0, method=self.mode)
        premise = "\n".join(premises)
        scores, ok = [], []
        for claim in claims:
            if self.check_numbers and ungrounded_numbers(claim, premise):  # invented figure: fail without a model call
                scores.append(0.0)
                ok.append(False)
                continue
            if self.mode == "llm_judge":
                yes = self._judge_says_yes(claim, premise)
                scores.append(1.0 if yes else 0.0)
                ok.append(yes)
                continue
            s = self.nli.entail(premise, claim)
            scores.append(s)
            if s >= self.threshold:
                ok.append(True)
            elif s >= self.threshold - self.band:  # borderline: let the judge decide
                ok.append(self._judge_says_yes(claim, premise))
            else:
                ok.append(False)
        return GateDecision(supported=all(ok), score=min(scores), method=self.mode)
