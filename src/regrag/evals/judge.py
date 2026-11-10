from __future__ import annotations

import os

GRADE_SYSTEM = (
    "You grade a question-answering system. Compare the system answer with the reference answer for the question. "
    "CORRECT = the system answer states the same key facts as the reference (extra detail is fine). "
    "PARTIAL = some key facts are right but one is missing or wrong. WRONG = it contradicts the reference or misses it. "
    "Ignore citation markers and 'Updated as on' lines. Reply with exactly one word: CORRECT, PARTIAL or WRONG."
)


def grade(llm, question: str, reference: str, answer: str) -> str:
    """CORRECT / PARTIAL / WRONG. Only the question and the two answers are sent, never the PDF text."""
    reply = llm.generate(GRADE_SYSTEM, f"Question: {question}\n\nReference answer: {reference}\n\nSystem answer: {answer}\n\nGrade:")
    word = reply.strip().upper().split()[0].strip(".:,") if reply.strip() else ""
    return word if word in ("CORRECT", "PARTIAL", "WRONG") else "UNGRADED"


class GeminiJudge:
    """Cloud judge (opt-in, flagged in results). Key from GEMINI_API_KEY; model name from the caller, not guessed here.

    Written against the google-genai SDK from memory and not yet run against the real API.
    """

    def __init__(self, model: str, api_key: str | None = None) -> None:
        from google import genai  # pip install -e ".[judge]"

        key = api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("Set GEMINI_API_KEY (as an environment variable or GitHub secret)")
        self._genai, self.model, self._client = genai, model, genai.Client(api_key=key)

    def generate(self, system: str, user: str) -> str:
        cfg = self._genai.types.GenerateContentConfig(system_instruction=system, temperature=0)
        return self._client.models.generate_content(model=self.model, contents=user, config=cfg).text or ""
