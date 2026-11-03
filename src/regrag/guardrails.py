from __future__ import annotations

import re

OFF_TOPIC_MESSAGE = "I can only answer questions about the indexed RBI Master Directions."
ADVICE_NOTE = "Note: this explains what the Directions say. It is not legal, compliance or financial advice."

# Order matters: Aadhaar (12 digits) before the generic long-number account rule.
_PII = [
    ("AADHAAR", re.compile(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}\b")),
    ("PAN", re.compile(r"\b[A-Za-z]{5}\d{4}[A-Za-z]\b")),
    ("PHONE", re.compile(r"(?<!\d)(?:\+91[\s-]?|0)?[6-9]\d{9}(?!\d)")),
    ("ACCOUNT", re.compile(r"\b\d{9,18}\b")),
]


def mask_pii(text: str) -> tuple[str, list[str]]:
    """Replace Aadhaar / PAN / phone / account numbers with [TYPE]; return masked text and the types found."""
    found: list[str] = []
    for label, rx in _PII:
        text, n = rx.subn(f"[{label}]", text)
        if n:
            found.append(label)
    return text, found


_DOMAIN = re.compile(
    r"\b(rbi|reserve bank|bank|banks|banking|kyc|know your customer|customer|customers|account|accounts|aadhaar|"
    r"pan|directions?|master direction|loan|loans|credit|debit|card|cards|deposit|deposits|regulat\w*|nbfc|fpi|"
    r"transaction|transactions|due diligence|cdd|edd|ckycr?|v-cip|beneficial owner\w*|risk|capital|provision\w*|"
    r"asset|assets|compliance|fiu|str|ctr|money laundering|aml|cft|sanction\w*|outsourc\w*|climate|governance|"
    r"digital|interest|penalt\w*|record|records|small account|pep|politically exposed|wire transfer|ovd|"
    r"identification|identity|verification|onboarding|director|designated|principal officer|updation|"
    r"adequacy|liquidity|stressed|npa|repo|prudential|amend\w*|updated|section|paragraph|chapter)\b",
    re.I,
)


def is_off_topic(question: str) -> bool:
    """Rule-based first pass: no banking-regulation vocabulary at all means off topic."""
    return _DOMAIN.search(question) is None


_ADVICE = re.compile(
    r"\bshould (i|we|my|our|the bank)\b|\bis (it|this) (legal|allowed|ok|okay|safe)\b|\bis my\b.*\b(valid|compliant|ok|okay|approved)\b"
    r"|\bcan (i|we) (avoid|evade|bypass|skip|get around)\b|\bhow (do|can) (i|we) (avoid|evade|bypass|skip)\b|\badvi[cs]e\b"
    r"|\bwhat should (i|we)\b|\bam i (liable|compliant|eligible)\b",
    re.I,
)


def is_advice_request(question: str) -> bool:
    return _ADVICE.search(question) is not None
