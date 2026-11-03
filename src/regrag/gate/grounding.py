from __future__ import annotations

import re

_NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")
_WORD_NUM = {"one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7", "eight": "8",
             "nine": "9", "ten": "10", "twelve": "12", "fifteen": "15", "thirty": "30", "sixty": "60", "ninety": "90"}


def numbers_in(text: str) -> set[str]:
    """Numbers as plain strings: '50,000' -> '50000'. Number words (one..ninety) are mapped to digits."""
    found = {m.group().replace(",", "").rstrip(".") for m in _NUM.finditer(text)}
    found |= {_WORD_NUM[w] for w in re.findall(r"[a-z]+", text.lower()) if w in _WORD_NUM}
    return found


def ungrounded_numbers(claim: str, premise: str) -> set[str]:
    """Numbers in the claim that appear nowhere in the source text (a cheap sign of an invented figure)."""
    return numbers_in(claim) - numbers_in(premise)


def support_rate(flags: list[bool]) -> float:
    """Share of claims that were supported (1.0 when there are no claims to doubt)."""
    return sum(flags) / len(flags) if flags else 1.0
