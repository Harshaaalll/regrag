from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

from regrag.config import ROOT

_UPDATED = re.compile(r"Updated as on\s+([A-Za-z]+\s+\d{1,2},\s*\d{4})", re.I)


@lru_cache
def _titles() -> dict[str, str]:
    path = ROOT / "data" / "raw" / "MANIFEST.json"
    entries = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    return {Path(e["file"]).stem: e.get("title", "") for e in entries}


def doc_title(doc_id: str) -> str:
    return _titles().get(doc_id, doc_id)


def updated_as_on(doc_id: str) -> str | None:
    """The 'Updated as on <date>' text from the document title, if the title has one."""
    m = _UPDATED.search(doc_title(doc_id))
    return m.group(1) if m else None
