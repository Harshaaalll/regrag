"""Record hand-saved PDFs in data/raw/MANIFEST.json (page count, SHA-256, source URL).

Save each PDF under its original name (e.g. 151MD.PDF) so the URL can be matched
from data/SAMPLE_URLS.md. Run: python scripts/register_pdfs.py
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from datetime import date
from pathlib import Path

import fitz

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
MANIFEST = RAW / "MANIFEST.json"
SAMPLES = ROOT / "data" / "SAMPLE_URLS.md"


def sample_urls() -> dict[str, str]:
    urls = re.findall(r"https://rbidocs\.rbi\.org\.in/\S+?\.PDF", SAMPLES.read_text(encoding="utf-8"), re.I)
    return {Path(u).stem.lower(): u for u in urls}


def main() -> None:
    entries = json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.exists() else []
    known = {e["file"] for e in entries}
    urls = sample_urls()
    for pdf in sorted(RAW.glob("*.[pP][dD][fF]")):
        if pdf.name in known:
            continue
        with fitz.open(pdf) as doc:
            pages = len(doc)
            title = " ".join(doc[0].get_text("text").split())[:160] if pages else ""
        url = urls.get(pdf.stem.lower())
        entries.append({
            "file": pdf.name, "title": title, "source_url": url,
            "retrieved": date.today().isoformat(), "method": "browser save",
            "pages": pages, "sha256": hashlib.sha256(pdf.read_bytes()).hexdigest(),
        })
        print(f"added {pdf.name} ({pages} pages)" + ("" if url else "  <- no URL matched, fill source_url by hand"))
    MANIFEST.write_text(json.dumps(entries, indent=1, ensure_ascii=False), encoding="utf-8")
    have = {Path(e["source_url"]).stem.lower() for e in entries if e.get("source_url")}
    missing = [s for s in urls if s not in have]
    print(f"{len(entries)} registered; still to save from SAMPLE_URLS.md: {len(missing)} {missing}")


if __name__ == "__main__":
    sys.exit(main())
