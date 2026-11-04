"""Ask one question from the command line:  regrag "Is an Aadhaar number mandatory for KYC?"

Needs the index (scripts/build_index.py) and Ollama running with the configured model.
"""
from __future__ import annotations

import argparse

from regrag.pipelines import variants
from regrag.utils.manifest import doc_title


def main() -> None:
    ap = argparse.ArgumentParser(prog="regrag", description="Ask a question about the indexed RBI Master Directions.")
    ap.add_argument("question")
    ap.add_argument("--pipeline", default="hybrid", choices=["classic", "hybrid", "agentic"])
    a = ap.parse_args()

    ans = variants.build(a.pipeline).answer(a.question)
    print(ans.text)
    if ans.citations:
        print("\nSources:")
        for c in ans.citations:
            print(f"  - {doc_title(c.doc_id)}, page {c.page}")
    print(f"\n({'abstained' if ans.abstained else 'answered'} in {ans.latency_s:.1f}s, pipeline: {ans.pipeline})")


if __name__ == "__main__":
    main()
