"""Download the RBI PDFs from a private Hugging Face dataset into data/raw/ (used by the benchmark workflow).

RBI blocks scripted downloads, so the PDFs are saved by hand once, uploaded to your own private dataset repo,
and fetched from there. Needs env vars: REGRAG_DATA_REPO (e.g. "yourname/regrag-pdfs") and HF_TOKEN (read access).
MANIFEST.json in this repo is kept as is; only *.pdf files are downloaded.
"""
from __future__ import annotations

import os
import sys

from huggingface_hub import snapshot_download

from regrag.config import ROOT, load_config


def main() -> None:
    repo, token = os.environ.get("REGRAG_DATA_REPO"), os.environ.get("HF_TOKEN")
    if not repo or not token:
        sys.exit("Set REGRAG_DATA_REPO and HF_TOKEN (see docs/BENCHMARK_CI.md).")
    raw = ROOT / load_config()["paths"]["raw"]
    snapshot_download(repo_id=repo, repo_type="dataset", token=token, local_dir=raw, allow_patterns=["*.pdf", "*.PDF"])
    print(f"{len(list(raw.glob('*.[pP][dD][fF]')))} PDFs in {raw}")


if __name__ == "__main__":
    main()
