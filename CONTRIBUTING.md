# Contributing to RegRAG

Thanks for helping. Small, clear changes are the easiest to review.

## Run it locally

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows; on Linux/macOS: source .venv/bin/activate
pip install -e ".[dev]"
pytest -q                     # unit tests use fake models: no downloads, no Ollama
```

Running real questions also needs [Ollama](https://ollama.com) with a model (`qwen3:4b`, or `qwen3:1.7b` on low RAM via `REGRAG_LLM_MODEL`) and an index built from RBI PDFs (`scripts/build_index.py`).

## RBI PDFs

RBI blocks scripted downloads, so save the PDFs by hand into `data/raw/`, then run `python scripts/register_pdfs.py`. Do not commit the PDFs. The starter list is in `data/SAMPLE_URLS.md`.

## Ground rules

- **No invented results.** README results tables stay `-` until a number is measured and the run is saved in `evals/results/`.
- **Golden questions** go in `evals/golden/`. Say who checked each one (`verified_by`). Do not mark LLM-checked questions as human-verified.
- New RAG variants implement `pipelines/base.py::Pipeline`. Keep modules small, with type hints and few comments.
- Add or update a test for every change, then run `pytest -q` before opening a PR.

## Opening a pull request

1. Fork, create a branch, make one focused change.
2. Run `pytest -q`.
3. Open a PR that says what changed and why.

## Issues

I aim to reply to new issues and PRs within **3 days** (this is a side project; the target will tighten if the project grows).
