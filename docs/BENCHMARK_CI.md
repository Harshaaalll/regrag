# Running the benchmark on GitHub Actions (free, no laptop needed)

**Status: written, never run.** Check the first run's logs before trusting it. Free-tier limits (public-repo minutes, private Hugging Face storage) are from memory and not verified; check GitHub's and Hugging Face's current pricing pages.

Why: a GitHub-hosted runner has far more RAM than the dev laptop, so the embedder, the reranker and Ollama can run in sequence without being killed. Public repos get free Actions minutes.

## One-time setup (you do this; Claude never handles tokens)

1. **Put the PDFs in a private Hugging Face dataset.**
   - Create a free account at huggingface.co, then New, Dataset, set visibility **Private**, name it e.g. `regrag-pdfs`.
   - Files, Upload files: drag in the 14 PDFs from `data/raw/` (keep the file names).
2. **Make a read token.** huggingface.co/settings/tokens, New token, type **Read**. Copy it once.
3. **Add it to the GitHub repo.** Repo, Settings, Secrets and variables, Actions:
   - Secrets tab, New repository secret: name `HF_TOKEN`, value = the token.
   - Variables tab, New repository variable: name `REGRAG_DATA_REPO`, value = `yourname/regrag-pdfs`.
4. **Run it.** Actions tab, benchmark, Run workflow. For a first test set `limit` to `3` (3 questions per stage).
5. **Get the results.** Open the finished run, download the `eval-results` artifact, and send the JSON files to Claude to fill the README tables.

## Notes
- The PDFs never enter this repo or the logs. Only the artifact (answers and scores) is uploaded.
- The workflow installs Ollama with its official install script (`curl ... | sh`), standard for CI; read it first if you prefer.
- A full run (index build, model downloads, about 6 benchmark stages on CPU) can take hours; the first run is the slowest, later runs reuse caches.
- Local runs still work (`scripts/run_all.cmd`) when the laptop has the RAM.
