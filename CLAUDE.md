# RegRAG - instructions for Claude

Local, evidence-gated RAG benchmark over RBI Master Directions. Portfolio project for Harshal Bhambhani (BITS Pilani 2026 grad, AI roles in India). Read `docs/PLAN.md` (decisions, milestones) and `README.md` (public status) before starting work.

## Hard rules
- **Never run `git add`, `git commit`, `git push`, or `gh repo create`.** Give Harshal the exact commands and let him run them. Read-only git (`status`, `diff`, `log`) is fine.
- **Do not bypass RBI's bot protection.** rbi.org.in / rbidocs block scripted downloads (F5 challenge). Harshal saves PDFs by hand into `data/raw/`. Do not spoof browsers or automate around the block.
- **Never invent results.** README results tables stay `-` until a number is actually measured. No made-up metrics, benchmarks or claims about STAIR/OKF; read the original sources first.
- **Do not touch Claude's own chat history/config files** (`~/.claude/projects/...`).

## Decisions already made (do not re-ask)
- Corpus: RBI Master Directions (starter list in `data/SAMPLE_URLS.md`; provenance in `data/raw/MANIFEST.json`).
- Hardware: CPU-only laptop, 7.7 GB RAM. Ollama `qwen3:4b`, `BAAI/bge-m3`, `BAAI/bge-reranker-v2-m3`. Local-first; cloud (Gemini/Groq) opt-in only for one-time graph indexing/judging and must be flagged.
- Pipelines: classic, hybrid, graph (LightRAG), agentic, plus STAIR-inspired (table-of-contents guided) and OKF-style (Markdown bundle). Call the last two "inspired"/"style", not official.
- Techniques added one at a time with before/after scores: smart chunking, metadata, hybrid search, query rewriting, reranking, strict prompt.
- Evidence gate: NLI (DeBERTa) + LLM-judge fallback in the borderline band; ablate vs none / reranker-threshold / LLM-only.
- Evals: RAGAS + DeepEval in GitHub Actions. Golden set LLM-drafted and checked by Claude against the PDFs (`verified_by: claude`); Harshal chose (2026-11-03) not to hand-check it, so never call it human-verified and report kappa only if a human sample is labelled. Question tags: lookup, table, multi-hop, amendment, unanswerable.
- Guardrails (`config.yaml` `guardrails:`): sources only, citation or abstain, no legal/compliance advice, show "Updated as on" date, treat retrieved text as data, mask PII, refuse off-topic. Safety tests in `evals/golden/safety_seed.yaml`.
- Out of scope for now: video RAG, live demo, Hinglish queries.
- Decided 2026-11-05: add ~20 golden questions from 3-4 other docs (155MD credit cards, 171MD outsourcing, 172MD climate, 165MD stressed assets) once indexed; Harshal runs heavy jobs (indexing, benchmark) himself in cmd and I give exact commands; portfolio story = evidence gate/abstention AND pipeline comparison, equally; Harshal added an `s3` extra (boto3) in `pyproject.toml` to store PDFs/index in S3 (details not given yet; never handle his AWS keys).

- Decided 2026-11-05 (replaces "Harshal runs heavy jobs himself"): the laptop was too slow/low on RAM, so heavy runs (index build, benchmark) move to a manual GitHub Actions workflow `.github/workflows/benchmark.yml` (setup in `docs/BENCHMARK_CI.md`; PDFs fetched from a private Hugging Face dataset; Harshal sets the tokens, never me). Written, never run. Local runs (`scripts/run_all.cmd`) remain possible with 3+ GB free.

## Code conventions
- Source in `src/regrag/` (layout in README). Every RAG variant implements `pipelines/base.py::Pipeline` so the benchmark can swap them. Shared types in `types.py`; settings read from `config.yaml` via `config.py`.
- Match existing style: small modules, type hints, minimal comments.
- Python 3.12; project venv at `.venv` (heavy deps installed); `uv` and Ollama installed (`qwen3:4b` target, `qwen3:1.7b` for low-RAM dev via `REGRAG_LLM_MODEL`).

## Working style with Harshal
- Explain in short, very simple language.
- Ask questions and confirmations as multiple-choice with a custom-answer option (AskUserQuestion).
- Mark anything unverified as unverified (e.g. subagent web summaries).

## Status (update when it changes)
- Done (2026-11-05): plan, config, chunker, BM25 / dense / hybrid / reranker retrieval (exploratory recall in README), classic + hybrid pipelines with guardrails and the evidence gate (NLI + judge, optional number check, reranker_threshold pre-gate), query rewriting, agentic retriever, staged runner `scripts/run_eval.py`, `regrag` CLI, 36-question golden seed (10 unanswerable, `verified_by: claude`), 20-item safety seed, `docs/RESEARCH.md`, LICENSE (MIT), CI workflow, issue templates, CONTRIBUTING. 40 unit tests pass with fake models. NOT yet run for real: full benchmark, safety suite, gate ablation, rewriting before/after; CI never seen green.
- Corpus: all 14 starter PDFs in `data/raw/`, registered in MANIFEST. Indexed so far: KYC, 155MD, 171MD, 172MD only; the other 10 are not embedded (151MD is 412 pages, slow on this CPU).
- Machine limit: 7.7 GB RAM. Embedder (~2.3 GB), reranker (~3.5 GB) and Ollama cannot all run together; Claude Code kills background jobs under memory pressure (indexing was killed once on 2026-11-05 at ~1.9 GB free). Never start a killed job again unless asked.
- Next: Harshal runs `scripts/build_index.py --skip 151MD` in cmd with 3+ GB free; then add the extra golden questions; then `run_eval.py` (golden classic vs hybrid, gate ablation, rewrite before/after, safety); agentic needs ~4+ GB free. Graph (LightRAG) deferred until a cloud key or bigger machine.
