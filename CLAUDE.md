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

## Code conventions
- Source in `src/regrag/` (layout in README). Every RAG variant implements `pipelines/base.py::Pipeline` so the benchmark can swap them. Shared types in `types.py`; settings read from `config.yaml` via `config.py`.
- Match existing style: small modules, type hints, minimal comments.
- Python 3.12; project venv at `.venv` (heavy deps installed); `uv` and Ollama installed (`qwen3:4b` target, `qwen3:1.7b` for low-RAM dev via `REGRAG_LLM_MODEL`).

## Working style with Harshal
- Explain in short, very simple language.
- Ask questions and confirmations as multiple-choice with a custom-answer option (AskUserQuestion).
- Mark anything unverified as unverified (e.g. subagent web summaries).

## Status (update when it changes)
- Done (2026-11-04): plan, config, guardrail config, safety seed (6 items), chunker, BM25 / dense / hybrid / reranker retrieval (exploratory recall numbers in README), classic + hybrid pipelines (smoke-tested only), 30-question golden seed (`verified_by: claude`), scripts to register PDFs and build the index.
- Corpus: 5 of 14 starter PDFs in `data/raw/` (KYC, 155MD, 171MD, 172MD, 151MD). Indexed: all except 151MD (412 pages, too slow on this CPU).
- Machine limit: 7.7 GB RAM. Embedder (~2.3 GB), reranker (~3.5 GB) and Ollama cannot all run together; background jobs get killed. Run models in stages, close Chrome/VS Code, never start a killed job again unless asked.
- Waiting on Harshal: save the remaining 9 PDFs (Chrome download automation does not work on his managed profile).
- Next: benchmark runner (classic vs hybrid over the golden set), evidence gate, query rewriting, PII/off-topic guardrails, then graph/agentic.
