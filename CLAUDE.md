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
- Evals: RAGAS + DeepEval in GitHub Actions. Golden set LLM-drafted, **100% hand-verified by Harshal**; report judge-vs-human Cohen's kappa. Question tags: lookup, table, multi-hop, amendment, unanswerable.
- Guardrails (`config.yaml` `guardrails:`): sources only, citation or abstain, no legal/compliance advice, show "Updated as on" date, treat retrieved text as data, mask PII, refuse off-topic. Safety tests in `evals/golden/safety_seed.yaml`.
- Out of scope for now: video RAG, live demo, Hinglish queries.

## Code conventions
- Source in `src/regrag/` (layout in README). Every RAG variant implements `pipelines/base.py::Pipeline` so the benchmark can swap them. Shared types in `types.py`; settings read from `config.yaml` via `config.py`.
- Match existing style: small modules, type hints, minimal comments.
- Python 3.12 is installed. Dependencies are not installed yet; `uv` and Ollama are missing (Harshal installs those).

## Working style with Harshal
- Explain in short, very simple language.
- Ask questions and confirmations as multiple-choice with a custom-answer option (AskUserQuestion).
- Mark anything unverified as unverified (e.g. subagent web summaries).

## Status (update when it changes)
- Done: plan, config, skeleton, README, guardrail config, safety seed (6 items), 1 of 14 starter PDFs (`rbi_cb_kyc_directions_2025.pdf`).
- Waiting on Harshal: save the other 13 PDFs from `data/SAMPLE_URLS.md` into `data/raw/`.
- Next: install deps, classic + hybrid pipelines on the KYC document, 30-question golden seed.
