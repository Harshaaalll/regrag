# RegRAG — plan (confirmed decisions)

**Thesis:** No single RAG fits all. Measure which pipeline wins per question type, with an evidence gate that abstains when sources don't support an answer. Fully local at query time.

## Decisions
| Area | Choice |
|---|---|
| Corpus | RBI circulars + Master Directions (public PDFs) |
| Stack | Ollama (qwen3:4b), bge-m3 embeddings, bge-reranker-v2-m3; CPU-only machine (7.7 GB RAM) |
| Compute | Local-first; cloud (Gemini/Groq) opt-in only for one-time graph indexing / judging, flagged in README |
| Gate | NLI entailment (DeBERTa) + LLM-judge fallback in the borderline band; ablate vs none / reranker-threshold / LLM-only |
| Pipelines | classic (dense), hybrid (BM25+dense+RRF+rerank), graph (LightRAG), agentic (route/evaluate/refine) |
| Evals | RAGAS + DeepEval in GitHub Actions with thresholds; own golden set |
| Multimodal | Docling/MinerU tables+figures; table-QA subset shows text-only failures |
| Golden set | LLM-drafted with page citations, **100% hand-verified by Harshal**; report Cohen's kappa judge vs human |
| Out of scope (later) | Video RAG, live demo |
| Timeline | ~3 weeks |

## Milestones
1. **Wk1** corpus (30-50 docs) → ingestion → classic + hybrid baselines → 30-question golden seed
2. **Wk2** LightRAG graph pipeline → agentic pipeline → gate + ablation → Docling tables
3. **Wk3** RAGAS/DeepEval CI → full golden set (~100, tagged: lookup, table, multi-hop, amendment, unanswerable) → results tables, diagrams, README

## Question-type tags (drive the per-type winner table)
lookup · table · multi-hop · amendment-tracking · unanswerable

## Guardrails (added 2026-11-03)
**Answer-time** (config.yaml `guardrails:`): scope-only, citation required else abstain, no legal/compliance advice, show "Updated as on" date + stale warning, retrieved text treated as data (anti prompt-injection), PII masking, off-topic refusal.
**Safety test set:** `evals/golden/safety_seed.yaml` (~20 items: advice, off-topic, injection, unanswerable, PII, stale). Pass rate reported in README.
**Data rules:** official rbi.org.in / rbidocs sources only; every PDF recorded in `data/raw/MANIFEST.json` (URL, retrieved date, pages, SHA-256); no scripted bulk downloading (RBI blocks it) - save by hand; re-check "Updated as on" dates before each benchmark run.

## Added to scope (2026-11-03)
- Compare RAG vs STAIR-style (table-of-contents guided) vs OKF-style (Markdown knowledge bundle). Read the original sources before making any claims.
- Apply the 6 techniques one at a time with before/after scores: smart chunking, metadata, hybrid search, query rewriting, reranking, strict prompt.
