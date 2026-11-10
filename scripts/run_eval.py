"""Run the golden benchmark or the safety set, in stages so the embedder and Ollama are never in memory together.

  stage 0 (optional, LLM)  rewrite questions            --rewrite
  stage 1 (embedder)       retrieve top chunks per pipeline and cache them
  stage 2 (LLM + gate)     answer from the cached chunks and score

Examples (PowerShell, from the project root):
  $env:REGRAG_LLM_MODEL="qwen3:1.7b"; .venv\\Scripts\\python scripts\\run_eval.py --suite golden --pipelines classic hybrid --gate none
  .venv\\Scripts\\python scripts\\run_eval.py --suite safety --pipelines hybrid --gate nli_plus_judge --limit 5
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import time


from regrag.config import ROOT, load_config
from regrag.evals.benchmark import load_golden, run_benchmark, summarize
from regrag.evals.safety import load_safety, run_safety
from regrag.evals.split import split_items
from regrag.gate.gate import EvidenceGate
from regrag.guardrails import is_off_topic, mask_pii
from regrag.llm.ollama_client import OllamaLLM
from regrag.pipelines.base import REGISTRY
from regrag.pipelines import variants  # noqa: F401  (registers classic / hybrid)
from regrag.retrieval.agentic import AgenticRetriever
from regrag.retrieval.rewrite import QueryRewriter, RewritingRetriever
from regrag.types import Chunk


class Cached:
    def __init__(self, hits: dict[str, list]) -> None:
        self.hits = hits

    def search(self, q: str, k: int = 5):
        return self.hits[q][:k]


class DictRewriter:
    def __init__(self, table: dict[str, str]) -> None:
        self.table = table

    def rewrite(self, q: str) -> str:
        return self.table.get(q, q)


def free_ollama(model: str) -> None:
    subprocess.run(["ollama", "stop", model], capture_output=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", choices=["golden", "safety"], default="golden")
    ap.add_argument("--pipelines", nargs="+", default=["hybrid"], choices=["classic", "hybrid", "agentic"])
    ap.add_argument("--gate", default="none", choices=["none", "reranker_threshold", "llm_judge", "nli_plus_judge"])
    ap.add_argument("--check-numbers", action="store_true", help="gate also fails claims whose numbers are not in the cited text")
    ap.add_argument("--rerank", action="store_true", help="rerank the top 20 with bge-reranker (separate stage, own memory)")
    ap.add_argument("--min-score", type=float, default=None, help="reranker_threshold: abstain below this top score")
    ap.add_argument("--rewrite", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="only the first N questions (smoke runs)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--split", choices=["all", "train", "test"], default="all",
                    help="golden only: seeded split by question tag. Tune thresholds on train, report on test")
    ap.add_argument("--seed", type=int, default=42, help="seed for --split")
    ap.add_argument("--resume", action="store_true",
                    help="golden: keep finished answers in evals/results/<suite>_<tag>.<pipeline>.jsonl and skip them on a repeat run (needs --tag)")
    ap.add_argument("--dry-run", action="store_true", help="print the plan (questions, split, stages) and exit; no models are loaded")
    a = ap.parse_args()

    cfg = load_config()
    model = os.environ.get("REGRAG_LLM_MODEL") or cfg["llm"]["generator"]["model"]
    rerank_planned = a.rerank or a.gate == "reranker_threshold"
    res_dir = ROOT / "evals" / "results"
    if a.resume and not a.tag:
        raise SystemExit("--resume needs a fixed --tag so the repeat run finds the same checkpoint files")
    if a.suite == "golden":
        items = load_golden(ROOT / cfg["paths"]["golden"])
    else:
        items = load_safety(ROOT / cfg["paths"]["golden"] / "safety_seed.yaml")
    if a.split != "all":
        if a.suite != "golden":
            raise SystemExit("--split only applies to the golden suite")
        items = split_items(items, a.seed)[0 if a.split == "train" else 1]
    if a.limit:
        items = items[: a.limit]
    if a.dry_run:
        by_tag: dict[str, int] = {}
        for it in items:
            kind = it.get("tag", it.get("type", "?"))
            by_tag[kind] = by_tag.get(kind, 0) + 1
        stages = (["rewrite"] if a.rewrite else []) + ["retrieve"] + (["rerank"] if rerank_planned else []) + ["answer"]
        print(f"DRY RUN: suite={a.suite} split={a.split} seed={a.seed} model={model} gate={a.gate} "
              f"check_numbers={a.check_numbers} pipelines={a.pipelines}")
        print(f"  {len(items)} questions {by_tag}")
        print(f"  stages: {' -> '.join(stages)}")
        return
    # what the pipeline will actually retrieve for: masked questions that are on topic
    qs = [mask_pii(it["q"])[0] for it in items]
    qs = [q for q in qs if not is_off_topic(q)]

    rewrites: dict[str, str] = {}
    if a.rewrite:
        print("stage 0: rewriting questions", flush=True)
        rw = QueryRewriter(OllamaLLM(model))
        rewrites = {q: rw.rewrite(q) for q in qs}
        free_ollama(model)

    print("stage 1: retrieval", flush=True)
    cache: dict[str, dict[str, list]] = {}
    top_n = cfg["reranker"]["top_n"]
    rerank = a.rerank or a.gate == "reranker_threshold"
    pool = 20 if rerank else top_n
    min_score = a.min_score if a.min_score is not None else cfg["gate"].get("reranker_min_score")
    for name in a.pipelines:
        # agentic needs the embedder AND the LLM at the same time (about 4+ GB free RAM)
        base = variants.make_retriever("hybrid" if name == "agentic" else name)
        corpus_docs = base.docs
        print("corpus:", corpus_docs, flush=True)
        if name == "agentic":
            base = AgenticRetriever(base, OllamaLLM(model))
        retr = RewritingRetriever(base, DictRewriter(rewrites), cfg["retrieval"]["rrf_k"]) if a.rewrite else base
        cache[name] = {q: retr.search(q, pool) for q in qs}
        del base, retr
        gc.collect()

    if rerank:  # the embedder is freed above; now the reranker gets the memory
        print("stage 1b: reranking", flush=True)
        from regrag.retrieval.rerank import Reranker
        rr = Reranker(cfg["reranker"]["model"])
        for name in a.pipelines:
            cache[name] = {q: rr.rerank(q, h, top_n) for q, h in cache[name].items()}
        del rr
        gc.collect()
        if a.suite == "golden":
            from regrag.gate.threshold import best_threshold
            tag_of = {mask_pii(it["q"])[0]: it["tag"] for it in items}
            for name in a.pipelines:
                top = {q: max((s for _, s in h), default=0.0) for q, h in cache[name].items()}
                ans = [s for q, s in top.items() if tag_of[q] != "unanswerable"]
                una = [s for q, s in top.items() if tag_of[q] == "unanswerable"]
                if ans and una:
                    t, bal = best_threshold(ans, una)
                    print(f"{name}: suggested min-score {t:.3f} (balanced accuracy {bal:.2f} on {len(ans)} answerable and "
                          f"{len(una)} unanswerable questions; {'tuned on the train split: now run with --split test --min-score' if a.split == 'train' else 'the same questions used to pick it, so it is optimistic; use --split train, then --split test'})", flush=True)

    print("stage 2: answering", flush=True)
    llm = OllamaLLM(model)
    gcfg = cfg["gate"]
    nli = None
    if a.gate == "nli_plus_judge":
        from regrag.gate.nli import NLI
        nli = NLI(gcfg["nli_model"])
    gate = None if a.gate in ("none", "reranker_threshold") else EvidenceGate(a.gate, nli, llm, gcfg["entail_threshold"], gcfg["borderline_band"],
                                                    check_numbers=a.check_numbers)
    retrieval_gate = None
    if a.gate == "reranker_threshold":
        if min_score is None:
            raise SystemExit("reranker_threshold needs --min-score (or gate.reranker_min_score in config.yaml); "
                             "run once with --rerank to get a suggestion")
        from regrag.gate.threshold import RetrievalGate
        retrieval_gate = RetrievalGate(min_score)
    g = cfg["guardrails"]

    out = {"suite": a.suite, "model": model, "gate": a.gate, "rewrite": a.rewrite, "check_numbers": a.check_numbers, "rerank": rerank, "min_score": min_score if retrieval_gate else None, "split": a.split, "seed": a.seed, "question_ids": [it["id"] for it in items], "n_questions": len(items),
           "corpus_docs": corpus_docs,"date": time.strftime("%Y-%m-%d %H:%M"), "pipelines": {}}
    for name in a.pipelines:
        pipe = REGISTRY[name](Cached(cache[name]), llm, top_n=top_n, abstain_message=g["abstain_message"], gate=gate,
                              redact_pii=g["redact_pii"], block_off_topic=g["block_off_topic"], no_advice=g["no_advice"],
                              retrieval_gate=retrieval_gate)
        if a.suite == "golden":
            ckpt = res_dir / f"golden_{a.tag}.{name}.jsonl" if a.resume else None
            rows = run_benchmark(pipe, items, checkpoint=ckpt)
            out["pipelines"][name] = {"summary": summarize(rows), "rows": rows}
        else:
            out["pipelines"][name] = run_safety(pipe, items)
        print(name, json.dumps(out["pipelines"][name].get("summary", {k: v for k, v in out["pipelines"][name].items() if k != "rows"}), indent=1), flush=True)
    free_ollama(model)

    res = res_dir
    res.mkdir(parents=True, exist_ok=True)
    path = res / f"{a.suite}_{a.tag or time.strftime('%Y%m%d_%H%M')}.json"
    path.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print("saved", path)


if __name__ == "__main__":
    main()
