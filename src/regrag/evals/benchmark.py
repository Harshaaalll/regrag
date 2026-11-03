from __future__ import annotations

from collections import defaultdict
from statistics import mean

from regrag.types import Answer


def score_row(item: dict, ans: Answer, doc_id: str) -> dict:
    """Deterministic per-question facts. Answer correctness needs a judge (RAGAS / LLM) and is not computed here."""
    unanswerable = item["tag"] == "unanswerable"
    gold = set(item["pages"])
    cited_pages = {c.page for c in ans.citations if c.doc_id == doc_id}
    return {
        "id": item["id"], "tag": item["tag"], "pipeline": ans.pipeline, "abstained": ans.abstained,
        "abstain_correct": ans.abstained == unanswerable,
        "cited_gold_page": (bool(cited_pages & gold) if not unanswerable and not ans.abstained else None),
        "n_citations": len(ans.citations), "latency_s": round(ans.latency_s, 1), "text": ans.text,
        "gate_score": ans.gate.score if ans.gate else None,
    }


def run_benchmark(pipeline, items: list[dict], doc_id: str) -> list[dict]:
    return [score_row(it, pipeline.answer(it["q"]), doc_id) for it in items]


def summarize(rows: list[dict]) -> dict:
    """Abstain accuracy over all questions; citation-hit rate over answered answerable questions only."""
    answerable = [r for r in rows if r["tag"] != "unanswerable"]
    answered = [r for r in answerable if not r["abstained"]]
    by_tag = defaultdict(list)
    for r in rows:
        by_tag[r["tag"]].append(r["abstain_correct"])
    return {
        "n": len(rows),
        "abstain_accuracy": mean(r["abstain_correct"] for r in rows),
        "answer_rate_on_answerable": len(answered) / len(answerable) if answerable else None,
        "cited_gold_page_rate_when_answered": mean(r["cited_gold_page"] for r in answered) if answered else None,
        "mean_latency_s": mean(r["latency_s"] for r in rows),
        "abstain_accuracy_by_tag": {t: mean(v) for t, v in by_tag.items()},
    }
