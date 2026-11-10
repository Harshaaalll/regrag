from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

import yaml

from regrag.types import Answer


def load_golden(folder: Path) -> list[dict]:
    """All seed_*.yaml files in the folder. Each item gets a `doc` (its own, else the file's default)."""
    items = []
    for path in sorted(folder.glob("seed_*.yaml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        items += [{"doc": data["doc"], **it} for it in data["items"]]
    return items


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


def _read_checkpoint(path: Path | None) -> dict[str, dict]:
    if not path or not path.exists():
        return {}
    rows = (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    return {r["id"]: r for r in rows if r.get("scored")}  # failed rows are retried on resume


def run_benchmark(pipeline, items: list[dict], doc_id: str = "", checkpoint: Path | None = None) -> list[dict]:
    """Answer every item. A crash on one question makes that row unscored (not wrong) and the run continues.

    With a checkpoint file, finished rows are appended as they complete and skipped when the run is repeated.
    """
    done, rows = _read_checkpoint(checkpoint), []
    for it in items:
        if it["id"] in done:
            rows.append(done[it["id"]])
            continue
        try:
            row = {**score_row(it, pipeline.answer(it["q"]), it.get("doc", doc_id)), "scored": True}
        except Exception as exc:  # noqa: BLE001 - any failure must not stop the benchmark
            row = {"id": it["id"], "tag": it["tag"], "pipeline": getattr(pipeline, "name", ""), "scored": False,
                   "error": f"{type(exc).__name__}: {exc}"[:300]}
        rows.append(row)
        if checkpoint:
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            with checkpoint.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return rows


def summarize(rows: list[dict]) -> dict:
    """Abstain accuracy over all questions; citation-hit rate over answered answerable questions only."""
    n_unscored = sum(not r.get("scored", True) for r in rows)
    rows = [r for r in rows if r.get("scored", True)]  # unscored questions are reported, never counted as wrong
    if not rows:
        return {"n": 0, "n_unscored": n_unscored}
    answerable = [r for r in rows if r["tag"] != "unanswerable"]
    answered = [r for r in answerable if not r["abstained"]]
    by_tag = defaultdict(list)
    for r in rows:
        by_tag[r["tag"]].append(r["abstain_correct"])
    return {
        "n": len(rows),
        "n_unscored": n_unscored,
        "abstain_accuracy": mean(r["abstain_correct"] for r in rows),
        "answer_rate_on_answerable": len(answered) / len(answerable) if answerable else None,
        "cited_gold_page_rate_when_answered": mean(r["cited_gold_page"] for r in answered) if answered else None,
        "mean_latency_s": mean(r["latency_s"] for r in rows),
        "abstain_accuracy_by_tag": {t: mean(v) for t, v in by_tag.items()},
    }
