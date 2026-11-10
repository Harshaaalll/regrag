"""Add answer-correctness grades (CORRECT / PARTIAL / WRONG) to a golden result file, using a Gemini judge.

CLOUD STEP: sends each answered question, its reference answer and the system answer to Google's Gemini API.
No PDF text is sent. Needs GEMINI_API_KEY and --model (e.g. the model name shown in Google AI Studio).
  python scripts/judge_results.py evals/results/golden_golden_nogate.json --model <gemini-model> [--dry-run]
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from regrag.config import ROOT, load_config
from regrag.evals.benchmark import load_golden
from regrag.evals.judge import GeminiJudge, grade


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("result", type=Path)
    ap.add_argument("--model", required=True)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    data = json.loads(a.result.read_text(encoding="utf-8"))
    gold = {it["id"]: it for it in load_golden(ROOT / load_config()["paths"]["golden"])}
    todo = {name: [r for r in p["rows"] if r.get("scored") and not r["abstained"] and r["tag"] != "unanswerable"]
            for name, p in data["pipelines"].items()}
    n = sum(len(v) for v in todo.values())
    if a.dry_run:
        print(f"would send {n} answered questions (question + reference + answer text) to Gemini model {a.model}")
        return
    judge = GeminiJudge(a.model)
    for name, rows in todo.items():
        for r in rows:
            r["correctness"] = grade(judge, gold[r["id"]]["q"], gold[r["id"]]["a"], r["text"])
        pipe = data["pipelines"][name]
        graded = [r["correctness"] for r in rows]
        answerable = [r for r in pipe["rows"] if r.get("scored") and r["tag"] != "unanswerable"]
        pipe["correctness"] = {
            "judge": f"gemini:{a.model} (cloud)", "counts": dict(Counter(graded)), "answered_graded": len(graded),
            "correct_of_answered": (graded.count("CORRECT") / len(graded)) if graded else None,
            "correct_of_all_answerable": (graded.count("CORRECT") / len(answerable)) if answerable else None,
        }
        print(name, json.dumps(pipe["correctness"]))
    out = a.result.with_suffix(".judged.json")
    out.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
    print("saved", out)


if __name__ == "__main__":
    main()
