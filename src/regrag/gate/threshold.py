from __future__ import annotations


class RetrievalGate:
    """Abstain before generating when even the best retrieved chunk scores below min_score (reranker score)."""

    def __init__(self, min_score: float) -> None:
        self.min_score = min_score

    def allows(self, hits: list[tuple[object, float]]) -> bool:
        return bool(hits) and max(s for _, s in hits) >= self.min_score


def best_threshold(answerable: list[float], unanswerable: list[float]) -> tuple[float, float]:
    """Threshold on the top score that best separates answerable (should pass) from unanswerable (should be blocked).

    Returns (threshold, balanced accuracy). Candidates are the observed scores; ties go to the lower threshold.
    """
    if not answerable or not unanswerable:
        raise ValueError("need at least one answerable and one unanswerable score")
    best = (min(answerable + unanswerable), -1.0)
    for t in sorted(set(answerable + unanswerable)):
        tpr = sum(s >= t for s in answerable) / len(answerable)
        tnr = sum(s < t for s in unanswerable) / len(unanswerable)
        if (tpr + tnr) / 2 > best[1]:
            best = (t, (tpr + tnr) / 2)
    return best
