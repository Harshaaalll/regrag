from __future__ import annotations

import random
from collections import defaultdict


def split_items(items: list[dict], seed: int = 42, train_fraction: float = 0.5) -> tuple[list[dict], list[dict]]:
    """Seeded train/test split of golden items, stratified by question tag.

    Tune thresholds (reranker cutoff, NLI threshold) on train; report scores on test.
    Same seed and items always give the same split; every item lands in exactly one part.
    """
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1")
    by_tag: dict[str, list[dict]] = defaultdict(list)
    for it in items:
        by_tag[it["tag"]].append(it)
    train, test = [], []
    for tag, group in sorted(by_tag.items()):
        group = sorted(group, key=lambda i: i["id"])
        random.Random(f"{seed}:{tag}").shuffle(group)
        n = round(len(group) * train_fraction)
        train += group[:n]
        test += group[n:]
    order = {it["id"]: i for i, it in enumerate(items)}
    return sorted(train, key=lambda i: order[i["id"]]), sorted(test, key=lambda i: order[i["id"]])
