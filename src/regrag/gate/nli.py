from __future__ import annotations

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class NLI:
    """Entailment probability of a hypothesis given a premise (DeBERTa zero-shot NLI)."""

    def __init__(self, model: str = "MoritzLaurer/deberta-v3-base-zeroshot-v2.0", window_words: int = 200, stride: int = 100) -> None:
        self.tok = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForSequenceClassification.from_pretrained(model).eval()
        self.entail_id = {k.lower(): v for k, v in self.model.config.label2id.items()}["entailment"]
        self.window_words, self.stride = window_words, stride

    def _windows(self, premise: str) -> list[str]:
        words = premise.split()
        if len(words) <= self.window_words:
            return [premise]
        return [" ".join(words[i : i + self.window_words]) for i in range(0, len(words) - self.stride, self.stride)]

    @torch.no_grad()
    def entail(self, premise: str, hypothesis: str) -> float:
        """Best entailment probability over sliding windows of the premise (the model reads ~512 tokens)."""
        wins = self._windows(premise)
        enc = self.tok(wins, [hypothesis] * len(wins), truncation=True, max_length=512, padding=True, return_tensors="pt")
        probs = torch.softmax(self.model(**enc).logits, dim=-1)[:, self.entail_id]
        return float(probs.max())
