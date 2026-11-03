from __future__ import annotations

import re

import ollama


class OllamaLLM:
    def __init__(self, model: str = "qwen3:4b", num_ctx: int = 4096, max_tokens: int = 400) -> None:
        self.model, self.num_ctx, self.max_tokens = model, num_ctx, max_tokens
        self._client = ollama.Client()

    def generate(self, system: str, user: str) -> str:
        resp = self._client.chat(
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user + " /no_think"}],
            options={"temperature": 0, "num_ctx": self.num_ctx, "num_predict": self.max_tokens},
            think=False,  # qwen3: skip the reasoning trace, answer directly
        )
        text = resp["message"]["content"]
        return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
