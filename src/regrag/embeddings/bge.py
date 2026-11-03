from __future__ import annotations

from sentence_transformers import SentenceTransformer


class Embedder:
    def __init__(self, model: str = "BAAI/bge-m3", device: str = "cpu", max_seq_length: int = 512) -> None:
        self.model = SentenceTransformer(model, device=device)
        self.model.max_seq_length = max_seq_length

    def encode(self, texts: list[str], batch_size: int = 8) -> list[list[float]]:
        return self.model.encode(
            texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False
        ).tolist()
