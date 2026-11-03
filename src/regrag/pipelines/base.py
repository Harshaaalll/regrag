from __future__ import annotations

from abc import ABC, abstractmethod

from regrag.types import Answer, Chunk


class Pipeline(ABC):
    """Every RAG variant implements this so the benchmark can swap them freely."""

    name: str

    @abstractmethod
    def index(self, chunks: list[Chunk]) -> None: ...

    @abstractmethod
    def answer(self, question: str) -> Answer: ...


REGISTRY: dict[str, type[Pipeline]] = {}


def register(cls: type[Pipeline]) -> type[Pipeline]:
    REGISTRY[cls.name] = cls
    return cls
