from __future__ import annotations

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    id: str
    doc_id: str
    text: str
    page: int | None = None
    section: str | None = None
    modality: str = "text"  # text | table | figure


class Citation(BaseModel):
    chunk_id: str
    doc_id: str
    page: int | None = None
    quote: str = ""


class GateDecision(BaseModel):
    supported: bool
    score: float
    method: str


class Answer(BaseModel):
    question: str
    text: str
    abstained: bool = False
    citations: list[Citation] = Field(default_factory=list)
    contexts: list[str] = Field(default_factory=list)  # what RAGAS/DeepEval consume
    gate: GateDecision | None = None
    pipeline: str = ""
    latency_s: float = 0.0
