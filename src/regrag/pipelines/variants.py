from __future__ import annotations

import os

from regrag.config import ROOT, load_config
from regrag.embeddings.bge import Embedder
from regrag.ingestion.loader import load_pdf
from regrag.chunking.heading_aware import chunk_pages
from regrag.llm.ollama_client import OllamaLLM
from regrag.pipelines.base import register
from regrag.pipelines.rag import RagPipeline
from regrag.retrieval.agentic import AgenticRetriever
from regrag.retrieval.bm25 import BM25Retriever
from regrag.retrieval.dense import DenseRetriever
from regrag.retrieval.hybrid import HybridRetriever
from regrag.vectordb.chroma_store import ChromaStore


@register
class ClassicRAG(RagPipeline):
    name = "classic"


@register
class HybridRAG(RagPipeline):
    name = "hybrid"


@register
class AgenticRAG(RagPipeline):
    name = "agentic"  # retriever is an AgenticRetriever wrapping the hybrid retriever


def make_retriever(name: str, skip: tuple[str, ...] = ("151MD",)):
    """classic = dense only; hybrid = BM25 + dense (RRF). Needs the on-disk index (scripts/build_index.py)."""
    cfg = load_config()
    dense = DenseRetriever(Embedder(cfg["embeddings"]["model"], cfg["embeddings"]["device"]),
                           ChromaStore(ROOT / cfg["paths"]["index"]))
    if name == "classic":
        return dense
    ch = cfg["chunking"]
    chunks = []
    for pdf in sorted((ROOT / cfg["paths"]["raw"]).glob("*.pdf")):
        if pdf.stem not in skip:
            chunks += chunk_pages(load_pdf(pdf), ch["max_tokens"], ch["overlap_tokens"])
    return HybridRetriever([BM25Retriever(chunks), dense], cfg["retrieval"]["rrf_k"])


def build(name: str, skip: tuple[str, ...] = ("151MD",), gate=None) -> RagPipeline:
    cfg = load_config()
    g = cfg["guardrails"]
    # REGRAG_LLM_MODEL overrides the config model (e.g. qwen3:1.7b for quick dev runs on low memory)
    llm = OllamaLLM(os.environ.get("REGRAG_LLM_MODEL") or cfg["llm"]["generator"]["model"])
    base = make_retriever("hybrid" if name == "agentic" else name, skip)
    retriever = AgenticRetriever(base, llm) if name == "agentic" else base
    return {"classic": ClassicRAG, "hybrid": HybridRAG, "agentic": AgenticRAG}[name](
        retriever, llm, abstain_message=g["abstain_message"], gate=gate,
        redact_pii=g["redact_pii"], block_off_topic=g["block_off_topic"], no_advice=g["no_advice"])
