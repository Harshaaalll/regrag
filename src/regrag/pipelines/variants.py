from __future__ import annotations

import os

from regrag.config import ROOT, load_config
from regrag.embeddings.bge import Embedder
from regrag.ingestion.loader import load_pdf
from regrag.chunking.heading_aware import chunk_pages
from regrag.llm.ollama_client import OllamaLLM
from regrag.pipelines.base import register
from regrag.pipelines.rag import RagPipeline
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


def build(name: str, skip: tuple[str, ...] = ("151MD",)) -> RagPipeline:
    """Wire a pipeline from config.yaml and the on-disk index (run scripts/build_index.py first)."""
    cfg = load_config()
    dense = DenseRetriever(Embedder(cfg["embeddings"]["model"], cfg["embeddings"]["device"]),
                           ChromaStore(ROOT / cfg["paths"]["index"]))
    if name == "classic":
        retriever = dense
    else:  # hybrid: BM25 over the same chunks that are in the index
        ch = cfg["chunking"]
        chunks = []
        for pdf in sorted((ROOT / cfg["paths"]["raw"]).glob("*.pdf")):
            if pdf.stem not in skip:
                chunks += chunk_pages(load_pdf(pdf), ch["max_tokens"], ch["overlap_tokens"])
        retriever = HybridRetriever([BM25Retriever(chunks), dense], cfg["retrieval"]["rrf_k"])
    # REGRAG_LLM_MODEL overrides the config model (e.g. qwen3:1.7b for quick dev runs on low memory)
    llm = OllamaLLM(os.environ.get("REGRAG_LLM_MODEL") or cfg["llm"]["generator"]["model"])
    return {"classic": ClassicRAG, "hybrid": HybridRAG}[name](
        retriever, llm, abstain_message=cfg["guardrails"]["abstain_message"])
