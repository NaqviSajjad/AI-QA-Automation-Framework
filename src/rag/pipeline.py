"""The RAG pipeline: load -> chunk -> embed -> store -> retrieve -> generate.

`RAGResult` deliberately carries the retrieved chunks alongside the answer.
That is what makes a RAG failure *diagnosable*: the same object tells you
whether the retriever missed the article or the generator ignored it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from src.llm.base import LLMProvider
from src.rag.chunker import chunk_documents
from src.rag.loader import load_knowledge_base
from src.rag.retriever import RetrievedChunk, Retriever, VectorStore
from src.utils.config import settings
from src.utils.logging import get_logger

logger = get_logger(__name__)

RAG_PROMPT = """You are a customer-support assistant.

Ground every factual claim in the context below. If the context does not contain the
answer, say clearly that you do not have the information and offer to escalate. Never
invent policies, prices, product names or transaction details. Acknowledge the issue,
give the relevant policy, and finish with a concrete next step.

USER QUERY:
{query}

CONTEXT:
{context}
"""


@dataclass
class RAGResult:
    """Answer + full retrieval evidence."""

    query: str
    answer: str
    chunks: list[RetrievedChunk] = field(default_factory=list)
    latency_ms: float = 0.0
    llm_calls: int = 0

    @property
    def contexts(self) -> list[str]:
        return [chunk.content for chunk in self.chunks]

    @property
    def context_text(self) -> str:
        return "\n\n".join(self.contexts)

    @property
    def sources(self) -> list[str]:
        return sorted({chunk.source for chunk in self.chunks})

    @property
    def has_context(self) -> bool:
        return bool(self.chunks)


class RAGPipeline:
    def __init__(
        self,
        llm: LLMProvider | None = None,
        knowledge_base_dir: str | Path | None = None,
        top_k: int | None = None,
        min_similarity: float | None = None,
    ) -> None:
        from src.llm.factory import get_llm_provider

        self.llm = llm or get_llm_provider()
        documents = load_knowledge_base(knowledge_base_dir)
        self.documents = documents
        self.chunks = chunk_documents(documents)
        self.store = VectorStore()
        self.store.add_documents(self.chunks)
        self.retriever = Retriever(self.store, top_k=top_k, min_similarity=min_similarity)

    # -- retrieval only (what tests/rag/test_retrieval.py asserts on) ------ #
    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        return self.retriever.retrieve(query, top_k=top_k)

    # -- retrieval + generation ------------------------------------------- #
    def run(self, query: str, top_k: int | None = None) -> RAGResult:
        chunks = self.retrieve(query, top_k=top_k)
        context = "\n\n".join(f"[{c.source}] {c.content}" for c in chunks)
        prompt = RAG_PROMPT.format(query=query, context=context or "(no relevant context found)")
        response = self.llm.complete(prompt)
        return RAGResult(
            query=query,
            answer=response.text,
            chunks=chunks,
            latency_ms=response.latency_ms,
            llm_calls=1,
        )

    def stats(self) -> dict[str, Any]:
        return {
            "documents": len(self.documents),
            "chunks": len(self.chunks),
            "backend": self.store.backend,
            "top_k": self.retriever.top_k,
            "min_similarity": self.retriever.min_similarity,
        }


@lru_cache(maxsize=1)
def get_pipeline() -> RAGPipeline:
    """Process-wide singleton - building the index once keeps the suite fast."""
    pipeline = RAGPipeline()
    logger.info("rag_pipeline_ready", extra=pipeline.stats())
    return pipeline


def reset_pipeline() -> None:
    get_pipeline.cache_clear()


__all__ = [
    "RAGPipeline",
    "RAGResult",
    "get_pipeline",
    "reset_pipeline",
    "settings",
]
