"""Vector store + retriever.

FAISS is used when the optional `ai` extra is installed; otherwise an exact
numpy cosine search over the same embeddings. The corpus is a few hundred
chunks, so exact search is instant and - importantly - gives *identical*
neighbours to FAISS, which keeps retrieval assertions stable across
environments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.rag.embeddings import EmbeddingBackend, get_embeddings
from src.utils.config import settings
from src.utils.logging import get_logger

logger = get_logger(__name__)

try:  # pragma: no cover - optional
    import faiss

    FAISS_AVAILABLE = True
except ImportError:  # pragma: no cover
    faiss = None  # type: ignore[assignment]
    FAISS_AVAILABLE = False


@dataclass
class RetrievedChunk:
    """A retrieval hit, with everything the evaluation layer needs to judge it."""

    content: str
    score: float
    rank: int
    source: str
    heading: str = ""
    metadata: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "content": self.content,
            "score": round(self.score, 4),
            "rank": self.rank,
            "source": self.source,
            "heading": self.heading,
        }


class VectorStore:
    """In-memory vector index with a FAISS fast path."""

    def __init__(self, embeddings: EmbeddingBackend | None = None, use_faiss: bool | None = None):
        self.embeddings = embeddings or get_embeddings()
        self.use_faiss = FAISS_AVAILABLE if use_faiss is None else (use_faiss and FAISS_AVAILABLE)
        self._documents: list[Any] = []
        self._matrix: np.ndarray | None = None
        self._idf: np.ndarray | None = None
        self._index: Any = None

    def __len__(self) -> int:
        return len(self._documents)

    @property
    def backend(self) -> str:
        return "faiss" if (self.use_faiss and self._index is not None) else "numpy"

    def add_documents(self, documents: list[Any]) -> None:
        if not documents:
            raise ValueError("Cannot build a vector store from zero documents")
        vectors = self.embeddings.embed_documents([d.page_content for d in documents])
        matrix = np.asarray(vectors, dtype="float32")

        # Inverse document frequency over the hashed feature space. Without it a
        # chunk that happens to repeat a common word ("charge" appears eleven
        # times in the duplicate-charge section) outranks the chunk that actually
        # answers the question. IDF is computed from this corpus only, so it stays
        # deterministic.
        document_frequency = (matrix > 0).sum(axis=0).astype("float32")
        self._idf = np.log((1.0 + len(documents)) / (1.0 + document_frequency)) + 1.0
        matrix = matrix * self._idf

        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        matrix = matrix / np.clip(norms, 1e-12, None)

        self._documents = list(documents)
        self._matrix = matrix
        if self.use_faiss:
            self._index = faiss.IndexFlatIP(matrix.shape[1])
            self._index.add(matrix)
        logger.info(
            "vector_store_built",
            extra={"chunks": len(documents), "dim": int(matrix.shape[1]), "backend": self.backend},
        )

    def similarity_search(self, query: str, k: int = 4) -> list[RetrievedChunk]:
        if self._matrix is None:
            raise RuntimeError("Vector store is empty - call add_documents() first")
        vector = np.asarray(self.embeddings.embed_query(query), dtype="float32")
        if self._idf is not None:
            vector = vector * self._idf
        norm = float(np.linalg.norm(vector)) or 1e-12
        vector = vector / norm

        k = min(k, len(self._documents))
        if self.use_faiss and self._index is not None:
            scores, indices = self._index.search(vector.reshape(1, -1), k)
            pairs = list(zip(indices[0].tolist(), scores[0].tolist(), strict=False))
        else:
            scores = self._matrix @ vector
            top = np.argsort(-scores)[:k]
            pairs = [(int(i), float(scores[i])) for i in top]

        hits: list[RetrievedChunk] = []
        for rank, (index, score) in enumerate(pairs, start=1):
            if index < 0:
                continue
            document = self._documents[index]
            hits.append(
                RetrievedChunk(
                    content=document.page_content,
                    score=float(score),
                    rank=rank,
                    source=document.metadata.get("source", "unknown"),
                    heading=document.metadata.get("heading", ""),
                    metadata=document.metadata,
                )
            )
        return hits


class Retriever:
    """Applies top-k and the similarity floor.

    The floor matters: without it the store always returns *something*, the
    generator always has 'context', and hallucination tests can never fail
    for the right reason.
    """

    def __init__(
        self,
        store: VectorStore,
        top_k: int | None = None,
        min_similarity: float | None = None,
        relative_floor: float | None = None,
    ):
        self.store = store
        self.top_k = top_k if top_k is not None else settings.rag.top_k
        self.min_similarity = (
            min_similarity if min_similarity is not None else settings.rag.min_similarity
        )
        self.relative_floor = (
            relative_floor if relative_floor is not None else settings.rag.relative_floor
        )

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        hits = self.store.similarity_search(query, k=top_k or self.top_k)
        # Absolute floor plus a floor relative to the best hit. Fixed top-k always
        # returns k chunks whether or not k chunks are relevant, which pads the
        # context with near-misses and quietly drags contextual relevancy down.
        cutoff = self.min_similarity
        if hits and self.relative_floor:
            cutoff = max(cutoff, hits[0].score * self.relative_floor)
        kept = [hit for hit in hits if hit.score >= cutoff]
        logger.info(
            "retrieval",
            extra={
                "query": query[:120],
                "returned": len(hits),
                "kept": len(kept),
                "top_score": round(hits[0].score, 4) if hits else 0.0,
            },
        )
        return kept
