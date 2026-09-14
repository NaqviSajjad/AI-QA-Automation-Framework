"""Embeddings.

Two backends:

* `deterministic` (default) - a hashed bag-of-words projection. No network, no
  credentials, byte-identical vectors on every machine. That reproducibility is
  what lets retrieval metrics act as a *gate* in CI rather than a suggestion.
* `openai`      - real semantic embeddings for nightly / local runs.

Both expose the LangChain `Embeddings` shape (`embed_documents`, `embed_query`)
so they can be handed straight to a LangChain vector store.
"""

from __future__ import annotations

import hashlib
import math
from typing import Protocol

from src.utils.config import settings
from src.utils.text import stem, tokenize  # noqa: F401  (re-exported for convenience)


class EmbeddingBackend(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


def _tokenize(text: str) -> list[str]:
    return tokenize(text)


def _normalise(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vector))
    return [v / norm for v in vector] if norm else vector


class DeterministicEmbeddings:
    """Hashed bag-of-words + character-bigram projection, L2-normalised.

    Not a semantic model: it captures lexical overlap. That is honest and it is
    enough for the retrieval assertions in this framework, which check that the
    *right article* comes back - and it never changes between runs.
    """

    def __init__(self, dim: int | None = None) -> None:
        self.dim = dim or settings.rag.embedding_dim

    def _bucket(self, token: str) -> int:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, "big") % self.dim

    def embed_query(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        tokens = _tokenize(text)
        for token in tokens:
            vector[self._bucket(token)] += 1.0
        for first, second in zip(tokens, tokens[1:], strict=False):
            vector[self._bucket(f"{first}_{second}")] += 0.5
        return _normalise(vector)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]


class OpenAIEmbeddings:
    """Thin wrapper so the pipeline does not import `openai` unless it is used."""

    def __init__(self, model: str | None = None) -> None:
        from openai import OpenAI

        if not settings.llm.api_key:
            raise RuntimeError("OPENAI_API_KEY required for embedding_backend=openai")
        self.model = model or settings.rag.embedding_model
        self._client = OpenAI(api_key=settings.llm.api_key, base_url=settings.llm.base_url or None)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        response = self._client.embeddings.create(model=self.model, input=texts)
        return [item.embedding for item in response.data]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def get_embeddings(backend: str | None = None) -> EmbeddingBackend:
    name = (backend or settings.rag.embedding_backend).lower()
    if name == "openai":
        return OpenAIEmbeddings()
    if name == "deterministic":
        return DeterministicEmbeddings()
    raise ValueError(f"Unknown embedding backend '{name}' (deterministic|openai)")


def cosine_similarity(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right, strict=False))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if not left_norm or not right_norm:
        return 0.0
    return numerator / (left_norm * right_norm)
