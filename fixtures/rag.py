"""RAG fixtures."""

from __future__ import annotations

import pytest

from src.rag.chunker import chunk_documents
from src.rag.loader import load_knowledge_base
from src.rag.pipeline import RAGPipeline, get_pipeline
from src.rag.retriever import Retriever, VectorStore


@pytest.fixture(scope="session")
def rag_pipeline() -> RAGPipeline:
    """The shared pipeline - building the index once keeps the suite fast."""
    return get_pipeline()


@pytest.fixture(scope="session")
def knowledge_documents() -> list:
    return load_knowledge_base()


@pytest.fixture(scope="session")
def knowledge_chunks(knowledge_documents: list) -> list:
    return chunk_documents(knowledge_documents)


@pytest.fixture
def retriever(rag_pipeline: RAGPipeline) -> Retriever:
    return rag_pipeline.retriever


@pytest.fixture
def empty_vector_store() -> VectorStore:
    return VectorStore()
