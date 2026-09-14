"""Retrieval-only tests.

Nothing here looks at a generated answer. If these fail, the problem is the
retriever - the generator never had a chance. That separation is the single
most useful thing a RAG test suite can give you.
"""

from __future__ import annotations

import pytest

from src.evaluation import judge
from src.rag.chunker import chunk_documents
from src.rag.embeddings import DeterministicEmbeddings, cosine_similarity
from src.utils.data_loader import case_ids, load_support_cases

pytestmark = [pytest.mark.rag]

# Deliberately vague openers ("I have a payment problem") are the first turn of a
# multi-turn case. There is no single right article to retrieve for them yet -
# that is the point of the follow-up - so they are scored in
# tests/agents/test_task_completion.py instead.
RETRIEVAL_CASES = [
    case
    for case in load_support_cases()
    if case.expected_source and case.reference_context and "multi-turn" not in case.tags
]


def test_knowledge_base_loads_every_article(knowledge_documents):
    sources = {d.metadata["source"] for d in knowledge_documents}
    assert sources == {
        "billing.md", "refunds.md", "payments.md",
        "subscriptions.md", "accounts.md", "technical_support.md",
    }


def test_chunking_preserves_headings_and_provenance(knowledge_chunks):
    assert len(knowledge_chunks) > len({c.metadata["source"] for c in knowledge_chunks})
    assert all(chunk.metadata.get("source") for chunk in knowledge_chunks)
    assert any(chunk.metadata.get("heading") for chunk in knowledge_chunks)
    assert all(chunk.page_content.strip() for chunk in knowledge_chunks)


def test_chunks_respect_the_configured_size(knowledge_documents):
    chunks = chunk_documents(knowledge_documents, chunk_size=400, chunk_overlap=80)
    # A little slack for the heading prefix that is prepended to each chunk.
    assert all(len(chunk.page_content) <= 400 + 120 for chunk in chunks)


def test_chunk_overlap_must_be_smaller_than_chunk_size(knowledge_documents):
    with pytest.raises(ValueError):
        chunk_documents(knowledge_documents, chunk_size=100, chunk_overlap=100)


def test_embeddings_are_deterministic():
    """Reproducibility is a testability property, not an implementation detail."""
    embeddings = DeterministicEmbeddings()
    first = embeddings.embed_query("duplicate charge on my invoice")
    second = embeddings.embed_query("duplicate charge on my invoice")
    assert first == second
    assert cosine_similarity(first, second) == pytest.approx(1.0, abs=1e-6)


def test_unrelated_texts_are_not_similar():
    embeddings = DeterministicEmbeddings()
    billing = embeddings.embed_query("I was charged twice for the same invoice")
    weather = embeddings.embed_query("the weather in Lisbon is warm in September")
    assert cosine_similarity(billing, weather) < 0.2


def test_vector_store_refuses_to_build_from_nothing(empty_vector_store):
    with pytest.raises(ValueError):
        empty_vector_store.add_documents([])


def test_searching_an_empty_store_is_an_error(empty_vector_store):
    with pytest.raises(RuntimeError):
        empty_vector_store.similarity_search("anything")


@pytest.mark.parametrize("case", RETRIEVAL_CASES, ids=case_ids(RETRIEVAL_CASES))
def test_the_right_article_is_retrieved(case, rag_pipeline):
    chunks = rag_pipeline.retrieve(case.user_query)
    assert chunks, f"{case.id}: nothing retrieved for '{case.user_query}'"

    sources = [chunk.source for chunk in chunks]
    assert case.expected_source in sources, (
        f"{case.id}: expected {case.expected_source}, retrieved {sources}"
    )


@pytest.mark.parametrize("case", RETRIEVAL_CASES, ids=case_ids(RETRIEVAL_CASES))
def test_retrieval_precision_and_recall(case, rag_pipeline, thresholds):
    chunks = rag_pipeline.retrieve(case.user_query)
    sources = [chunk.source for chunk in chunks]
    contexts = [chunk.content for chunk in chunks]

    precision = judge.contextual_precision(
        sources, case.relevant_sources, thresholds["contextual_precision"]
    )
    recall = judge.contextual_recall(
        case.reference_context, contexts, thresholds["contextual_recall"]
    )
    assert precision.passed, f"{case.id} precision: {precision.reason}"
    assert recall.passed, f"{case.id} recall: {recall.reason}"


def test_results_are_ranked_by_score(rag_pipeline):
    chunks = rag_pipeline.retrieve("I was charged twice this month.")
    scores = [chunk.score for chunk in chunks]
    assert scores == sorted(scores, reverse=True)
    assert all(chunk.rank == index + 1 for index, chunk in enumerate(chunks))


def test_top_k_is_respected(rag_pipeline):
    assert len(rag_pipeline.retrieve("refund policy", top_k=2)) <= 2


def test_the_similarity_floor_filters_noise(rag_pipeline):
    """Without a floor the retriever always returns something, and 'no relevant
    context' - the state that should trigger an honest 'I don't know' - can
    never occur."""
    from src.rag.retriever import Retriever

    strict = Retriever(rag_pipeline.store, top_k=4, min_similarity=0.99)
    assert strict.retrieve("what is the airspeed velocity of an unladen swallow") == []


def test_retrieval_is_stable_across_runs(rag_pipeline):
    first = [c.content for c in rag_pipeline.retrieve("How do I update my card?")]
    second = [c.content for c in rag_pipeline.retrieve("How do I update my card?")]
    assert first == second
