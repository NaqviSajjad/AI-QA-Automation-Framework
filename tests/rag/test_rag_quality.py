"""End-to-end RAG quality, with retrieval and generation scored separately.

When one of these fails, `diagnose_rag_failure` says which half broke. "The
answer was wrong" is not an actionable bug report; "the retriever returned
subscriptions.md for a refund question" is.
"""

from __future__ import annotations

import pytest

from src.evaluation import judge
from src.utils.data_loader import case_ids, load_support_cases

pytestmark = [pytest.mark.rag]

RAG_CASES = [c for c in load_support_cases() if c.expected_source and not c.expects_uncertainty]


def diagnose_rag_failure(retrieval_ok: bool, generation_ok: bool) -> str:
    if not retrieval_ok and not generation_ok:
        return "RETRIEVAL (generation cannot be judged on the wrong context)"
    if not retrieval_ok:
        return "RETRIEVAL"
    if not generation_ok:
        return "GENERATION"
    return "OK"


@pytest.mark.parametrize("case", RAG_CASES, ids=case_ids(RAG_CASES))
def test_rag_answer_is_grounded_in_what_was_retrieved(case, rag_pipeline, thresholds):
    result = rag_pipeline.run(case.user_query)

    retrieval = judge.contextual_relevancy(
        case.user_query, result.contexts, thresholds["contextual_relevancy"]
    )
    generation = judge.faithfulness(result.answer, result.contexts, thresholds["faithfulness"])

    verdict = diagnose_rag_failure(retrieval.passed, generation.passed)
    assert verdict == "OK", (
        f"{case.id} failure origin: {verdict}\n"
        f"  retrieval: {retrieval.score:.3f} ({retrieval.reason})\n"
        f"  generation: {generation.score:.3f} ({generation.reason})\n"
        f"  sources: {result.sources}"
    )


def test_pipeline_reports_what_it_built(rag_pipeline):
    stats = rag_pipeline.stats()
    assert stats["documents"] == 6
    assert stats["chunks"] > 20
    assert stats["backend"] in {"faiss", "numpy"}


def test_answer_cites_the_sources_it_used(rag_pipeline):
    result = rag_pipeline.run("How long does a refund take on a credit card?")
    assert "refunds.md" in result.sources
    assert result.has_context


def test_no_context_leads_to_an_honest_answer(rag_pipeline):
    """The interesting case: retrieval legitimately finds nothing."""
    result = rag_pipeline.run("What is the warranty on the Acme Cloud hardware appliance?")
    faith = judge.faithfulness(result.answer, result.contexts)
    assert faith.passed or "do not have" in result.answer.lower()


def test_generation_is_scored_against_retrieved_context_not_the_whole_corpus(rag_pipeline):
    """A claim that is true elsewhere in the corpus but absent from the retrieved
    context is still unfaithful for this answer - that is what faithfulness means."""
    result = rag_pipeline.run("How long does a refund take on a credit card?")
    off_context_claim = (
        "Uploads are limited to 250 MB per file and fail with a 413 error above that limit."
    )
    score = judge.faithfulness(off_context_claim, result.contexts)
    assert not score.passed


@pytest.mark.slow
def test_full_dataset_average_meets_the_gate(rag_pipeline, thresholds):
    """Per-case gates catch regressions; the corpus average is what the quality
    gate publishes."""
    faith, relevancy = [], []
    for case in RAG_CASES:
        result = rag_pipeline.run(case.user_query)
        faith.append(judge.faithfulness(result.answer, result.contexts).score)
        relevancy.append(judge.contextual_relevancy(case.user_query, result.contexts).score)

    mean_faith = sum(faith) / len(faith)
    mean_relevancy = sum(relevancy) / len(relevancy)
    assert mean_faith >= thresholds["faithfulness"], f"mean faithfulness {mean_faith:.3f}"
    assert mean_relevancy >= thresholds["contextual_relevancy"], (
        f"mean contextual relevancy {mean_relevancy:.3f}"
    )
