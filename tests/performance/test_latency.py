"""AI-specific performance: latency, call budgets, token cost.

Not a load-testing suite. These are the numbers that decide whether an AI
feature is affordable and usable: how long a customer waits, how many model and
tool calls one answer costs, and what that comes to per thousand conversations.
"""

from __future__ import annotations

import time

import pytest

from src.evaluation.collector import record_event
from src.utils.config import settings
from src.utils.data_loader import load_support_cases

pytestmark = [pytest.mark.performance]

SAMPLE_QUERIES = [case.user_query for case in load_support_cases()[:12]]


def _limits() -> dict:
    return settings.evaluation.get("performance", {})


@pytest.fixture(scope="module")
def latency_samples(support_agent) -> list[float]:
    samples = []
    for query in SAMPLE_QUERIES:
        started = time.perf_counter()
        support_agent.run(query)
        samples.append((time.perf_counter() - started) * 1000)
    return sorted(samples)


def test_single_answer_is_within_the_latency_budget(support_agent):
    started = time.perf_counter()
    support_agent.run("I was charged twice this month.")
    elapsed_ms = (time.perf_counter() - started) * 1000

    record_event("latency", latency_ms=elapsed_ms, scenario="billing_complaint")
    assert elapsed_ms < _limits().get("max_latency_ms", 8000), f"{elapsed_ms:.0f}ms"


def test_p95_latency_is_within_budget(latency_samples):
    index = max(0, int(len(latency_samples) * 0.95) - 1)
    p95 = latency_samples[index]
    assert p95 < _limits().get("p95_latency_ms", 7000), f"p95 {p95:.0f}ms"


def test_no_single_answer_is_an_outlier(latency_samples):
    assert max(latency_samples) < _limits().get("max_latency_ms", 8000)


def test_llm_and_tool_call_budgets_hold_across_the_dataset(support_agent):
    """Cost control is a correctness property for agents: an agent that loops is
    both slow and expensive, and both show up here before production does."""
    limits = _limits()
    for case in load_support_cases():
        state = support_agent.run(case.user_query)
        assert state.llm_calls <= limits.get("max_llm_calls", 5), case.id
        assert len(state.tool_calls) <= limits.get("max_tool_calls", 3), case.id


def test_token_usage_per_answer_is_bounded(mock_llm):
    prompt = "USER QUERY:\nI was charged twice this month.\n\nCONTEXT:\n" + ("policy text. " * 50)
    response = mock_llm.complete(prompt)

    record_event("llm", total_tokens=response.total_tokens, model=response.model)
    assert response.total_tokens <= _limits().get("max_tokens_per_request", 2000)


def test_estimated_cost_per_thousand_conversations_is_reported(mock_llm):
    """Publishes the number rather than asserting a threshold: with the mock
    provider the cost is zero, and a green assertion on a fake number would be
    worse than no assertion."""
    limits = _limits()
    prompt = "USER QUERY:\nI was charged twice this month."
    response = mock_llm.complete(prompt)

    cost = (
        response.prompt_tokens / 1000 * limits.get("cost_per_1k_input_tokens_usd", 0.0)
        + response.completion_tokens / 1000 * limits.get("cost_per_1k_output_tokens_usd", 0.0)
    )
    record_event("cost", usd_per_conversation=cost, usd_per_1k=cost * 1000)
    assert cost >= 0


def test_retrieval_alone_is_fast(rag_pipeline):
    """Retrieval is the part of a RAG answer you control. If it is slow, the
    model is not the problem."""
    started = time.perf_counter()
    for query in SAMPLE_QUERIES:
        rag_pipeline.retrieve(query)
    average_ms = (time.perf_counter() - started) * 1000 / len(SAMPLE_QUERIES)

    record_event("latency", latency_ms=average_ms, scenario="retrieval_only")
    assert average_ms < 250, f"retrieval averaged {average_ms:.1f}ms"
