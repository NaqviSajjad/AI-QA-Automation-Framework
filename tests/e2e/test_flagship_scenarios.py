"""The three showcase scenarios.

Each one runs the whole stack - browser, HTTP API, agent, RAG, tools, model -
captures the answer *from the UI*, and only then hands it to the evaluation
layer. That order is the whole point of the framework:

    Playwright proves the journey worked.
    The evaluation layer proves the answer was any good.
    Both have to pass.
"""

from __future__ import annotations

import pytest

from src.agents.nodes import expresses_uncertainty
from src.evaluation.collector import record_event
from src.evaluation.deterministic import evaluate_response_deterministically, find_forbidden_claims
from src.utils.data_loader import case_by_id

pytestmark = [pytest.mark.flagship, pytest.mark.ui, pytest.mark.llm]


# =========================================================================== #
# Flagship 1 - billing complaint, end to end
# =========================================================================== #
@pytest.mark.smoke
def test_billing_complaint_end_to_end(chatbot_page, conversation_api, ai_evaluator, tracer):
    """Customer signs in, reports a duplicate charge, and gets a resolution that
    is relevant, grounded, hallucination-free and correctly routed."""
    case = case_by_id("CS-001")

    # --- 1. the user journey (Playwright) --------------------------------- #
    chatbot_page.send_message(case.user_query)
    answer = chatbot_page.get_latest_response()

    assert answer, "no answer was rendered in the UI"
    assert not chatbot_page.has_error()
    assert not chatbot_page.is_loading()
    assert chatbot_page.get_category() == "billing"
    conversation_id = chatbot_page.get_conversation_id()
    assert conversation_id.startswith("conv_")

    # --- 2. the API and the agent's own record ---------------------------- #
    stored = conversation_api.get_conversation(conversation_id)
    assert stored.status_code == 200
    assert stored.json()["messages"][-1]["content"] == answer

    span = tracer.last()
    assert span is not None and span.metrics["tool_calls"] == 1

    # --- 3. deterministic validation -------------------------------------- #
    report = evaluate_response_deterministically(
        answer, category="billing", must_not=case.forbidden_phrases
    )
    assert report.passed, report.summary()

    # --- 4. AI quality evaluation ----------------------------------------- #
    from src.agents.graph import get_agent

    state = get_agent().run(case.user_query)  # same input, for retrieval evidence
    evaluation = ai_evaluator.evaluate(
        question=case.user_query,
        answer=answer,
        contexts=state.evidence,
        reference=case.expected_behavior,
        expected_source=case.expected_source,
        retrieved_sources=state.retrieved_sources,
        reference_context=case.reference_context,
        expected_tool=case.expected_tool,
        actual_tool=state.selected_tool,
        category="billing",
        must_not=case.forbidden_phrases,
        case_id=case.id,
        metrics=[
            "answer_relevancy", "faithfulness", "correctness", "hallucination",
            "contextual_precision", "tool_correctness", "task_completion",
        ],
    )
    record_event("flagship", scenario="billing_complaint", passed=evaluation.passed)

    assert evaluation.answer_relevancy >= 0.85, evaluation.explain()
    assert evaluation.faithfulness >= 0.85, evaluation.explain()
    assert evaluation.hallucination <= 0.20, evaluation.explain()
    assert evaluation.tool_correctness == 1.0
    assert evaluation.passed, evaluation.explain()


# =========================================================================== #
# Flagship 2 - multi-turn conversation
# =========================================================================== #
def test_multi_turn_payment_conversation(chatbot_page, conversation_api, ai_evaluator):
    """The second turn is meaningless without the first. Context retention,
    classification, tool choice and answer quality are all asserted on the turn
    that actually carries the resolution."""
    case = case_by_id("CS-034")

    chatbot_page.send_message(case.user_query)
    first = chatbot_page.get_latest_response()
    assert first

    chatbot_page.send_message(case.follow_ups[0])
    second = chatbot_page.get_latest_response()
    conversation_id = chatbot_page.get_conversation_id()

    # --- conversation integrity ------------------------------------------- #
    assert second and second != first
    assert chatbot_page.history.ai_count == 2
    stored = conversation_api.get_conversation(conversation_id).json()
    assert stored["message_count"] == 4
    assert [m["role"] for m in stored["messages"]] == ["user", "assistant", "user", "assistant"]

    # --- the follow-up was understood in context -------------------------- #
    assert any(word in second.lower() for word in ("charge", "billing", "payment", "duplicate"))

    # --- quality of the resolving turn ------------------------------------ #
    from src.agents.graph import get_agent

    history = [{"role": m["role"], "content": m["content"]} for m in stored["messages"][:2]]
    state = get_agent().run(case.follow_ups[0], history=history)

    evaluation = ai_evaluator.evaluate(
        question=case.follow_ups[0],
        answer=second,
        contexts=state.evidence,
        reference=case.follow_up_expected_behavior or case.expected_behavior,
        category=state.category,
        must_not=case.forbidden_phrases,
        escalated=state.requires_escalation,
        expected_escalation=case.expected_escalation,
        case_id=f"{case.id}-turn2",
        metrics=["faithfulness", "hallucination", "task_completion"],
    )
    record_event("flagship", scenario="multi_turn", passed=evaluation.passed)
    assert evaluation.passed, evaluation.explain()


# =========================================================================== #
# Flagship 3 - hallucination prevention
# =========================================================================== #
def test_hallucination_prevention_end_to_end(chatbot_page, ai_evaluator):
    """Asked about something that does not exist, the assistant must say so.

    Note which metrics are asserted here and which are not: answer relevancy is
    deliberately absent, because a correct refusal does not restate the question.
    Applying every metric everywhere would fail the right behaviour.
    """
    case = case_by_id("CS-032")

    chatbot_page.send_message(case.user_query)
    answer = chatbot_page.get_latest_response()

    assert answer
    assert expresses_uncertainty(answer), f"invented an answer: {answer}"
    assert not find_forbidden_claims(answer)
    assert "specialist" in answer.lower() or "human" in answer.lower()

    report = evaluate_response_deterministically(
        answer, category="refund", must_not=case.forbidden_phrases
    )
    assert report.passed, report.summary()

    from src.agents.graph import get_agent

    state = get_agent().run(case.user_query)
    evaluation = ai_evaluator.evaluate(
        question=case.user_query,
        answer=answer,
        contexts=state.evidence,
        reference=case.expected_behavior,
        category=state.category,
        must_not=case.forbidden_phrases,
        case_id=case.id,
        metrics=["faithfulness", "hallucination", "correctness"],
    )
    record_event("flagship", scenario="hallucination_prevention", passed=evaluation.passed)
    assert evaluation.passed, evaluation.explain()


# =========================================================================== #
# Flagship 4 - the quality gate itself
# =========================================================================== #
@pytest.mark.slow
def test_the_quality_gate_evaluates_the_whole_dataset():
    """The gate CI runs, exercised as a test."""
    from scripts.quality_gate import evaluate_gate

    report = evaluate_gate()
    assert report.checks, "quality gate produced no checks"
    assert report.passed, "\n" + report.render()
