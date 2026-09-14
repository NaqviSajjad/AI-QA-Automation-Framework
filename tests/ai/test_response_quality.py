"""Semantic quality of the AI's answers.

Deterministic checks run first and unconditionally; the scored metrics run on
top. If a case fails, the failure message names the metric and the reason, so
the next question ("retrieval or generation?") has an answer immediately.
"""

from __future__ import annotations

import pytest

from src.evaluation.deterministic import evaluate_response_deterministically
from src.utils.data_loader import case_ids, load_support_cases

pytestmark = [pytest.mark.llm]

# Cases whose correct behaviour is a clarification or a hand-off. Answer
# relevancy is not a meaningful gate for those - a correct "I need more detail"
# deliberately does not restate the question - so they are covered by
# tests/agents/test_task_completion.py instead.
ANSWERING_CASES = [
    case
    for case in load_support_cases()
    if case.expected_source and not case.expects_uncertainty and "multi-turn" not in case.tags
]


@pytest.fixture(scope="module")
def answers(support_agent) -> dict:
    """Run every answering case once; the metrics below reuse the results."""
    return {case.id: support_agent.run(case.user_query) for case in ANSWERING_CASES}


@pytest.mark.parametrize("case", ANSWERING_CASES, ids=case_ids(ANSWERING_CASES))
def test_answer_is_relevant_and_grounded(case, answers, ai_evaluator):
    state = answers[case.id]

    evaluation = ai_evaluator.evaluate(
        question=case.user_query,
        answer=state.response,
        contexts=state.evidence,
        reference=case.expected_behavior,
        category=state.category,
        priority=state.priority,
        must_not=case.forbidden_phrases,
        expected_tool=case.expected_tool,
        actual_tool=state.selected_tool,
        case_id=case.id,
        metrics=["answer_relevancy", "faithfulness", "correctness", "hallucination"],
    )
    assert evaluation.passed, f"{case.id} ({ai_evaluator.backend_note()}): {evaluation.explain()}"


@pytest.mark.parametrize("case", ANSWERING_CASES, ids=case_ids(ANSWERING_CASES))
def test_answer_passes_the_deterministic_battery(case, answers):
    """Runs with no model and no thresholds - this one is allowed to block a merge."""
    state = answers[case.id]
    report = evaluate_response_deterministically(
        state.response,
        category=state.category,
        priority=state.priority,
        must_not=case.forbidden_phrases,
        expected_tool=case.expected_tool,
        actual_tool=state.selected_tool,
    )
    assert report.passed, f"{case.id}: {report.summary()}"


@pytest.mark.smoke
def test_a_grounded_answer_scores_above_an_ungrounded_one(support_agent, builtin_evaluator):
    """The metric has to discriminate, or the threshold means nothing.

    A framework that only ever shows passing scores has not been shown to
    detect anything. This compares the real answer with a plausible-sounding
    fabrication and asserts the evaluator can tell them apart.
    """
    question = "I was charged twice this month."
    state = support_agent.run(question)
    fabricated = (
        "Your duplicate charge of 47.50 EUR was refunded on 12 August under policy "
        "BILL-233, and the funds reached your Revolut account the same evening."
    )

    good = builtin_evaluator.evaluate(
        question, state.response, contexts=state.evidence, metrics=["faithfulness"]
    )
    bad = builtin_evaluator.evaluate(
        question, fabricated, contexts=state.evidence, metrics=["faithfulness"]
    )
    assert good.faithfulness > bad.faithfulness
    assert not bad.passed
