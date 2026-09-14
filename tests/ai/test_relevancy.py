"""Answer-relevancy tests.

Relevancy asks a narrow question: did the answer address *this* customer's
issue, stay on topic, and end with something they can act on?
"""

from __future__ import annotations

import pytest

from src.utils.data_loader import case_by_id

pytestmark = [pytest.mark.llm]

ON_TOPIC_CASES = ["CS-001", "CS-006", "CS-010", "CS-015", "CS-020", "CS-025"]

OFF_TOPIC_ANSWER = (
    "Acme Cloud was founded in 2015 and now serves customers in 40 countries. "
    "Our offices are in Berlin and Lisbon, and we run a quarterly customer webinar "
    "that you are welcome to join."
)


@pytest.mark.parametrize("case_id", ON_TOPIC_CASES)
def test_answers_are_relevant_to_the_question(case_id, support_agent, ai_evaluator):
    case = case_by_id(case_id)
    state = support_agent.run(case.user_query)

    evaluation = ai_evaluator.evaluate(
        question=case.user_query,
        answer=state.response,
        contexts=state.evidence,
        category=state.category,
        case_id=case.id,
        metrics=["answer_relevancy"],
    )
    assert evaluation.passed, f"{case_id}: {evaluation.explain()}"


@pytest.mark.parametrize("case_id", ON_TOPIC_CASES)
def test_answers_end_with_an_actionable_next_step(case_id, support_agent):
    """A support answer that states policy and stops is not a resolution."""
    case = case_by_id(case_id)
    answer = support_agent.run(case.user_query).response.lower()
    assert any(
        marker in answer
        for marker in ("next step", "could you", "please confirm", "would you like", "i can ")
    ), f"{case_id} gave no next step: {answer}"


def test_an_off_topic_answer_is_scored_down(builtin_evaluator):
    """The metric must be able to fail, or the threshold is theatre."""
    evaluation = builtin_evaluator.evaluate(
        "I was charged twice this month.",
        OFF_TOPIC_ANSWER,
        contexts=[],
        metrics=["answer_relevancy"],
    )
    assert not evaluation.passed
    assert evaluation.answer_relevancy < 0.6


def test_relevancy_separates_a_good_answer_from_a_padded_one(support_agent, builtin_evaluator):
    question = "How long does a refund take to arrive on a credit card?"
    good = support_agent.run(question).response
    padded = f"{OFF_TOPIC_ANSWER} {OFF_TOPIC_ANSWER}"

    good_score = builtin_evaluator.evaluate(
        question, good, metrics=["answer_relevancy"]
    ).answer_relevancy
    padded_score = builtin_evaluator.evaluate(
        question, padded, metrics=["answer_relevancy"]
    ).answer_relevancy
    assert good_score > padded_score + 0.2
