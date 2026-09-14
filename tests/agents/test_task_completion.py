"""Task completion: did the workflow actually resolve the customer's request?

Relevance and faithfulness can both look fine while the customer still leaves
without a resolution. Completion is the metric that asks whether the *support
outcome* was achieved - right intent, actionable next step, right escalation
decision.
"""

from __future__ import annotations

import pytest

from src.evaluation import judge
from src.utils.data_loader import case_by_id, case_ids, load_support_cases

pytestmark = [pytest.mark.agent]

COMPLETION_CASES = [c for c in load_support_cases() if "multi-turn" not in c.tags]


@pytest.mark.parametrize("case", COMPLETION_CASES, ids=case_ids(COMPLETION_CASES))
def test_the_support_outcome_is_achieved(case, support_agent, thresholds):
    state = support_agent.run(case.user_query)

    score = judge.task_completion(
        state.response,
        case.expected_behavior,
        escalated=state.requires_escalation,
        expected_escalation=case.expected_escalation,
        threshold=thresholds["task_completion"],
    )
    assert score.passed, f"{case.id}: {score.reason}\n  answer: {state.response}"


def test_escalation_decisions_match_the_dataset(support_agent):
    for case in load_support_cases():
        state = support_agent.run(case.user_query)
        assert state.requires_escalation == case.expected_escalation, (
            f"{case.id}: escalated={state.requires_escalation}, "
            f"expected={case.expected_escalation}"
        )


def test_multi_turn_case_is_resolved_by_the_follow_up(support_agent, thresholds):
    """The first turn is deliberately vague; the resolution belongs to turn two."""
    case = case_by_id("CS-034")
    history: list[dict[str, str]] = []

    first = support_agent.run(case.user_query)
    history += [
        {"role": "user", "content": case.user_query},
        {"role": "assistant", "content": first.response},
    ]

    follow_up = case.follow_ups[0]
    second = support_agent.run(follow_up, history=history)

    assert second.category in {"payment", "billing"}
    assert second.selected_tool is not None
    score = judge.task_completion(
        second.response,
        case.follow_up_expected_behavior or case.expected_behavior,
        threshold=thresholds["task_completion"],
    )
    assert score.passed, f"{case.id} follow-up: {score.reason}\n  answer: {second.response}"


def test_every_answer_offers_a_way_forward(support_agent):
    """No dead ends: an answer either resolves, asks, or escalates."""
    for case in load_support_cases():
        answer = support_agent.run(case.user_query).response.lower()
        assert any(
            marker in answer
            for marker in ("next step", "could you", "would you like", "i can ",
                           "please confirm", "handing this over", "specialist")
        ), f"{case.id} left the customer with nothing to do: {answer}"
