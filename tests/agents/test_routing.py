"""Routing: does the agent send each request down the right branch?"""

from __future__ import annotations

import pytest

from src.agents import nodes
from src.agents.state import VALID_CATEGORIES, AgentState
from src.utils.data_loader import case_ids, load_support_cases

pytestmark = [pytest.mark.agent]

ROUTABLE_CASES = [c for c in load_support_cases() if c.category in VALID_CATEGORIES]


@pytest.mark.parametrize("case", ROUTABLE_CASES, ids=case_ids(ROUTABLE_CASES))
def test_requests_are_classified_into_the_expected_category(case, support_agent):
    state = support_agent.run(case.user_query)
    assert state.category == case.category, (
        f"{case.id}: '{case.user_query}' -> {state.category}, expected {case.category}"
    )


@pytest.mark.parametrize("query,expected", [
    ("I was charged twice this month.", "billing"),
    ("I want my money back for last week's payment.", "refund"),
    ("My card was declined again.", "payment"),
    ("How do I downgrade my plan?", "subscription"),
    ("I am locked out of my account.", "account"),
    ("The export keeps failing with a 500 error.", "technical"),
    ("Hello, are you there?", "unknown"),
])
def test_representative_intents_route_correctly(query, expected, support_agent):
    assert support_agent.run(query).category == expected


def test_classification_is_always_inside_the_enum(support_agent):
    """Whatever the model returns, the workflow only ever carries a valid label."""
    for case in load_support_cases():
        state = support_agent.run(case.user_query)
        assert state.category in VALID_CATEGORIES
        assert state.priority in {"low", "medium", "high", "critical"}


def test_router_sends_knowledge_questions_to_retrieval():
    state = AgentState(user_query="How long does a refund take?", category="refund")
    assert nodes.route(state) == "retrieve"


def test_router_sends_unknown_intent_to_clarification():
    state = AgentState(user_query="hello", category="unknown")
    assert nodes.route(state) == "clarify"


def test_router_short_circuits_to_escalation():
    """Escalation wins over everything: a customer asking for a human gets one."""
    state = AgentState(user_query="let me speak to a manager", category="billing")
    state.requires_escalation = True
    assert nodes.route(state) == "escalate"


@pytest.mark.parametrize("query", [
    "I want to speak to a human about this.",
    "Put me through to a manager, this is fraud.",
    "I will take legal action if this is not fixed.",
])
def test_escalation_triggers_are_detected(query, support_agent):
    state = support_agent.run(query)
    assert state.requires_escalation
    assert "escalate" in state.node_path


def test_priority_rises_with_severity(support_agent):
    low = support_agent.run("How do I upgrade my plan?")
    high = support_agent.run("I was charged twice this month.")
    critical = support_agent.run("There is a fraudulent charge on my account, I want a manager.")

    order = ["low", "medium", "high", "critical"]
    assert order.index(low.priority) < order.index(high.priority) <= order.index(critical.priority)
