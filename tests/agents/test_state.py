"""State model and individual node transitions.

Each node is tested on its own, with a hand-built state. When the workflow test
fails, these say which node did it.
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from src.agents import nodes
from src.agents.state import AgentState, SupportResponse, ToolCall
from src.evaluation.deterministic import (
    SUPPORT_RESPONSE_SCHEMA,
    parse_json_block,
    validate_json_schema,
    validate_model,
)

pytestmark = [pytest.mark.agent]


# --------------------------------------------------------------------------- #
# Structured output contract
# --------------------------------------------------------------------------- #
def test_valid_structured_response_is_accepted():
    payload = {
        "category": "billing",
        "priority": "high",
        "requires_escalation": False,
        "response": "I can help investigate this duplicate charge and confirm the next steps.",
    }
    assert validate_json_schema(payload) == []
    assert validate_model(payload, SupportResponse) == []


@pytest.mark.parametrize("payload,broken", [
    ({"priority": "high", "requires_escalation": False, "response": "x" * 40}, "category"),
    ({"category": "billing", "requires_escalation": False, "response": "x" * 40}, "priority"),
    ({"category": "billing", "priority": "high", "response": "x" * 40}, "requires_escalation"),
])
def test_missing_required_fields_are_rejected(payload, broken):
    errors = validate_json_schema(payload)
    assert any(broken in error for error in errors)


def test_invalid_enum_value_is_rejected():
    payload = {
        "category": "spaceship", "priority": "high",
        "requires_escalation": False, "response": "x" * 40,
    }
    assert validate_json_schema(payload)
    assert validate_model(payload, SupportResponse)


def test_wrong_type_is_rejected():
    payload = {
        "category": "billing", "priority": "high",
        "requires_escalation": "yes please", "response": "x" * 40,
    }
    assert validate_json_schema(payload)


def test_empty_or_trivial_response_is_rejected():
    with pytest.raises(ValidationError):
        SupportResponse(category="billing", priority="low", requires_escalation=False, response="")
    with pytest.raises(ValidationError):
        SupportResponse(category="billing", priority="low", requires_escalation=False, response="ok")


def test_unexpected_fields_are_refused_by_the_model():
    with pytest.raises(ValidationError):
        SupportResponse(
            category="billing", priority="low", requires_escalation=False,
            response="I can help with this billing question right away.",
            internal_reasoning="should not be here",
        )


def test_malformed_json_is_reported_not_swallowed():
    with pytest.raises(json.JSONDecodeError):
        parse_json_block('{"category": "billing", ,,}')


def test_schema_is_the_same_shape_as_the_model():
    assert set(SUPPORT_RESPONSE_SCHEMA["required"]) <= set(SupportResponse.model_fields)


# --------------------------------------------------------------------------- #
# State transitions
# --------------------------------------------------------------------------- #
def test_fresh_state_has_sensible_defaults():
    state = AgentState(user_query="I was charged twice.")
    assert state.conversation_id.startswith("conv_")
    assert state.trace_id.startswith("trace_")
    assert state.retry_count == 0
    assert not state.completed


def test_classification_populates_category_and_priority(mock_llm):
    state = nodes.classify_node(AgentState(user_query="I was charged twice this month."), mock_llm)
    assert state.category == "billing"
    assert state.priority in {"medium", "high", "critical"}
    assert state.node_path == ["classify"]


def test_retrieval_populates_context_and_sources(mock_llm):
    state = AgentState(user_query="How long does a refund take?", category="refund")
    state = nodes.retrieve_node(state)
    assert state.retrieved_context
    assert "refunds.md" in state.retrieved_sources


def test_tool_node_records_the_call(mock_llm):
    state = AgentState(user_query="I was charged twice.", category="billing")
    state = nodes.tool_node(state)
    assert state.selected_tool == "get_billing_history"
    assert isinstance(state.tool_calls[0], ToolCall)
    assert state.tool_result


def test_tool_budget_is_enforced():
    state = AgentState(user_query="I was charged twice.", category="billing")
    state.tool_calls = [ToolCall(name="get_billing_history") for _ in range(3)]
    state = nodes.tool_node(state)
    assert "tool call budget exceeded" in state.validation_errors


def test_validation_rejects_an_empty_answer():
    state = AgentState(user_query="x", category="billing", priority="low")
    state.response = ""
    state = nodes.validate_node(state)
    assert "response is empty" in state.validation_errors


def test_validation_rejects_a_forbidden_claim():
    state = AgentState(user_query="x", category="refund", priority="low")
    state.response = "Good news - I have already issued your refund and the money is on its way."
    state = nodes.validate_node(state)
    assert any("forbidden claim" in error for error in state.validation_errors)


def test_validation_allows_a_quoted_policy_that_mentions_the_same_words():
    """The negation guard: quoting the rule is not breaking it."""
    state = AgentState(user_query="x", category="refund", priority="low")
    state.response = (
        "An agent must not state that a refund has been processed before finance confirms it, "
        "so I will raise the request and keep you updated."
    )
    state = nodes.validate_node(state)
    assert state.validation_errors == []


def test_valid_answer_produces_the_structured_response():
    state = AgentState(user_query="x", category="billing", priority="high")
    state.response = (
        "I can help investigate this duplicate charge, and a confirmed duplicate is "
        "reversed to the original payment method."
    )
    state = nodes.validate_node(state)
    assert state.validation_errors == []
    assert isinstance(state.structured_response, SupportResponse)
    assert state.completed


def test_retry_clears_the_failed_answer():
    state = AgentState(user_query="x", category="billing", priority="low")
    state.validation_errors = ["response is empty"]
    state = nodes.retry_node(state)
    assert state.retry_count == 1
    assert state.response is None
    assert state.validation_errors == []


def test_retry_decision_escalates_after_the_budget():
    state = AgentState(user_query="x", category="billing", priority="low")
    state.validation_errors = ["response is empty"]
    state.retry_count = 99
    assert nodes.should_retry(state) == "escalate"


def test_escalation_marks_the_state_and_names_the_reference():
    state = AgentState(user_query="x", category="billing", priority="low")
    state = nodes.escalate_node(state)
    assert state.requires_escalation
    assert state.conversation_id in state.response
    assert state.priority in {"high", "critical"}


def test_evidence_combines_context_and_tool_output():
    state = AgentState(user_query="x", category="billing")
    state.retrieved_context = ["policy text"]
    state.tool_result = {"duplicate_suspected": True}
    assert len(state.evidence) == 2
    assert "duplicate_suspected" in state.evidence[1]


def test_public_dict_hides_internal_reasoning():
    state = AgentState(user_query="x", category="billing", priority="low")
    state.response = "A duplicate charge is reversed to the original payment method."
    body = state.to_public_dict()
    assert "retrieved_context" not in body
    assert "structured_response" not in body
    assert set(body) >= {"conversation_id", "category", "priority", "response"}
