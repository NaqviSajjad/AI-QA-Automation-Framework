"""Tool selection, arguments and authorisation.

"The agent called the right tool" is only half the assertion. The other half -
with what arguments, and could it have called something it shouldn't - is where
the real defects live.
"""

from __future__ import annotations

import pytest

from src.agents.tools import (
    ALLOWED_TOOLS,
    ARGUMENT_SCHEMAS,
    SENSITIVE_TOOLS,
    TOOL_REGISTRY,
    ToolArgumentError,
    ToolAuthorizationError,
    execute_tool,
    is_authorized,
    validate_arguments,
)
from src.evaluation import judge
from src.utils.data_loader import case_ids, load_support_cases

pytestmark = [pytest.mark.agent]

TOOL_CASES = [c for c in load_support_cases() if c.expected_tool]


# --------------------------------------------------------------------------- #
# Selection
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("case", TOOL_CASES, ids=case_ids(TOOL_CASES))
def test_the_expected_tool_is_selected(case, support_agent, thresholds):
    state = support_agent.run(case.user_query)
    score = judge.tool_correctness(
        case.expected_tool, state.selected_tool, thresholds["tool_correctness"]
    )
    assert score.passed, f"{case.id}: {score.reason}"


def test_no_tool_is_called_for_an_unclassified_request(support_agent):
    state = support_agent.run("Hello, are you there?")
    assert state.selected_tool is None
    assert state.tool_calls == []


def test_at_most_one_tool_call_per_turn(support_agent):
    for case in TOOL_CASES[:8]:
        state = support_agent.run(case.user_query)
        assert len(state.tool_calls) <= 1


# --------------------------------------------------------------------------- #
# Arguments
# --------------------------------------------------------------------------- #
def test_tool_arguments_are_valid_and_typed(support_agent):
    state = support_agent.run("I was charged twice this month.")
    call = state.tool_calls[0]

    assert call.name == "get_billing_history"
    assert call.arguments["account_id"] == state.account_id
    assert isinstance(call.arguments["months"], int)
    assert call.authorized and call.error is None


def test_ticket_arguments_carry_the_customer_context(support_agent):
    state = support_agent.run("My export keeps failing and never arrives by email.")
    call = state.tool_calls[0]

    assert call.name == "create_support_ticket"
    assert call.arguments["summary"]
    assert call.arguments["severity"] in {"1", "2", "3"}
    assert call.result["ticket_id"].startswith("TCK-")


@pytest.mark.parametrize("tool,arguments", [
    ("get_billing_history", {"account_id": "ACC-1", "months": 0}),
    ("get_billing_history", {"account_id": "ACC-1001", "months": 99}),
    ("get_account_status", {}),
    ("get_account_status", {"account_id": "ACC-1001", "extra": "nope"}),
    ("create_support_ticket", {"account_id": "ACC-1001", "summary": "hi"}),
])
def test_invalid_arguments_are_rejected(tool, arguments):
    with pytest.raises(ToolArgumentError):
        validate_arguments(tool, arguments)


@pytest.mark.parametrize("key", ["card_number", "cvc", "password", "api_key"])
def test_sensitive_arguments_are_never_forwarded_to_a_tool(key):
    with pytest.raises(ToolArgumentError):
        validate_arguments("get_account_status", {"account_id": "ACC-1001", key: "4242424242424242"})


def test_every_registered_tool_has_an_argument_schema():
    assert set(TOOL_REGISTRY) == set(ARGUMENT_SCHEMAS)


# --------------------------------------------------------------------------- #
# Authorisation
# --------------------------------------------------------------------------- #
def test_allowlisted_tools_execute():
    result = execute_tool("get_subscription", {"account_id": "ACC-1001"})
    assert result["plan"]


@pytest.mark.parametrize("tool", sorted(SENSITIVE_TOOLS))
def test_tools_outside_the_allowlist_are_blocked(tool):
    with pytest.raises(ToolAuthorizationError):
        execute_tool(tool, {"account_id": "ACC-1001"})
    assert not is_authorized(tool)


def test_the_allowlist_and_the_registry_agree():
    assert set(TOOL_REGISTRY) == ALLOWED_TOOLS
    assert not (ALLOWED_TOOLS & SENSITIVE_TOOLS)


def test_no_agent_run_ever_attempts_an_unauthorised_tool(support_agent):
    for case in load_support_cases():
        state = support_agent.run(case.user_query)
        assert state.unauthorized_tool_attempts == [], (
            f"{case.id} attempted {state.unauthorized_tool_attempts}"
        )
        assert all(call.name in ALLOWED_TOOLS for call in state.tool_calls)
