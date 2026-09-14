"""API contract tests.

Status codes, schemas, auth and error bodies. Everything here is deterministic;
none of it depends on what the model said.
"""

from __future__ import annotations

import pytest

from src.agents.state import VALID_CATEGORIES, VALID_PRIORITIES
from src.evaluation.deterministic import validate_json_schema

pytestmark = [pytest.mark.api]

CHAT_RESPONSE_SCHEMA = {
    "type": "object",
    "required": ["conversation_id", "category", "priority", "requires_escalation", "response"],
    "properties": {
        "conversation_id": {"type": "string", "pattern": "^conv_[0-9a-f]{12}$"},
        "trace_id": {"type": "string"},
        "category": {"type": "string", "enum": list(VALID_CATEGORIES)},
        "priority": {"type": "string", "enum": list(VALID_PRIORITIES)},
        "requires_escalation": {"type": "boolean"},
        "response": {"type": "string", "minLength": 20},
        "tool_used": {"type": ["string", "null"]},
        "sources": {"type": "array", "items": {"type": "string"}},
        "latency_ms": {"type": "number", "minimum": 0},
        "llm_calls": {"type": "integer", "minimum": 0},
        "tool_calls": {"type": "integer", "minimum": 0},
        "retry_count": {"type": "integer", "minimum": 0},
        "node_path": {"type": "array", "items": {"type": "string"}},
    },
}


@pytest.mark.smoke
def test_health_reports_the_active_configuration(anonymous_api):
    response = anonymous_api.health()
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["llm_provider"] in {"mock", "openai"}
    assert body["agent_runtime"] in {"langgraph", "builtin"}


@pytest.mark.smoke
def test_login_returns_a_token(anonymous_api):
    response = anonymous_api.login()
    assert response.status_code == 200
    assert response.json()["token"]


def test_login_with_bad_credentials_is_401(anonymous_api):
    response = anonymous_api.login(password="wrong")
    assert response.status_code == 401
    assert "error" in response.json()


@pytest.mark.parametrize("path,method", [
    ("/chat", "POST"),
    ("/conversations/conv_000000000000", "GET"),
    ("/feedback", "POST"),
    ("/escalation", "POST"),
])
def test_endpoints_require_authentication(anonymous_api, path, method):
    response = anonymous_api.request(method, path, json={})
    assert response.status_code == 401


@pytest.mark.smoke
def test_chat_response_matches_the_schema(chat_api):
    response = chat_api.send_message("I was charged twice this month.")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")

    body = response.json()
    errors = validate_json_schema(body, CHAT_RESPONSE_SCHEMA)
    assert not errors, f"Schema violations: {errors}"


def test_chat_reports_the_tool_it_used(chat_api):
    body = chat_api.send_message("I was charged twice this month.").json()
    assert body["tool_used"] == "get_billing_history"
    assert body["tool_calls"] == 1
    assert body["sources"]


def test_chat_keeps_the_conversation_when_the_id_is_supplied(chat_api):
    first = chat_api.send_message("I have a payment problem.").json()
    second = chat_api.send_message(
        "It was declined yesterday.", conversation_id=first["conversation_id"]
    ).json()
    assert second["conversation_id"] == first["conversation_id"]


@pytest.mark.parametrize("payload,expected", [
    ({}, 422),
    ({"message": ""}, 422),
    ({"message": "x" * 5000}, 422),
    ({"message": 12345}, 422),
    ({"wrong_field": "hello"}, 422),
])
def test_malformed_requests_are_rejected(chat_api, payload, expected):
    assert chat_api.send_raw(payload).status_code == expected


def test_invalid_json_body_is_rejected(chat_api):
    assert chat_api.send_raw("{not json").status_code == 422


def test_conversation_can_be_fetched_and_reset(chat_api, conversation_api):
    conversation_id = chat_api.send_message("I cannot sign in.").json()["conversation_id"]

    fetched = conversation_api.get_conversation(conversation_id)
    assert fetched.status_code == 200
    body = fetched.json()
    assert body["message_count"] == 2
    assert body["messages"][0]["role"] == "user"

    assert conversation_api.reset_conversation(conversation_id).status_code == 200
    assert conversation_api.get_conversation(conversation_id).status_code == 404


def test_unknown_conversation_is_404(conversation_api):
    response = conversation_api.get_conversation("conv_doesnotexist")
    assert response.status_code == 404
    assert response.headers.get("X-Request-Id")


def test_feedback_is_recorded(chat_api):
    conversation_id = chat_api.send_message("How do I upgrade my plan?").json()["conversation_id"]
    response = chat_api.feedback(conversation_id, "up", "Clear answer")
    assert response.status_code == 201
    assert response.json()["status"] == "recorded"


def test_feedback_rating_is_constrained(chat_api):
    conversation_id = chat_api.send_message("How do I upgrade my plan?").json()["conversation_id"]
    assert chat_api.feedback(conversation_id, "sideways").status_code == 422


def test_escalation_creates_a_ticket(chat_api, escalation_api):
    conversation_id = chat_api.send_message("My export keeps failing.").json()["conversation_id"]
    response = escalation_api.escalate(conversation_id, "Customer asked for a human")
    assert response.status_code == 201

    body = response.json()
    assert body["ticket_id"].startswith("ESC-")
    assert body["status"] == "queued"


def test_escalation_requires_a_real_conversation(escalation_api):
    assert escalation_api.escalate("conv_nope", "reason").status_code == 404


@pytest.mark.usefixtures("requires_mock_provider")
def test_model_timeout_maps_to_504(chat_api):
    from src.llm.mock_provider import FORCE_TIMEOUT

    response = chat_api.send_message(f"I was charged twice {FORCE_TIMEOUT}")
    assert response.status_code == 504
    assert response.json()["error"] == "ai_timeout"


@pytest.mark.usefixtures("requires_mock_provider")
def test_empty_model_response_maps_to_502(chat_api):
    from src.llm.mock_provider import FORCE_EMPTY

    response = chat_api.send_message(f"I want a refund {FORCE_EMPTY}")
    assert response.status_code == 502
    assert response.json()["error"] == "ai_empty_response"
