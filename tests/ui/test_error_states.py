"""Failure paths.

An AI feature fails in ways a CRUD screen does not - the model times out, comes
back empty, or the service is briefly unavailable. The interface has to stay
honest and recoverable in all three cases, so each one is triggered
deliberately rather than hoped for.
"""

from __future__ import annotations

import pytest

from src.llm.mock_provider import FORCE_EMPTY, FORCE_ERROR, FORCE_TIMEOUT

pytestmark = [pytest.mark.ui]


@pytest.mark.usefixtures("requires_mock_provider")
def test_model_timeout_shows_an_error_and_a_retry(chatbot_page):
    chatbot_page.send_message(f"My card was declined {FORCE_TIMEOUT}")

    assert chatbot_page.has_error()
    assert chatbot_page.can_retry()
    assert "retry" in chatbot_page.error_text().lower()
    assert chatbot_page.history.ai_count == 0
    assert not chatbot_page.is_loading()


@pytest.mark.usefixtures("requires_mock_provider")
def test_empty_model_response_is_surfaced_not_swallowed(chatbot_page):
    chatbot_page.send_message(f"I want a refund {FORCE_EMPTY}")

    assert chatbot_page.has_error()
    assert chatbot_page.history.ai_count == 0


@pytest.mark.usefixtures("requires_mock_provider")
def test_upstream_failure_shows_a_recoverable_error(chatbot_page):
    chatbot_page.send_message(f"I cannot sign in {FORCE_ERROR}")

    assert chatbot_page.has_error()
    assert chatbot_page.can_retry()


def test_network_failure_is_reported_to_the_customer(chatbot_page):
    chatbot_page.page.route("**/api/chat", lambda route: route.abort())
    chatbot_page.send_message("I was charged twice this month.", wait=False)
    chatbot_page.wait_for("error-message")

    assert "network" in chatbot_page.error_text().lower()
    assert chatbot_page.can_retry()


def test_retry_after_a_transient_failure_succeeds(chatbot_page):
    calls = {"count": 0}

    def flaky(route):
        calls["count"] += 1
        if calls["count"] == 1:
            route.fulfill(status=502, content_type="application/json",
                          body='{"error":"ai_unavailable","message":"temporary failure"}')
        else:
            route.continue_()

    chatbot_page.page.route("**/api/chat", flaky)
    chatbot_page.send_message("I was charged twice this month.", wait=False)
    chatbot_page.wait_for("error-message")
    assert chatbot_page.can_retry()

    chatbot_page.retry()
    assert chatbot_page.get_latest_response()
    assert calls["count"] == 2


def test_server_error_does_not_lose_the_customer_message(chatbot_page):
    chatbot_page.page.route(
        "**/api/chat",
        lambda route: route.fulfill(status=500, content_type="application/json", body="{}"),
    )
    chatbot_page.send_message("I need help with my subscription.", wait=False)
    chatbot_page.wait_for("error-message")

    assert chatbot_page.history.user_count == 1
    assert chatbot_page.has_error()
