"""Chat UI behaviour.

These tests answer "did the interface work?" and nothing else. Whether the
answer was *good* is the job of tests/ai - keeping the two separate is what
makes a red build interpretable.
"""

from __future__ import annotations

import pytest

from src.utils.data_loader import case_by_id, load_support_cases

pytestmark = [pytest.mark.ui]

FUNCTIONAL_CASES = ["CS-001", "CS-006", "CS-010", "CS-015", "CS-020", "CS-025"]


@pytest.mark.smoke
def test_chat_page_is_ready(chatbot_page):
    assert chatbot_page.is_loaded
    assert chatbot_page.input.is_send_enabled
    assert chatbot_page.history.is_empty()
    assert not chatbot_page.is_escalated()
    assert not chatbot_page.has_error()


@pytest.mark.smoke
def test_sending_a_message_produces_an_answer(chatbot_page):
    chatbot_page.send_message("I was charged twice this month.")
    assert chatbot_page.get_latest_response()
    assert chatbot_page.history.user_count == 1
    assert chatbot_page.history.ai_count == 1
    assert not chatbot_page.has_error()


def test_input_is_cleared_after_sending(chatbot_page):
    chatbot_page.send_message("My card was declined.")
    assert chatbot_page.input.value == ""


def test_conversation_id_is_issued_and_stable(chatbot_page):
    assert chatbot_page.get_conversation_id() == ""
    chatbot_page.send_message("I want a refund for the payment I made four days ago.")
    first = chatbot_page.get_conversation_id()
    assert first.startswith("conv_")

    chatbot_page.send_message("Is that within the refund window?")
    assert chatbot_page.get_conversation_id() == first


def test_loading_indicator_is_hidden_once_the_answer_arrives(chatbot_page):
    chatbot_page.send_message("How do I upgrade my plan?")
    assert not chatbot_page.is_loading()


def test_empty_message_is_rejected_client_side(chatbot_page):
    chatbot_page.input.type("   ").send()
    assert chatbot_page.has_error()
    assert chatbot_page.history.ai_count == 0


@pytest.mark.parametrize("case_id", FUNCTIONAL_CASES)
def test_each_support_category_gets_an_answer(chatbot_page, case_id):
    case = case_by_id(case_id)
    chatbot_page.send_message(case.user_query)

    response = chatbot_page.get_latest_response()
    assert response, f"{case_id} produced no visible answer"
    assert len(response.split()) >= 8
    assert chatbot_page.get_category() == case.category


def test_ui_reports_the_classification_and_priority(chatbot_page):
    chatbot_page.send_message("I am locked out after too many failed sign in attempts.")
    message = chatbot_page.get_latest_message()
    assert message is not None
    assert message.category == "account"
    assert message.priority in {"low", "medium", "high", "critical"}


@pytest.mark.smoke
def test_escalation_indicator_appears_when_a_human_is_requested(chatbot_page):
    chatbot_page.send_message(case_by_id("CS-035").user_query)
    assert chatbot_page.is_escalated()
    assert "specialist" in chatbot_page.get_latest_response().lower()


def test_dataset_is_large_enough_to_be_meaningful():
    cases = load_support_cases()
    assert len(cases) >= 30
    assert len({c.category for c in cases}) >= 6
