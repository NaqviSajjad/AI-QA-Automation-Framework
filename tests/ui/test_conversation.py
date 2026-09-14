"""Multi-turn conversation behaviour in the UI."""

from __future__ import annotations

import pytest

from pages.conversation_page import ConversationPage

pytestmark = [pytest.mark.ui]


def test_turns_are_recorded_in_order(chatbot_page):
    chatbot_page.have_conversation(
        ["I have a payment problem.", "My card was declined yesterday."]
    )
    conversation = ConversationPage(chatbot_page.page, chatbot_page.base_url)

    assert conversation.turns() == 4
    assert conversation.is_alternating()
    assert conversation.user_messages() == [
        "I have a payment problem.",
        "My card was declined yesterday.",
    ]


def test_context_is_retained_across_turns(chatbot_page):
    chatbot_page.send_message("I have a payment problem.")
    chatbot_page.send_message("It happened yesterday, what should I do?")

    second = chatbot_page.get_latest_response().lower()
    # The follow-up alone is meaningless; only a system that kept the first turn
    # can still be talking about payments here.
    assert any(word in second for word in ("payment", "card", "charge", "bank"))


def test_follow_up_keeps_the_same_conversation(chatbot_page):
    chatbot_page.send_message("How do I upgrade my plan?")
    conversation_id = chatbot_page.get_conversation_id()
    chatbot_page.send_message("And when does that take effect?")

    assert chatbot_page.get_conversation_id() == conversation_id
    assert chatbot_page.history.ai_count == 2


def test_history_shows_every_answer(chatbot_page):
    chatbot_page.have_conversation(
        ["I was charged twice this month.", "How long does a refund take?"]
    )
    responses = chatbot_page.get_all_responses()
    assert len(responses) == 2
    assert all(responses)
    assert responses[0] != responses[1]


def test_reset_clears_the_conversation(chatbot_page):
    chatbot_page.send_message("I cannot sign in.")
    assert chatbot_page.history.ai_count == 1

    chatbot_page.reset_conversation()
    assert chatbot_page.history.is_empty()
    assert chatbot_page.get_conversation_id() == ""
    assert not chatbot_page.is_escalated()


def test_a_new_conversation_gets_a_new_id(chatbot_page):
    chatbot_page.send_message("I cannot sign in.")
    first = chatbot_page.get_conversation_id()

    chatbot_page.reset_conversation()
    chatbot_page.send_message("I cannot sign in.")
    assert chatbot_page.get_conversation_id() not in ("", first)
