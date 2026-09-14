"""Conversation-level view.

Wraps the transcript so multi-turn assertions (context retention, ordering,
reset) read as statements about the conversation rather than about the DOM.
"""

from __future__ import annotations

from playwright.sync_api import Page

from pages.base_page import BasePage
from pages.components.chat_message import ChatMessage
from pages.components.conversation_history import ConversationHistory


class ConversationPage(BasePage):
    path = "/chat"

    def __init__(self, page: Page, base_url: str | None = None) -> None:
        super().__init__(page, base_url)
        self.history = ConversationHistory(page)

    def transcript(self) -> list[ChatMessage]:
        return self.history.all_messages()

    def turns(self) -> int:
        return self.history.turn_count

    def user_messages(self) -> list[str]:
        return [m.text for m in self.transcript() if m.role == "user"]

    def assistant_messages(self) -> list[str]:
        return [m.text for m in self.transcript() if m.role == "assistant"]

    def is_alternating(self) -> bool:
        """User and assistant turns must alternate - a dropped turn is a bug."""
        roles = [m.role for m in self.transcript()]
        return all(roles[i] != roles[i + 1] for i in range(len(roles) - 1))

    def mentions_across_turns(self, needle: str) -> list[int]:
        return [
            index
            for index, message in enumerate(self.transcript())
            if needle.lower() in message.text.lower()
        ]

    def conversation_id(self) -> str:
        value = self.text_of("conversation-id")
        return "" if value in ("", "-") else value

    def is_empty(self) -> bool:
        return self.history.is_empty()
