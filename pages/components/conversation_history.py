"""The scrolling transcript component."""

from __future__ import annotations

from playwright.sync_api import Page

from pages.components.chat_message import ChatMessage


class ConversationHistory:
    def __init__(self, page: Page) -> None:
        self.page = page
        self.root = page.get_by_test_id("conversation-history")
        self.user_messages = page.get_by_test_id("user-message")
        self.ai_messages = page.get_by_test_id("ai-response")

    @property
    def ai_count(self) -> int:
        return self.ai_messages.count()

    @property
    def user_count(self) -> int:
        return self.user_messages.count()

    @property
    def turn_count(self) -> int:
        return self.ai_count + self.user_count

    def latest_ai(self) -> ChatMessage | None:
        if not self.ai_count:
            return None
        return ChatMessage.from_locator(self.ai_messages.nth(self.ai_count - 1), "assistant")

    def latest_user(self) -> ChatMessage | None:
        if not self.user_count:
            return None
        return ChatMessage.from_locator(self.user_messages.nth(self.user_count - 1), "user")

    def all_ai(self) -> list[ChatMessage]:
        return [
            ChatMessage.from_locator(self.ai_messages.nth(i), "assistant")
            for i in range(self.ai_count)
        ]

    def all_messages(self) -> list[ChatMessage]:
        """Every bubble in DOM order, so multi-turn context can be asserted."""
        messages: list[ChatMessage] = []
        for locator in self.root.locator("[data-testid]").all():
            testid = locator.get_attribute("data-testid")
            if testid == "user-message":
                messages.append(ChatMessage.from_locator(locator, "user"))
            elif testid == "ai-response":
                messages.append(ChatMessage.from_locator(locator, "assistant"))
        return messages

    def is_empty(self) -> bool:
        return self.turn_count == 0
