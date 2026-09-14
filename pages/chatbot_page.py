"""The customer-support chat page.

This is the class that hands captured AI output to the evaluation layer.
Playwright's job ends at `get_latest_response()`; judging that string is
somebody else's (src/evaluation). Keeping the boundary sharp is what stops the
UI layer turning into a half-built evaluation library.
"""

from __future__ import annotations

from playwright.sync_api import Page

from pages.base_page import BasePage
from pages.components.chat_input import ChatInput
from pages.components.chat_message import ChatMessage
from pages.components.conversation_history import ConversationHistory
from src.utils.config import settings


class ChatbotPage(BasePage):
    path = "/chat"

    RESPONSE_TIMEOUT_MS = 30_000

    def __init__(self, page: Page, base_url: str | None = None) -> None:
        super().__init__(page, base_url)
        self.input = ChatInput(page)
        self.history = ConversationHistory(page)

    # -- actions ------------------------------------------------------------ #
    def send_message(self, message: str, wait: bool = True) -> ChatbotPage:
        """Type a message, send it, and (by default) wait for the AI to answer."""
        before = self.history.ai_count
        self.input.submit(message)
        if wait:
            self.wait_for_response(expected_count=before + 1)
        return self

    def wait_for_response(self, expected_count: int | None = None, timeout: int | None = None) -> ChatbotPage:
        """Wait for a new assistant bubble, or for the error state to appear.

        Waiting on the *count* of responses rather than on a spinner disappearing
        is what keeps these tests off the flaky list: it is a statement about the
        outcome, not about an intermediate animation.
        """
        target = expected_count if expected_count is not None else self.history.ai_count + 1
        self.page.wait_for_function(
            """(target) => {
                const answers = document.querySelectorAll('[data-testid="ai-response"]').length;
                const error = document.querySelector('[data-testid="error-message"]');
                const errorShown = error && !error.classList.contains('hidden');
                return answers >= target || errorShown;
            }""",
            arg=target,
            timeout=timeout or self.RESPONSE_TIMEOUT_MS,
        )
        return self

    def retry(self) -> ChatbotPage:
        before = self.history.ai_count
        self.testid("retry-button").click()
        self.wait_for_response(expected_count=before + 1)
        return self

    def reset_conversation(self) -> ChatbotPage:
        self.testid("reset-button").click()
        self.page.wait_for_function(
            "() => document.querySelectorAll('[data-testid=\"ai-response\"]').length === 0"
        )
        return self

    def logout(self) -> ChatbotPage:
        self.testid("logout-button").click()
        self.page.wait_for_url("**/login", timeout=settings.playwright.navigation_timeout_ms)
        return self

    def have_conversation(self, messages: list[str]) -> list[str]:
        """Send several turns and return the assistant's answers in order."""
        answers = []
        for message in messages:
            self.send_message(message)
            answers.append(self.get_latest_response())
        return answers

    # -- state -------------------------------------------------------------- #
    @property
    def is_loaded(self) -> bool:
        return self.input.field.is_visible()

    def get_latest_response(self) -> str:
        message = self.history.latest_ai()
        return message.text if message else ""

    def get_latest_message(self) -> ChatMessage | None:
        return self.history.latest_ai()

    def get_all_responses(self) -> list[str]:
        return [message.text for message in self.history.all_ai()]

    def get_conversation_id(self) -> str:
        value = self.text_of("conversation-id")
        return "" if value in ("", "-") else value

    def get_category(self) -> str | None:
        message = self.history.latest_ai()
        return message.category if message else None

    def get_priority(self) -> str | None:
        message = self.history.latest_ai()
        return message.priority if message else None

    def is_escalated(self) -> bool:
        return self.testid("escalation-indicator").is_visible()

    def is_loading(self) -> bool:
        return self.testid("loading-indicator").is_visible()

    def has_error(self) -> bool:
        return self.testid("error-message").is_visible()

    def error_text(self) -> str:
        return self.text_of("error-message") if self.has_error() else ""

    def can_retry(self) -> bool:
        return self.testid("retry-button").is_visible()

    def turn_count(self) -> int:
        return self.history.turn_count
