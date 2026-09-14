"""The message composer component."""

from __future__ import annotations

from playwright.sync_api import Page


class ChatInput:
    def __init__(self, page: Page) -> None:
        self.page = page
        self.field = page.get_by_test_id("chat-input")
        self.send_button = page.get_by_test_id("send-button")

    def type(self, message: str) -> ChatInput:
        self.field.fill(message)
        return self

    def send(self) -> ChatInput:
        self.send_button.click()
        return self

    def submit(self, message: str) -> ChatInput:
        return self.type(message).send()

    @property
    def value(self) -> str:
        return self.field.input_value()

    @property
    def is_send_enabled(self) -> bool:
        return self.send_button.is_enabled()

    def placeholder(self) -> str:
        return self.field.get_attribute("placeholder") or ""
