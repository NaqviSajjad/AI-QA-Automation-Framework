"""Sign-in page."""

from __future__ import annotations

from pages.base_page import BasePage
from src.utils.config import settings


class LoginPage(BasePage):
    path = "/login"

    # -- actions ------------------------------------------------------------ #
    def login(self, email: str, password: str) -> LoginPage:
        self.testid("email-input").fill(email)
        self.testid("password-input").fill(password)
        self.testid("login-button").click()
        return self

    def login_as_default_user(self) -> LoginPage:
        return self.login(settings.auth.email, settings.auth.password)

    def submit_empty(self) -> LoginPage:
        self.testid("login-button").click()
        return self

    def wait_for_chat(self, timeout: int | None = None) -> LoginPage:
        self.page.wait_for_url("**/chat", timeout=timeout or settings.playwright.navigation_timeout_ms)
        return self

    # -- state -------------------------------------------------------------- #
    @property
    def is_loaded(self) -> bool:
        return self.testid("login-form").is_visible()

    @property
    def has_error(self) -> bool:
        return self.testid("error-message").is_visible()

    def error_text(self) -> str:
        self.wait_for("error-message")
        return self.text_of("error-message")

    def email_validation_message(self) -> str:
        return self.testid("email-input").evaluate("el => el.validationMessage")
