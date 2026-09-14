"""Sign-in behaviour. Pure deterministic UI testing - no AI involved."""

from __future__ import annotations

import pytest

from src.utils.config import settings

pytestmark = [pytest.mark.ui]


@pytest.mark.smoke
def test_login_page_loads(login_page):
    login_page.open()
    assert login_page.is_loaded
    assert not login_page.has_error


@pytest.mark.smoke
def test_valid_credentials_open_the_chat(login_page):
    login_page.open().login_as_default_user().wait_for_chat()
    assert login_page.current_path == "/chat"


def test_invalid_password_is_rejected(login_page):
    login_page.open().login(settings.auth.email, "not-the-password")
    assert "invalid" in login_page.error_text().lower()
    assert login_page.current_path == "/login"


def test_unknown_user_is_rejected(login_page):
    login_page.open().login("nobody@example.com", settings.auth.password)
    assert login_page.error_text()
    assert login_page.current_path == "/login"


def test_empty_form_is_blocked_by_client_validation(login_page):
    login_page.open().submit_empty()
    assert login_page.current_path == "/login"
    assert login_page.email_validation_message() != ""


def test_chat_is_not_reachable_while_signed_out(page, app_server):
    page.goto(f"{app_server}/chat", wait_until="domcontentloaded")
    assert page.url.endswith("/login")


@pytest.mark.smoke
def test_logout_returns_to_login_and_revokes_access(login_page, app_server):
    """Signs in through the UI on purpose.

    Logging out invalidates the server-side session, so this test must not reuse
    the shared authenticated state the rest of the suite depends on - that is
    exactly the kind of hidden coupling that produces "passes alone, fails in
    the suite" flakiness.
    """
    from pages.chatbot_page import ChatbotPage

    login_page.open().login_as_default_user().wait_for_chat()
    chat = ChatbotPage(login_page.page, app_server)
    chat.logout()

    chat.page.goto(f"{app_server}/chat", wait_until="domcontentloaded")
    assert chat.page.url.endswith("/login")
