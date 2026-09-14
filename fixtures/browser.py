"""Playwright fixtures.

Built on top of pytest-playwright, so `--browser chromium --browser firefox
--browser webkit`, `--headed`, `--tracing`, `--video` and `--screenshot` all
work as documented upstream. What is added here is project-specific: viewport
and base URL, sensible timeouts, an authenticated storage state that is created
once per session, and ready-to-use page objects.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from playwright.sync_api import Browser, BrowserContext, Page

from pages.chatbot_page import ChatbotPage
from pages.login_page import LoginPage
from src.utils.config import settings


@pytest.fixture(scope="session")
def browser_context_args(browser_context_args: dict[str, Any], app_server: str) -> dict[str, Any]:
    return {
        **browser_context_args,
        "base_url": app_server,
        "viewport": settings.playwright.viewport,
        "ignore_https_errors": True,
        "locale": "en-GB",
    }


@pytest.fixture(scope="session")
def storage_state_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("auth") / "state.json"


@pytest.fixture(scope="session")
def authenticated_state(browser: Browser, app_server: str, storage_state_path: Path) -> Path:
    """Sign in once per session and reuse the cookie.

    Logging in through the UI in every test would be re-testing the login form
    fifty times and adding a second of latency to each case. `tests/ui/test_login`
    owns that behaviour; everything else starts already signed in.
    """
    context = browser.new_context(base_url=app_server, viewport=settings.playwright.viewport)
    page = context.new_page()
    login = LoginPage(page, base_url=app_server)
    login.open()
    login.login(settings.auth.email, settings.auth.password)
    login.wait_for_chat()
    context.storage_state(path=str(storage_state_path))
    context.close()
    return storage_state_path


@pytest.fixture
def authenticated_context(
    browser: Browser,
    browser_context_args: dict[str, Any],
    authenticated_state: Path,
    request: pytest.FixtureRequest,
) -> Iterator[BrowserContext]:
    context = browser.new_context(
        **browser_context_args, storage_state=json.loads(authenticated_state.read_text())
    )
    context.set_default_timeout(settings.playwright.default_timeout_ms)
    context.set_default_navigation_timeout(settings.playwright.navigation_timeout_ms)
    yield context
    context.close()


@pytest.fixture
def authenticated_page(authenticated_context: BrowserContext) -> Iterator[Page]:
    page = authenticated_context.new_page()
    yield page
    page.close()


@pytest.fixture
def login_page(page: Page, app_server: str) -> LoginPage:
    page.set_default_timeout(settings.playwright.default_timeout_ms)
    return LoginPage(page, base_url=app_server)


@pytest.fixture
def chatbot_page(authenticated_page: Page, app_server: str) -> ChatbotPage:
    """The signed-in support chat, already open and ready for a message."""
    chat = ChatbotPage(authenticated_page, base_url=app_server)
    chat.open()
    return chat
