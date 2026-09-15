"""Page-object base.

Two rules hold the UI layer together:

1. tests never see a selector - they call intention-revealing methods;
2. page objects never assert - they expose state, and the test decides.

Everything is located by `data-testid`, so restyling the application cannot
break the suite.
"""

from __future__ import annotations

from typing import Literal

from playwright.sync_api import Locator, Page

from src.utils.config import settings
from src.utils.logging import get_logger

logger = get_logger(__name__)


class BasePage:
    path: str = "/"

    def __init__(self, page: Page, base_url: str | None = None) -> None:
        self.page = page
        self.base_url = (base_url or settings.app.base_url).rstrip("/")

    # -- navigation --------------------------------------------------------- #
    def open(self, path: str | None = None) -> BasePage:
        url = f"{self.base_url}{path or self.path}"
        self.page.goto(url, wait_until="domcontentloaded")
        return self

    @property
    def current_path(self) -> str:
        from urllib.parse import urlparse

        return urlparse(self.page.url).path

    # -- locating ----------------------------------------------------------- #
    def testid(self, name: str) -> Locator:
        return self.page.get_by_test_id(name)

    def is_visible(self, name: str) -> bool:
        return self.testid(name).is_visible()

    def text_of(self, name: str) -> str:
        return (self.testid(name).inner_text() or "").strip()

    def wait_for(
        self,
        name: str,
        state: Literal["attached", "detached", "hidden", "visible"] = "visible",
        timeout: int | None = None,
    ) -> Locator:
        locator = self.testid(name)
        locator.wait_for(state=state, timeout=timeout or settings.playwright.default_timeout_ms)
        return locator

    def screenshot(self, name: str) -> bytes:
        return self.page.screenshot(path=str(settings.reports_path / f"{name}.png"), full_page=True)
