"""A single message bubble."""

from __future__ import annotations

from dataclasses import dataclass

from playwright.sync_api import Locator


@dataclass
class ChatMessage:
    role: str
    text: str
    meta: str = ""

    @classmethod
    def from_locator(cls, locator: Locator, role: str) -> ChatMessage:
        meta_locator = locator.get_by_test_id("ai-response-meta")
        meta = meta_locator.inner_text().strip() if meta_locator.count() else ""
        text = (locator.inner_text() or "").strip()
        if meta:
            text = text.replace(meta, "").strip()
        return cls(role=role, text=text, meta=meta)

    @property
    def category(self) -> str | None:
        """The classification the app displayed alongside the answer."""
        for part in self.meta.split("|"):
            if "category:" in part:
                return part.split("category:")[1].strip()
        return None

    @property
    def priority(self) -> str | None:
        for part in self.meta.split("|"):
            if "priority:" in part:
                return part.split("priority:")[1].strip()
        return None

    def __contains__(self, needle: str) -> bool:
        return needle.lower() in self.text.lower()

    def __str__(self) -> str:  # pragma: no cover - convenience
        return f"{self.role}: {self.text}"
