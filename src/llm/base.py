"""LLM provider contract.

Everything above this layer (agent, RAG, evaluators, the demo app) depends on
`LLMProvider` only. Swapping OpenAI for a mock - or for a provider that does
not exist yet - is a factory change, not a framework change.
"""

from __future__ import annotations

import abc
import json
import re
from typing import Any

from pydantic import BaseModel, Field


class LLMError(RuntimeError):
    """Base class for provider failures surfaced to the application."""


class LLMTimeoutError(LLMError):
    """The provider did not answer inside the configured timeout."""


class LLMEmptyResponseError(LLMError):
    """The provider answered, but with nothing usable."""


class LLMResponse(BaseModel):
    """A single completion plus the metadata the QA layer needs.

    Latency and token counts are captured here because performance and cost
    assertions (tests/performance) are part of AI quality, not an afterthought.
    """

    text: str
    model: str
    provider: str
    latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str = "stop"
    raw: dict[str, Any] = Field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def json_payload(self) -> dict[str, Any]:
        """Best-effort extraction of a JSON object from the completion.

        LLMs wrap JSON in prose or fences often enough that the parsing rule
        belongs in one place - and the framework tests that rule directly.
        """
        text = self.text.strip()
        fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if fenced:
            text = fenced.group(1)
        else:
            start, end = text.find("{"), text.rfind("}")
            if start != -1 and end > start:
                text = text[start : end + 1]
        return json.loads(text)


class LLMProvider(abc.ABC):
    """Minimal, synchronous provider interface."""

    name: str = "base"

    def __init__(self, model: str, temperature: float = 0.0, **kwargs: Any) -> None:
        self.model = model
        self.temperature = temperature
        self.options = kwargs
        self.call_count = 0

    @abc.abstractmethod
    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Return the completion text for `prompt`."""

    @abc.abstractmethod
    def complete(self, prompt: str, **kwargs: Any) -> LLMResponse:
        """Return the completion plus metadata."""

    def embed(self, texts: list[str]) -> list[list[float]]:  # pragma: no cover - optional
        raise NotImplementedError(f"{self.name} provider does not implement embeddings")

    def reset(self) -> None:
        self.call_count = 0
