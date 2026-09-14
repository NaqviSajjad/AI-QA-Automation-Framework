"""Provider factory + registry.

Adding a provider is one `register_provider()` call - no other module changes.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from src.llm.base import LLMError, LLMProvider
from src.llm.mock_provider import MockLLMProvider
from src.utils.config import settings
from src.utils.logging import get_logger

logger = get_logger(__name__)

_REGISTRY: dict[str, Callable[..., LLMProvider]] = {}


def register_provider(name: str, builder: Callable[..., LLMProvider]) -> None:
    _REGISTRY[name.lower()] = builder


def available_providers() -> list[str]:
    return sorted(_REGISTRY)


def _build_openai(**kwargs: Any) -> LLMProvider:
    from src.llm.openai_provider import OpenAIProvider  # imported lazily: optional dep

    return OpenAIProvider(**kwargs)


register_provider("mock", MockLLMProvider)
register_provider("openai", _build_openai)


def get_llm_provider(provider: str | None = None, **overrides: Any) -> LLMProvider:
    """Return a provider instance. Defaults come from config/env."""
    name = (provider or settings.llm.provider).lower()
    if name not in _REGISTRY:
        raise LLMError(f"Unknown LLM provider '{name}'. Available: {available_providers()}")

    kwargs: dict[str, Any] = {
        "model": settings.llm.model,
        "temperature": settings.llm.temperature,
    }
    if name == "openai":
        kwargs.update(
            api_key=settings.llm.api_key,
            base_url=settings.llm.base_url,
            timeout_s=settings.llm.timeout_s,
            max_retries=settings.llm.max_retries,
            max_tokens=settings.llm.max_tokens,
        )
    else:
        kwargs["model"] = overrides.pop("model", "mock-support-1")
    kwargs.update(overrides)

    logger.info("llm_provider_selected", extra={"provider": name, "model": kwargs.get("model")})
    return _REGISTRY[name](**kwargs)
