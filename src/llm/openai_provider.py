"""OpenAI-compatible provider.

Works against api.openai.com or any OpenAI-compatible endpoint (Azure OpenAI,
vLLM, Ollama's compat layer, OpenRouter) via `OPENAI_BASE_URL`. The `openai`
package is an optional dependency - importing this module without it raises a
clear error instead of failing somewhere deep in a test.
"""

from __future__ import annotations

import time
from typing import Any

from src.llm.base import (
    LLMEmptyResponseError,
    LLMError,
    LLMProvider,
    LLMResponse,
    LLMTimeoutError,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)


class OpenAIProvider(LLMProvider):
    name = "openai"

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout_s: int = 30,
        max_retries: int = 2,
        max_tokens: int = 600,
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model, temperature=temperature, **kwargs)
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise LLMError(
                "The 'openai' package is required for LLM_PROVIDER=openai. "
                "Install it with: pip install '.[ai]'"
            ) from exc

        if not api_key:
            raise LLMError("OPENAI_API_KEY is not set. Use LLM_PROVIDER=mock for credential-free runs.")

        self.max_tokens = max_tokens
        self.timeout_s = timeout_s
        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url or None,
            timeout=timeout_s,
            max_retries=max_retries,
        )

    def generate(self, prompt: str, **kwargs: Any) -> str:
        return self.complete(prompt, **kwargs).text

    def complete(self, prompt: str, **kwargs: Any) -> LLMResponse:
        from openai import APITimeoutError, OpenAIError

        system = kwargs.pop("system", None)
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        started = time.perf_counter()
        self.call_count += 1
        try:
            completion = self._client.chat.completions.create(
                model=kwargs.pop("model", self.model),
                messages=messages,  # type: ignore[arg-type]
                temperature=kwargs.pop("temperature", self.temperature),
                max_tokens=kwargs.pop("max_tokens", self.max_tokens),
                **kwargs,
            )
        except APITimeoutError as exc:
            raise LLMTimeoutError(f"OpenAI request timed out after {self.timeout_s}s") from exc
        except OpenAIError as exc:
            raise LLMError(f"OpenAI request failed: {exc}") from exc

        latency_ms = (time.perf_counter() - started) * 1000
        choice = completion.choices[0]
        text = (choice.message.content or "").strip()
        if not text:
            raise LLMEmptyResponseError("OpenAI returned an empty completion")

        usage = completion.usage
        logger.info(
            "llm_call",
            extra={
                "provider": self.name,
                "model": self.model,
                "latency_ms": round(latency_ms, 2),
                "total_tokens": getattr(usage, "total_tokens", 0),
            },
        )
        return LLMResponse(
            text=text,
            model=completion.model,
            provider=self.name,
            latency_ms=round(latency_ms, 2),
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            finish_reason=choice.finish_reason or "stop",
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        model = self.options.get("embedding_model", "text-embedding-3-small")
        response = self._client.embeddings.create(model=model, input=texts)
        return [item.embedding for item in response.data]
