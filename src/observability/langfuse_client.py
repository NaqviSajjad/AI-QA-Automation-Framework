"""Langfuse tracing - optional, and genuinely optional.

`LANGFUSE_ENABLED=false` (the default) gives a no-op tracer that still records
spans **in memory**. That matters for testing: the AI-failure-investigation
workflow in docs/observability.md can be exercised in CI with no external
service, and the same code paths light up Langfuse when credentials exist.
"""

from __future__ import annotations

import contextlib
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from src.utils.config import settings
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class Span:
    """One traced unit of work, with everything needed to debug an AI failure."""

    name: str
    trace_id: str
    input: dict[str, Any] = field(default_factory=dict)
    output: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    scores: dict[str, float] = field(default_factory=dict)
    started_at: float = field(default_factory=time.time)
    ended_at: float | None = None
    _handle: Any = None

    def update(self, **fields: Any) -> None:
        for key, value in fields.items():
            current = getattr(self, key, None)
            if isinstance(current, dict) and isinstance(value, dict):
                current.update(value)
            else:
                setattr(self, key, value)
        if self._handle is not None:  # pragma: no cover - needs real Langfuse
            try:
                self._handle.update(output=self.output, metadata={**self.metadata, **self.metrics})
            except Exception as exc:
                logger.warning("langfuse_update_failed", extra={"error": str(exc)})

    def score(self, name: str, value: float, comment: str | None = None) -> None:
        self.scores[name] = float(value)
        if self._handle is not None:  # pragma: no cover
            try:
                self._handle.score(name=name, value=float(value), comment=comment)
            except Exception as exc:
                logger.warning("langfuse_score_failed", extra={"error": str(exc)})

    @property
    def duration_ms(self) -> float:
        end = self.ended_at or time.time()
        return round((end - self.started_at) * 1000, 2)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "trace_id": self.trace_id,
            "input": self.input,
            "output": self.output,
            "metadata": self.metadata,
            "metrics": self.metrics,
            "scores": self.scores,
            "duration_ms": self.duration_ms,
        }


class Tracer:
    """Uniform tracing API; Langfuse-backed when enabled, in-memory otherwise."""

    def __init__(self, enabled: bool | None = None) -> None:
        self.enabled = settings.observability.langfuse_enabled if enabled is None else enabled
        self.spans: list[Span] = []
        self._client = self._connect() if self.enabled else None

    def _connect(self) -> Any:  # pragma: no cover - needs credentials
        try:
            from langfuse import Langfuse
        except ImportError:
            logger.warning("langfuse_not_installed", extra={"hint": "pip install '.[obs]'"})
            return None
        if not (settings.observability.public_key and settings.observability.secret_key):
            logger.warning("langfuse_keys_missing")
            return None
        try:
            return Langfuse(
                public_key=settings.observability.public_key,
                secret_key=settings.observability.secret_key,
                host=settings.observability.host,
            )
        except Exception as exc:
            logger.warning("langfuse_connect_failed", extra={"error": str(exc)})
            return None

    @property
    def active(self) -> bool:
        return self._client is not None

    @contextmanager
    def trace(self, name: str, **fields: Any):
        span = Span(name=name, trace_id=f"trace_{uuid.uuid4().hex[:16]}", **fields)
        if self._client is not None:  # pragma: no cover
            try:
                span._handle = self._client.trace(
                    name=name, input=span.input, metadata=span.metadata
                )
                span.trace_id = getattr(span._handle, "id", span.trace_id)
            except Exception as exc:
                logger.warning("langfuse_trace_failed", extra={"error": str(exc)})
        try:
            yield span
        finally:
            span.ended_at = time.time()
            self.spans.append(span)
            logger.info("trace", extra={"trace": span.as_dict()})
            if self._client is not None:  # pragma: no cover
                with contextlib.suppress(Exception):
                    self._client.flush()

    def last(self) -> Span | None:
        return self.spans[-1] if self.spans else None

    def find(self, trace_id: str) -> Span | None:
        return next((s for s in self.spans if s.trace_id == trace_id), None)

    def clear(self) -> None:
        self.spans.clear()


_TRACER: Tracer | None = None


def get_tracer(refresh: bool = False) -> Tracer:
    global _TRACER
    if _TRACER is None or refresh:
        _TRACER = Tracer()
    return _TRACER
