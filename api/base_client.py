"""Shared HTTP client.

Records latency for every call so the performance suite measures the same
requests the functional suite makes, rather than a separate synthetic load.
"""

from __future__ import annotations

from typing import Any, TypeVar

import httpx

from src.evaluation.collector import record_event
from src.utils.config import settings
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Lets `with ChatAPI(...) as client` keep the subclass type rather than widening
# to BaseAPIClient - otherwise every fixture loses its API surface.
ClientT = TypeVar("ClientT", bound="BaseAPIClient")


class APIResponse:
    """A thin wrapper that keeps status, body, headers and latency together."""

    def __init__(self, response: httpx.Response, latency_ms: float) -> None:
        self._response = response
        self.latency_ms = latency_ms

    @property
    def status_code(self) -> int:
        return self._response.status_code

    @property
    def headers(self) -> httpx.Headers:
        return self._response.headers

    @property
    def text(self) -> str:
        return self._response.text

    def json(self) -> Any:
        try:
            return self._response.json()
        except ValueError:
            return {}

    @property
    def ok(self) -> bool:
        return self._response.is_success

    def __repr__(self) -> str:  # pragma: no cover
        return f"<APIResponse {self.status_code} in {self.latency_ms:.0f}ms>"


class BaseAPIClient:
    def __init__(self, base_url: str | None = None, token: str | None = None, timeout: float = 30.0):
        self.base_url = (base_url or settings.app.api_base_url).rstrip("/")
        self.token = token
        self._client = httpx.Client(timeout=timeout, follow_redirects=False)

    def close(self) -> None:
        self._client.close()

    def __enter__(self: ClientT) -> ClientT:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def _headers(self, extra: dict[str, str] | None = None) -> dict[str, str]:
        headers = {"Accept": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        headers.update(extra or {})
        return headers

    def request(
        self, method: str, path: str, *, headers: dict[str, str] | None = None, **kwargs: Any
    ) -> APIResponse:
        url = f"{self.base_url}{path}"
        response = self._client.request(method, url, headers=self._headers(headers), **kwargs)
        latency_ms = response.elapsed.total_seconds() * 1000
        record_event("latency", latency_ms=latency_ms, endpoint=path, method=method)
        logger.info(
            "api_call",
            extra={"method": method, "path": path, "status": response.status_code,
                   "latency_ms": round(latency_ms, 2)},
        )
        return APIResponse(response, latency_ms)

    def get(self, path: str, **kwargs: Any) -> APIResponse:
        return self.request("GET", path, **kwargs)

    def post(self, path: str, **kwargs: Any) -> APIResponse:
        return self.request("POST", path, **kwargs)

    def delete(self, path: str, **kwargs: Any) -> APIResponse:
        return self.request("DELETE", path, **kwargs)
