"""Chat and authentication endpoints."""

from __future__ import annotations

from typing import Any

from api.base_client import APIResponse, BaseAPIClient
from src.utils.config import settings


class ChatAPI(BaseAPIClient):
    def health(self) -> APIResponse:
        return self.get("/health")

    def login(self, email: str | None = None, password: str | None = None) -> APIResponse:
        response = self.post(
            "/auth/login",
            json={
                "email": email if email is not None else settings.auth.email,
                "password": password if password is not None else settings.auth.password,
            },
        )
        if response.ok:
            self.token = response.json().get("token")
        return response

    def logout(self) -> APIResponse:
        response = self.post("/auth/logout")
        self.token = None
        return response

    def send_message(
        self,
        message: str,
        conversation_id: str | None = None,
        account_id: str | None = None,
        **kwargs: Any,
    ) -> APIResponse:
        payload: dict[str, Any] = {"message": message}
        if conversation_id:
            payload["conversation_id"] = conversation_id
        if account_id:
            payload["account_id"] = account_id
        payload.update(kwargs)
        return self.post("/chat", json=payload)

    def send_raw(self, payload: Any) -> APIResponse:
        """Send an arbitrary body - used for malformed-request contract tests."""
        if isinstance(payload, (dict, list)):
            return self.post("/chat", json=payload)
        return self.post("/chat", content=payload, headers={"Content-Type": "application/json"})

    def feedback(self, conversation_id: str, rating: str, comment: str | None = None) -> APIResponse:
        body: dict[str, Any] = {"conversation_id": conversation_id, "rating": rating}
        if comment is not None:
            body["comment"] = comment
        return self.post("/feedback", json=body)
