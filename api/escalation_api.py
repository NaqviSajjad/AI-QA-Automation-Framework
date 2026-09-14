"""Human-handoff endpoint."""

from __future__ import annotations

from api.base_client import APIResponse, BaseAPIClient


class EscalationAPI(BaseAPIClient):
    def escalate(self, conversation_id: str, reason: str) -> APIResponse:
        return self.post(
            "/escalation", json={"conversation_id": conversation_id, "reason": reason}
        )
