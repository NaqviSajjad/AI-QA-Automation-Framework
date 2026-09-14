"""Conversation retrieval and reset endpoints."""

from __future__ import annotations

from api.base_client import APIResponse, BaseAPIClient


class ConversationAPI(BaseAPIClient):
    def get_conversation(self, conversation_id: str) -> APIResponse:
        return self.get(f"/conversations/{conversation_id}")

    def reset_conversation(self, conversation_id: str) -> APIResponse:
        return self.delete(f"/conversations/{conversation_id}")

    def message_count(self, conversation_id: str) -> int:
        response = self.get_conversation(conversation_id)
        return int(response.json().get("message_count", 0)) if response.ok else 0
