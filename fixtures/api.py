"""API client fixtures."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from api.chat_api import ChatAPI
from api.conversation_api import ConversationAPI
from api.escalation_api import EscalationAPI


@pytest.fixture
def api_base_url(app_server: str) -> str:
    return f"{app_server}/api"


@pytest.fixture
def anonymous_api(api_base_url: str) -> Iterator[ChatAPI]:
    """An unauthenticated client - used to prove endpoints are actually protected."""
    with ChatAPI(base_url=api_base_url) as client:
        yield client


@pytest.fixture
def chat_api(api_base_url: str) -> Iterator[ChatAPI]:
    with ChatAPI(base_url=api_base_url) as client:
        response = client.login()
        assert response.ok, f"Test user could not sign in: {response.status_code} {response.text}"
        yield client


@pytest.fixture
def conversation_api(chat_api: ChatAPI, api_base_url: str) -> Iterator[ConversationAPI]:
    with ConversationAPI(base_url=api_base_url, token=chat_api.token) as client:
        yield client


@pytest.fixture
def escalation_api(chat_api: ChatAPI, api_base_url: str) -> Iterator[EscalationAPI]:
    with EscalationAPI(base_url=api_base_url, token=chat_api.token) as client:
        yield client
