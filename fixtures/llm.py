"""LLM and agent fixtures."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from src.agents.graph import SupportAgent
from src.llm.base import LLMProvider
from src.llm.factory import get_llm_provider
from src.llm.mock_provider import MockLLMProvider
from src.utils.config import settings


@pytest.fixture(scope="session")
def llm_provider() -> LLMProvider:
    """Whichever provider the environment selects (mock by default)."""
    return get_llm_provider()


@pytest.fixture
def mock_llm() -> MockLLMProvider:
    """Always the mock, regardless of LLM_PROVIDER.

    Deterministic tests - schema validation, fault injection, routing - must not
    change behaviour just because someone exported an API key.
    """
    return MockLLMProvider()


@pytest.fixture(scope="session")
def support_agent() -> SupportAgent:
    return SupportAgent()


@pytest.fixture
def fresh_agent() -> Iterator[SupportAgent]:
    """A per-test agent, for cases that need to count LLM or tool calls."""
    agent = SupportAgent()
    yield agent
    agent.llm.reset()


@pytest.fixture
def uses_real_llm() -> bool:
    return settings.llm.provider != "mock"


@pytest.fixture
def requires_mock_provider(uses_real_llm: bool) -> None:
    if uses_real_llm:
        pytest.skip("deterministic assertion: only valid against the mock provider")
