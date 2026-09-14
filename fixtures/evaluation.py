"""Evaluation fixtures."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from src.evaluation.evaluator import AIEvaluator, get_evaluator
from src.observability.langfuse_client import Tracer, get_tracer
from src.utils.config import settings
from src.utils.data_loader import (
    AdversarialCase,
    SupportCase,
    load_adversarial_cases,
    load_support_cases,
)


@pytest.fixture(scope="session")
def ai_evaluator() -> AIEvaluator:
    """The evaluator the flagship tests use. Backend is resolved from config/env."""
    return get_evaluator()


@pytest.fixture(scope="session")
def builtin_evaluator() -> AIEvaluator:
    """Deterministic scorers, whatever the environment - for reproducible gates."""
    return AIEvaluator("builtin")


@pytest.fixture(scope="session")
def support_cases() -> list[SupportCase]:
    return load_support_cases()


@pytest.fixture(scope="session")
def adversarial_cases() -> list[AdversarialCase]:
    return load_adversarial_cases()


@pytest.fixture
def tracer() -> Iterator[Tracer]:
    """Per-test tracing, so a failure can be traced back to its own spans."""
    tracer = get_tracer()
    before = len(tracer.spans)
    yield tracer
    if not settings.observability.langfuse_enabled:
        del tracer.spans[:before]


@pytest.fixture
def thresholds() -> dict[str, float]:
    return {
        name: settings.metric_threshold(name)
        for name in (
            "answer_relevancy", "faithfulness", "correctness", "contextual_relevancy",
            "contextual_precision", "contextual_recall", "task_completion", "tool_correctness",
        )
    }
