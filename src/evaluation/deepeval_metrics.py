"""DeepEval adapter.

DeepEval is an *optional* dependency and every one of its metrics needs a real
judge model. This module therefore reports its own availability and returns the
framework's common `MetricScore`, so a test can ask for DeepEval and transparently
fall back to the deterministic scorer when the extra or the key is missing.

Installed with:  pip install '.[eval]'
Enabled with:    EVALUATOR_BACKEND=deepeval  +  OPENAI_API_KEY
"""

from __future__ import annotations

from typing import Any

from src.evaluation.judge import MetricScore
from src.utils.config import settings
from src.utils.logging import get_logger

logger = get_logger(__name__)

try:  # pragma: no cover - optional dependency
    from deepeval.metrics import (
        AnswerRelevancyMetric,
        ContextualPrecisionMetric,
        ContextualRecallMetric,
        ContextualRelevancyMetric,
        FaithfulnessMetric,
        GEval,
    )
    from deepeval.test_case import LLMTestCase, LLMTestCaseParams

    DEEPEVAL_INSTALLED = True
except ImportError:  # pragma: no cover
    DEEPEVAL_INSTALLED = False


def is_available() -> bool:
    """DeepEval only works with an installed package *and* a judge model."""
    return DEEPEVAL_INSTALLED and bool(settings.llm.api_key)


def unavailable_reason() -> str:
    if not DEEPEVAL_INSTALLED:
        return "deepeval is not installed (pip install '.[eval]')"
    if not settings.llm.api_key:
        return "no judge-model credentials (OPENAI_API_KEY is unset)"
    return ""


def _judge_model() -> str:
    return settings.evaluation.get("evaluator", {}).get("judge_model", "gpt-4o-mini")


def _run(metric: Any, test_case: Any, name: str, threshold: float) -> MetricScore:
    metric.measure(test_case)
    return MetricScore(
        name=name,
        score=float(metric.score or 0.0),
        threshold=threshold,
        reason=str(getattr(metric, "reason", "") or ""),
        details={"backend": "deepeval", "model": _judge_model()},
    )


def answer_relevancy(question: str, answer: str, threshold: float = 0.85) -> MetricScore:
    case = LLMTestCase(input=question, actual_output=answer)
    return _run(
        AnswerRelevancyMetric(threshold=threshold, model=_judge_model(), include_reason=True),
        case, "answer_relevancy", threshold,
    )


def faithfulness(answer: str, contexts: list[str], question: str = "", threshold: float = 0.85) -> MetricScore:
    case = LLMTestCase(input=question or "n/a", actual_output=answer, retrieval_context=contexts)
    return _run(
        FaithfulnessMetric(threshold=threshold, model=_judge_model(), include_reason=True),
        case, "faithfulness", threshold,
    )


def hallucination(answer: str, contexts: list[str], question: str = "", max_score: float = 0.20) -> MetricScore:
    """DeepEval has a HallucinationMetric, but it scores *against* a ground-truth
    corpus. For a RAG answer the meaningful measure is 1 - faithfulness."""
    faith = faithfulness(answer, contexts, question=question, threshold=1 - max_score)
    return MetricScore(
        "hallucination", round(1.0 - faith.score, 4), max_score, faith.reason,
        {"backend": "deepeval"}, lower_is_better=True,
    )


def correctness(question: str, answer: str, reference: str, threshold: float = 0.80) -> MetricScore:
    """G-Eval: an LLM grades the answer against the expected behaviour.

    Model-based and therefore probabilistic - see docs/ai-evaluation.md for how
    this is used (nightly, with a deterministic gate underneath it).
    """
    metric = GEval(
        name="Correctness",
        criteria=(
            "Determine whether the actual output conveys the same customer-support "
            "outcome as the expected behaviour: correct policy, correct next step, "
            "no invented facts. Wording may differ."
        ),
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
        ],
        threshold=threshold,
        model=_judge_model(),
    )
    case = LLMTestCase(input=question, actual_output=answer, expected_output=reference)
    return _run(metric, case, "correctness", threshold)


def contextual_relevancy(question: str, contexts: list[str], answer: str = "", threshold: float = 0.75) -> MetricScore:
    case = LLMTestCase(input=question, actual_output=answer or "n/a", retrieval_context=contexts)
    return _run(
        ContextualRelevancyMetric(threshold=threshold, model=_judge_model()),
        case, "contextual_relevancy", threshold,
    )


def contextual_precision(question: str, contexts: list[str], reference: str, answer: str = "", threshold: float = 0.80) -> MetricScore:
    case = LLMTestCase(
        input=question, actual_output=answer or "n/a",
        expected_output=reference, retrieval_context=contexts,
    )
    return _run(
        ContextualPrecisionMetric(threshold=threshold, model=_judge_model()),
        case, "contextual_precision", threshold,
    )


def contextual_recall(question: str, contexts: list[str], reference: str, answer: str = "", threshold: float = 0.80) -> MetricScore:
    case = LLMTestCase(
        input=question, actual_output=answer or "n/a",
        expected_output=reference, retrieval_context=contexts,
    )
    return _run(
        ContextualRecallMetric(threshold=threshold, model=_judge_model()),
        case, "contextual_recall", threshold,
    )
