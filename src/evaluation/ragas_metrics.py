"""RAGAS adapter - RAG-specific evaluation.

RAGAS is the right tool for separating *retrieval* failures from *generation*
failures, which is the single most useful diagnostic in a RAG system:

    context_precision / context_recall  -> is the retriever doing its job?
    faithfulness / answer_relevancy     -> is the generator doing its job?

Optional dependency + judge model required, same contract as the DeepEval
adapter: `is_available()` first, then call.

Installed with:  pip install '.[eval]'
Enabled with:    EVALUATOR_BACKEND=ragas  +  OPENAI_API_KEY
"""

from __future__ import annotations

from typing import Any

from src.evaluation.judge import MetricScore
from src.utils.config import settings
from src.utils.logging import get_logger

logger = get_logger(__name__)

try:  # pragma: no cover - optional dependency
    from datasets import Dataset
    from ragas import evaluate as ragas_evaluate
    from ragas.metrics import (
        answer_relevancy as _answer_relevancy,
    )
    from ragas.metrics import (
        context_precision as _context_precision,
    )
    from ragas.metrics import (
        context_recall as _context_recall,
    )
    from ragas.metrics import (
        faithfulness as _faithfulness,
    )

    RAGAS_INSTALLED = True
except ImportError:  # pragma: no cover
    RAGAS_INSTALLED = False


METRIC_NAMES = {
    "faithfulness": "faithfulness",
    "answer_relevancy": "answer_relevancy",
    "context_precision": "contextual_precision",
    "context_recall": "contextual_recall",
}


def is_available() -> bool:
    return RAGAS_INSTALLED and bool(settings.llm.api_key)


def unavailable_reason() -> str:
    if not RAGAS_INSTALLED:
        return "ragas is not installed (pip install '.[eval]')"
    if not settings.llm.api_key:
        return "no judge-model credentials (OPENAI_API_KEY is unset)"
    return ""


def evaluate_rag(
    questions: list[str],
    answers: list[str],
    contexts: list[list[str]],
    ground_truths: list[str] | None = None,
) -> dict[str, MetricScore]:
    """Run the RAGAS suite over a batch and return framework MetricScores."""
    if not is_available():  # pragma: no cover
        raise RuntimeError(f"RAGAS unavailable: {unavailable_reason()}")

    payload: dict[str, Any] = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
    }
    metrics = [_faithfulness, _answer_relevancy]
    if ground_truths:
        payload["ground_truth"] = ground_truths
        metrics += [_context_precision, _context_recall]

    result = ragas_evaluate(Dataset.from_dict(payload), metrics=metrics)
    scores: dict[str, MetricScore] = {}
    for ragas_name, framework_name in METRIC_NAMES.items():
        if ragas_name not in result:
            continue
        value = float(result[ragas_name])
        scores[framework_name] = MetricScore(
            name=framework_name,
            score=round(value, 4),
            threshold=settings.metric_threshold(framework_name),
            reason=f"ragas batch of {len(questions)} samples",
            details={"backend": "ragas"},
        )
    return scores


def evaluate_single(
    question: str, answer: str, contexts: list[str], ground_truth: str | None = None
) -> dict[str, MetricScore]:
    return evaluate_rag([question], [answer], [contexts], [ground_truth] if ground_truth else None)
