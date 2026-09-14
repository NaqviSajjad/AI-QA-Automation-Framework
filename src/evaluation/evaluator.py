"""The evaluation facade the tests actually call.

One object, one `evaluate()` call, three backends behind it:

    EVALUATOR_BACKEND=builtin   deterministic scorers (default, CI-safe)
    EVALUATOR_BACKEND=deepeval  LLM-judged metrics
    EVALUATOR_BACKEND=ragas     RAG-specialised metrics
    EVALUATOR_BACKEND=auto      the best backend that is actually usable

`auto` is what makes the framework portable: the same test file gates a PR with
deterministic scores and produces model-judged scores in the nightly run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.evaluation import deepeval_metrics, judge, ragas_metrics
from src.evaluation.collector import record_evaluation
from src.evaluation.deterministic import DeterministicReport, evaluate_response_deterministically
from src.evaluation.judge import MetricScore
from src.utils.config import settings
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class EvaluationResult:
    """Metric scores plus the deterministic report, in one object.

    Attribute access (`result.faithfulness`) returns the float so assertions read
    the way the spec's flagship test does; `result.metric("faithfulness")` returns
    the full MetricScore when the reason string is needed in a failure message.
    """

    question: str
    answer: str
    backend: str
    scores: dict[str, MetricScore] = field(default_factory=dict)
    deterministic: DeterministicReport | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def metric(self, name: str) -> MetricScore:
        if name not in self.scores:
            raise KeyError(f"Metric '{name}' was not computed. Available: {sorted(self.scores)}")
        return self.scores[name]

    def __getattr__(self, item: str) -> float:  # pragma: no cover - trivial
        scores = self.__dict__.get("scores", {})
        if item in scores:
            return scores[item].score
        raise AttributeError(item)

    @property
    def passed(self) -> bool:
        deterministic_ok = self.deterministic.passed if self.deterministic else True
        return deterministic_ok and all(s.passed for s in self.scores.values())

    @property
    def failures(self) -> list[str]:
        failed = [
            f"{s.name}={s.score:.3f} < {s.threshold} ({s.reason})"
            for s in self.scores.values()
            if not s.passed
        ]
        if self.deterministic and not self.deterministic.passed:
            failed.append(f"deterministic: {self.deterministic.summary()}")
        return failed

    def as_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "backend": self.backend,
            "passed": self.passed,
            "scores": {name: score.as_dict() for name, score in self.scores.items()},
            "deterministic_pass_rate": self.deterministic.pass_rate if self.deterministic else None,
            "metadata": self.metadata,
        }

    def explain(self) -> str:
        """A failure message an engineer can act on without opening a debugger."""
        if self.passed:
            return "all metrics passed"
        return " | ".join(self.failures)


class AIEvaluator:
    def __init__(self, backend: str | None = None) -> None:
        requested = (backend or settings.evaluation.get("evaluator", {}).get("backend", "auto")).lower()
        self.requested_backend = requested
        self.backend = self._resolve(requested)
        logger.info(
            "evaluator_ready", extra={"requested": requested, "resolved": self.backend}
        )

    @staticmethod
    def _resolve(requested: str) -> str:
        if requested == "deepeval":
            return "deepeval" if deepeval_metrics.is_available() else "builtin"
        if requested == "ragas":
            return "ragas" if ragas_metrics.is_available() else "builtin"
        if requested == "auto":
            if deepeval_metrics.is_available():
                return "deepeval"
            if ragas_metrics.is_available():
                return "ragas"
        return "builtin"

    def backend_note(self) -> str:
        if self.backend != "builtin":
            return f"model-based backend: {self.backend}"
        reason = deepeval_metrics.unavailable_reason() or "builtin backend selected"
        return f"deterministic builtin scorers ({reason})"

    # -- main entry point --------------------------------------------------- #
    def evaluate(
        self,
        question: str,
        answer: str,
        contexts: list[str] | None = None,
        reference: str | None = None,
        *,
        expected_source: str | None = None,
        retrieved_sources: list[str] | None = None,
        reference_context: str | None = None,
        expected_tool: str | None = None,
        actual_tool: str | None = None,
        expected_escalation: bool = False,
        escalated: bool = False,
        must_not: list[str] | None = None,
        category: str | None = None,
        priority: str | None = None,
        case_id: str | None = None,
        metrics: list[str] | None = None,
        record: bool = True,
    ) -> EvaluationResult:
        contexts = contexts or []
        wanted = set(metrics or ["answer_relevancy", "faithfulness"])
        scores: dict[str, MetricScore] = {}

        def threshold(name: str, default: float) -> float:
            return settings.metric_threshold(name, default)

        use_deepeval = self.backend == "deepeval"

        if "answer_relevancy" in wanted:
            t = threshold("answer_relevancy", 0.85)
            scores["answer_relevancy"] = (
                deepeval_metrics.answer_relevancy(question, answer, t)
                if use_deepeval
                else judge.answer_relevancy(question, answer, t, topic=category)
            )
        if "faithfulness" in wanted:
            t = threshold("faithfulness", 0.85)
            scores["faithfulness"] = (
                deepeval_metrics.faithfulness(answer, contexts, question, t)
                if use_deepeval
                else judge.faithfulness(answer, contexts, t)
            )
        if "hallucination" in wanted:
            t = settings.evaluation.get("metrics", {}).get("hallucination", {}).get("max", 0.20)
            scores["hallucination"] = (
                deepeval_metrics.hallucination(answer, contexts, question, t)
                if use_deepeval
                else judge.hallucination(answer, contexts, t)
            )
        if "correctness" in wanted and reference:
            t = threshold("correctness", 0.80)
            scores["correctness"] = (
                deepeval_metrics.correctness(question, answer, reference, t)
                if use_deepeval
                else judge.correctness(answer, reference, t)
            )
        if "contextual_relevancy" in wanted:
            t = threshold("contextual_relevancy", 0.75)
            scores["contextual_relevancy"] = (
                deepeval_metrics.contextual_relevancy(question, contexts, answer, t)
                if use_deepeval
                else judge.contextual_relevancy(question, contexts, t)
            )
        if "contextual_precision" in wanted and expected_source is not None:
            scores["contextual_precision"] = judge.contextual_precision(
                retrieved_sources or [], expected_source, threshold("contextual_precision", 0.80)
            )
        if "contextual_recall" in wanted and reference_context:
            scores["contextual_recall"] = judge.contextual_recall(
                reference_context, contexts, threshold("contextual_recall", 0.80)
            )
        if "tool_correctness" in wanted:
            scores["tool_correctness"] = judge.tool_correctness(
                expected_tool, actual_tool, threshold("tool_correctness", 0.90)
            )
        if "task_completion" in wanted and reference:
            scores["task_completion"] = judge.task_completion(
                answer, reference,
                escalated=escalated, expected_escalation=expected_escalation,
                threshold=threshold("task_completion", 0.85),
            )

        deterministic = evaluate_response_deterministically(
            answer,
            category=category,
            priority=priority,
            must_not=must_not,
            expected_tool=expected_tool if actual_tool is not None else None,
            actual_tool=actual_tool,
        )

        result = EvaluationResult(
            question=question,
            answer=answer,
            backend=self.backend,
            scores=scores,
            deterministic=deterministic,
            metadata={"case_id": case_id, "category": category, "sources": retrieved_sources or []},
        )
        # A/B runs against a non-active prompt are excluded from the session
        # totals: mixing them in would make the published quality report describe
        # a prompt that is not shipping.
        if record:
            record_evaluation(result.as_dict())
        return result


_EVALUATOR: AIEvaluator | None = None


def get_evaluator(backend: str | None = None, refresh: bool = False) -> AIEvaluator:
    global _EVALUATOR
    if _EVALUATOR is None or refresh or (backend and backend != _EVALUATOR.requested_backend):
        _EVALUATOR = AIEvaluator(backend)
    return _EVALUATOR
