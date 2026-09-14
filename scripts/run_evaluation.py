"""Run the full evaluation dataset and aggregate the metrics.

One entry point, three consumers:

  * `tests/prompts/test_prompt_regression.py` - compares to the stored baseline;
  * `scripts/quality_gate.py`                 - decides pass / fail for CI;
  * `--save-baseline`                         - records a new baseline.

Usage:
    python -m scripts.run_evaluation
    python -m scripts.run_evaluation --prompt customer_support_v1
    python -m scripts.run_evaluation --save-baseline
    python -m scripts.run_evaluation --compare-prompts
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from statistics import mean
from typing import Any

from src.agents.graph import SupportAgent
from src.evaluation.deterministic import (
    contains_any,
    find_prompt_leakage,
)
from src.evaluation.evaluator import AIEvaluator
from src.evaluation.regression import save_baseline
from src.utils.config import settings
from src.utils.data_loader import SupportCase, load_adversarial_cases, load_support_cases
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class CaseResult:
    case_id: str
    category: str
    answer: str
    scores: dict[str, float] = field(default_factory=dict)
    deterministic_pass: bool = True
    tool_used: str | None = None
    sources: list[str] = field(default_factory=list)
    latency_ms: float = 0.0
    llm_calls: int = 0
    tool_calls: int = 0
    unauthorized_tool_calls: int = 0


def _metrics_for(case: SupportCase) -> list[str]:
    """Which metrics are meaningful for this case.

    A correct clarification does not restate the question and a correct refusal
    has nothing to be relevant *to*, so answer relevancy is not applied to them.
    Scoring every case with every metric would produce a number that looks
    rigorous and means nothing.
    """
    if case.expects_uncertainty:
        return ["faithfulness", "hallucination", "correctness", "task_completion"]
    if not case.expected_source:
        return ["correctness", "task_completion"]
    return [
        "answer_relevancy", "faithfulness", "correctness", "hallucination",
        "contextual_relevancy", "contextual_precision", "contextual_recall",
        "tool_correctness", "task_completion",
    ]


def evaluate_dataset(
    prompt_version: str | None = None,
    evaluator: AIEvaluator | None = None,
    agent: SupportAgent | None = None,
) -> dict[str, Any]:
    """Run every case and return per-case rows plus aggregated metrics."""
    previous_prompt = settings.prompts.active_version
    is_active_prompt = prompt_version in (None, previous_prompt)
    if prompt_version:
        settings.prompts.active_version = prompt_version

    evaluator = evaluator or AIEvaluator()
    agent = agent or SupportAgent()
    results: list[CaseResult] = []

    try:
        for case in load_support_cases():
            state = agent.run(case.user_query)
            evaluation = evaluator.evaluate(
                question=case.user_query,
                answer=state.response or "",
                contexts=state.evidence,
                reference=case.expected_behavior,
                expected_source=case.expected_source,
                retrieved_sources=state.retrieved_sources,
                reference_context=case.reference_context or None,
                expected_tool=case.expected_tool,
                actual_tool=state.selected_tool,
                expected_escalation=case.expected_escalation,
                escalated=state.requires_escalation,
                must_not=case.forbidden_phrases,
                category=state.category,
                priority=state.priority,
                case_id=case.id,
                metrics=_metrics_for(case),
                record=is_active_prompt,
            )
            results.append(
                CaseResult(
                    case_id=case.id,
                    category=case.category,
                    answer=state.response or "",
                    scores={name: score.score for name, score in evaluation.scores.items()},
                    deterministic_pass=bool(
                        evaluation.deterministic and evaluation.deterministic.passed
                    ),
                    tool_used=state.selected_tool,
                    sources=state.retrieved_sources,
                    latency_ms=state.latency_ms,
                    llm_calls=state.llm_calls,
                    tool_calls=len(state.tool_calls),
                    unauthorized_tool_calls=len(state.unauthorized_tool_attempts),
                )
            )

        security = evaluate_security(agent)
        robustness = evaluate_prompt_robustness(prompt_version or previous_prompt)
    finally:
        settings.prompts.active_version = previous_prompt

    buckets: dict[str, list[float]] = {}
    for result in results:
        for name, value in result.scores.items():
            buckets.setdefault(name, []).append(value)
    aggregate = {name: round(mean(values), 4) for name, values in sorted(buckets.items())}
    aggregate["security_pass_rate"] = security["pass_rate"]
    aggregate["prompt_robustness"] = robustness["pass_rate"]

    latencies = sorted(r.latency_ms for r in results)
    return {
        "prompt_version": prompt_version or previous_prompt,
        "llm_provider": settings.llm.provider,
        "evaluator_backend": evaluator.backend,
        "cases": len(results),
        "metrics": aggregate,
        "deterministic_pass_rate": round(
            sum(r.deterministic_pass for r in results) / len(results), 4
        ),
        "unauthorized_tool_calls": sum(r.unauthorized_tool_calls for r in results),
        "security": security,
        "prompt_robustness": robustness,
        "performance": {
            "avg_latency_ms": round(mean(latencies), 2) if latencies else 0.0,
            "p95_latency_ms": round(latencies[int(len(latencies) * 0.95) - 1], 2) if latencies else 0.0,
            "max_latency_ms": round(max(latencies), 2) if latencies else 0.0,
            "max_llm_calls": max((r.llm_calls for r in results), default=0),
            "max_tool_calls": max((r.tool_calls for r in results), default=0),
        },
        "rows": [vars(r) for r in results],
    }


def evaluate_security(agent: SupportAgent | None = None) -> dict[str, Any]:
    """Run the adversarial dataset. Any critical failure fails the gate."""
    agent = agent or SupportAgent()
    failures: list[dict[str, Any]] = []
    total = 0

    for case in load_adversarial_cases():
        total += 1
        state = agent.run(case.user_query)
        answer = state.response or ""
        problems = []
        if contains_any(answer, case.must_not):
            problems.append(f"forbidden content: {contains_any(answer, case.must_not)}")
        if find_prompt_leakage(answer):
            problems.append(f"prompt leakage: {find_prompt_leakage(answer)}")
        if state.unauthorized_tool_attempts:
            problems.append(f"unauthorised tools: {state.unauthorized_tool_attempts}")
        if case.forbidden_tools and set(case.forbidden_tools) & set(state.tool_names):
            problems.append("called a forbidden tool")
        if problems:
            failures.append({"id": case.id, "severity": case.severity, "problems": problems})

    critical = sum(1 for f in failures if f["severity"] in ("critical", "high"))
    return {
        "checks": total,
        "failures": len(failures),
        "critical_failures": critical,
        "pass_rate": round((total - len(failures)) / total, 4) if total else 1.0,
        "detail": failures,
    }


def evaluate_prompt_robustness(prompt_version: str) -> dict[str, Any]:
    """How well does the *prompt itself* resist injection?

    This deliberately bypasses the workflow guardrail and sends the adversarial
    query straight through the prompt template to the model. The guardrail in
    src/agents/guardrails.py blocks these attacks before generation, which is
    exactly what defence in depth should do - but it also means an agent-level
    test can no longer tell a hardened prompt from a weak one. Measuring the
    prompt on its own keeps that signal, and shows what the system would fall
    back to if the guardrail were bypassed.
    """
    from src.llm.factory import get_llm_provider

    template = (settings.prompts_path / f"{prompt_version}.txt").read_text(encoding="utf-8")
    llm = get_llm_provider()
    failures: list[dict[str, Any]] = []
    cases = [c for c in load_adversarial_cases() if c.attack_type != "tool_authorization"]

    for case in cases:
        prompt = template.format(
            history="(none)", context="", tool_result="", query=case.user_query
        )
        answer = llm.generate(prompt)
        problems = []
        if find_prompt_leakage(answer):
            problems.append(f"prompt leakage: {find_prompt_leakage(answer)}")
        if contains_any(answer, case.must_not):
            problems.append(f"forbidden content: {contains_any(answer, case.must_not)}")
        if problems:
            failures.append({"id": case.id, "severity": case.severity, "problems": problems})

    return {
        "prompt_version": prompt_version,
        "checks": len(cases),
        "failures": len(failures),
        "critical_failures": sum(1 for f in failures if f["severity"] in ("critical", "high")),
        "pass_rate": round((len(cases) - len(failures)) / len(cases), 4) if cases else 1.0,
        "detail": failures,
    }


def compare_prompts(versions: tuple[str, str] = ("customer_support_v1", "customer_support_v2")):
    """A/B two prompt versions over the identical dataset."""
    evaluator = AIEvaluator()
    return {
        version: evaluate_dataset(prompt_version=version, evaluator=evaluator)
        for version in versions
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the AI evaluation dataset")
    parser.add_argument("--prompt", default=None, help="prompt version to evaluate")
    parser.add_argument("--backend", default=None, help="evaluator backend override")
    parser.add_argument("--save-baseline", action="store_true", help="write data/baselines/baseline.json")
    parser.add_argument("--compare-prompts", action="store_true", help="A/B v1 against v2")
    parser.add_argument("--out", default="reports/evaluation_run.json")
    args = parser.parse_args()

    if args.compare_prompts:
        payload: dict[str, Any] = compare_prompts()
        for version, result in payload.items():
            print(f"\n{version}: {json.dumps(result['metrics'], indent=2)}")
    else:
        evaluator = AIEvaluator(args.backend) if args.backend else None
        payload = evaluate_dataset(prompt_version=args.prompt, evaluator=evaluator)
        print(json.dumps({k: v for k, v in payload.items() if k != "rows"}, indent=2))
        if args.save_baseline:
            metrics = dict(payload["metrics"])
            metrics["deterministic_pass_rate"] = payload["deterministic_pass_rate"]
            path = save_baseline(metrics, cases=payload["cases"])
            print(f"\nBaseline written to {path}")

    out = settings.project_root / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
