"""The CI quality gate.

Reads `config/evaluation.yaml`, runs the evaluation dataset, compares the result
to the stored baseline, and exits non-zero with reasons if anything falls short.

    python -m scripts.quality_gate
    python -m scripts.quality_gate --from-file reports/evaluation_run.json

Every threshold comes from configuration. Nothing here is hard-coded, so the bar
can be raised without touching code - and lowering it is a visible diff.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.evaluation.regression import compare_to_baseline
from src.utils.config import settings
from src.utils.logging import get_logger

logger = get_logger(__name__)

LOWER_IS_BETTER = {"hallucination", "unauthorized_tool_calls", "critical_security_failures"}


@dataclass
class GateCheck:
    name: str
    actual: float
    limit: float
    passed: bool
    comparison: str

    def render(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return f"  [{status}] {self.name:<28} {self.actual:>8.4f}  {self.comparison} {self.limit}"


@dataclass
class GateReport:
    checks: list[GateCheck] = field(default_factory=list)
    regression_table: str = ""
    regression_passed: bool = True
    metrics: dict[str, Any] = field(default_factory=dict)
    run: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks) and self.regression_passed

    @property
    def failures(self) -> list[GateCheck]:
        return [check for check in self.checks if not check.passed]

    def render(self) -> str:
        lines = [
            "=" * 60,
            "AI QUALITY GATE",
            "=" * 60,
            f"  provider={self.run.get('llm_provider')} "
            f"prompt={self.run.get('prompt_version')} "
            f"evaluator={self.run.get('evaluator_backend')} "
            f"cases={self.run.get('cases')}",
            "",
        ]
        lines += [check.render() for check in self.checks]
        if self.regression_table:
            lines += ["", "  Baseline comparison:", *[f"    {line}" for line in self.regression_table.splitlines()]]
        lines += ["", "=" * 60]
        lines.append(f"AI QUALITY GATE: {'PASS' if self.passed else 'FAIL'}")
        if not self.passed:
            for check in self.failures:
                lines.append(f"  - {check.name}: {check.actual:.4f} violates {check.comparison} {check.limit}")
            if not self.regression_passed:
                lines.append("  - metric regression against the baseline (see table above)")
        lines.append("=" * 60)
        return "\n".join(lines)

    def as_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "checks": [vars(check) for check in self.checks],
            "regression_passed": self.regression_passed,
            "metrics": self.metrics,
            "run": self.run,
        }


def _add(report: GateReport, name: str, actual: float | None, limit: float | None) -> None:
    if actual is None or limit is None:
        return
    lower_is_better = name in LOWER_IS_BETTER
    passed = actual <= limit if lower_is_better else actual >= limit
    report.checks.append(
        GateCheck(name, float(actual), float(limit), passed, "<=" if lower_is_better else ">=")
    )


def evaluate_gate(run: dict[str, Any] | None = None) -> GateReport:
    if run is None:
        from scripts.run_evaluation import evaluate_dataset

        run = evaluate_dataset()

    gate = settings.evaluation.get("quality_gate", {})
    metrics = dict(run["metrics"])
    metrics["deterministic_pass_rate"] = run["deterministic_pass_rate"]

    report = GateReport(metrics=metrics, run={k: run.get(k) for k in
                                              ("llm_provider", "prompt_version",
                                               "evaluator_backend", "cases")})

    _add(report, "deterministic_tests", run["deterministic_pass_rate"],
         gate.get("deterministic_tests", {}).get("minimum_pass_rate"))
    for metric in ("answer_relevancy", "faithfulness", "correctness",
                   "contextual_precision", "contextual_recall",
                   "task_completion", "tool_correctness"):
        _add(report, metric, metrics.get(metric), gate.get(metric, {}).get("minimum"))

    _add(report, "hallucination", metrics.get("hallucination"),
         gate.get("hallucination_rate", {}).get("maximum"))
    _add(report, "unauthorized_tool_calls", run.get("unauthorized_tool_calls", 0),
         gate.get("unauthorized_tool_calls", {}).get("maximum"))
    _add(report, "critical_security_failures", run["security"]["critical_failures"],
         gate.get("critical_security_failures", {}).get("maximum"))

    # Regression against the stored baseline.
    allowed_drop = gate.get("prompt_regression", {}).get("maximum_allowed_drop")
    regression = compare_to_baseline(metrics, allowed_drop=allowed_drop)
    fail_on_missing = settings.evaluation.get("regression", {}).get("fail_on_missing_baseline", False)
    report.regression_table = regression.table()
    report.regression_passed = regression.passed and (
        regression.baseline_found or not fail_on_missing
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the AI quality gate")
    parser.add_argument("--from-file", default=None, help="use an existing evaluation run")
    parser.add_argument("--out", default="reports/quality_gate.json")
    args = parser.parse_args()

    run = None
    if args.from_file:
        run = json.loads(Path(args.from_file).read_text(encoding="utf-8"))

    report = evaluate_gate(run)
    print(report.render())

    out = settings.project_root / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report.as_dict(), indent=2), encoding="utf-8")
    return 0 if report.passed else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
