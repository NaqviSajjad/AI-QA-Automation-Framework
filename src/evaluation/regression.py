"""Prompt / metric regression gate.

The rule is simple and it is the reason this framework exists:

    baseline_score - current_score > allowed_drop   ->   FAIL

A prompt edit that quietly costs three points of faithfulness is invisible to a
traditional test suite. Here it fails the build, with the metric named.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.utils.config import settings
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Metrics where a *higher* number is worse.
LOWER_IS_BETTER = {"hallucination", "unauthorized_tool_calls", "latency_p95_ms"}


@dataclass
class MetricDelta:
    metric: str
    baseline: float
    current: float
    allowed_drop: float

    @property
    def difference(self) -> float:
        return round(self.current - self.baseline, 4)

    @property
    def regressed(self) -> bool:
        if self.metric in LOWER_IS_BETTER:
            return (self.current - self.baseline) > self.allowed_drop
        return (self.baseline - self.current) > self.allowed_drop

    @property
    def status(self) -> str:
        if self.regressed:
            return "REGRESSION"
        return "IMPROVED" if abs(self.difference) > 1e-9 and not self.regressed and (
            (self.difference > 0) != (self.metric in LOWER_IS_BETTER)
        ) else "STABLE"

    def as_row(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "baseline": self.baseline,
            "current": self.current,
            "difference": self.difference,
            "status": self.status,
        }


@dataclass
class RegressionReport:
    deltas: list[MetricDelta] = field(default_factory=list)
    missing_baseline: list[str] = field(default_factory=list)
    baseline_found: bool = True

    @property
    def passed(self) -> bool:
        return not any(delta.regressed for delta in self.deltas)

    @property
    def regressions(self) -> list[MetricDelta]:
        return [d for d in self.deltas if d.regressed]

    def table(self) -> str:
        header = f"{'Metric':<24}{'Baseline':>10}{'Current':>10}{'Diff':>10}  Status"
        lines = [header, "-" * len(header)]
        for delta in self.deltas:
            lines.append(
                f"{delta.metric:<24}{delta.baseline:>10.4f}{delta.current:>10.4f}"
                f"{delta.difference:>+10.4f}  {delta.status}"
            )
        if self.missing_baseline:
            lines.append(f"(no baseline for: {', '.join(self.missing_baseline)})")
        return "\n".join(lines)

    def as_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "baseline_found": self.baseline_found,
            "rows": [d.as_row() for d in self.deltas],
            "missing_baseline": self.missing_baseline,
        }


def baseline_path() -> Path:
    configured = settings.evaluation.get("regression", {}).get(
        "baseline_file", "data/baselines/baseline.json"
    )
    return settings.project_root / configured


def load_baseline(path: Path | None = None) -> dict[str, Any]:
    target = path or baseline_path()
    if not target.exists():
        return {}
    return json.loads(target.read_text(encoding="utf-8"))


def save_baseline(metrics: dict[str, float], path: Path | None = None, **meta: Any) -> Path:
    target = path or baseline_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "prompt_version": settings.prompts.active_version,
        "llm_provider": settings.llm.provider,
        "evaluator_backend": settings.evaluation.get("evaluator", {}).get("backend", "auto"),
        **meta,
        "metrics": {k: round(float(v), 4) for k, v in metrics.items()},
    }
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("baseline_saved", extra={"path": str(target), "metrics": len(metrics)})
    return target


def compare_to_baseline(
    current: dict[str, float],
    baseline: dict[str, Any] | None = None,
    allowed_drop: float | None = None,
) -> RegressionReport:
    raw = baseline if baseline is not None else load_baseline()
    baseline_metrics = raw.get("metrics", raw) if isinstance(raw, dict) else {}
    drop = (
        allowed_drop
        if allowed_drop is not None
        else float(settings.evaluation.get("regression", {}).get("allowed_drop", 0.03))
    )

    report = RegressionReport(baseline_found=bool(baseline_metrics))
    for metric, value in sorted(current.items()):
        if metric not in baseline_metrics:
            report.missing_baseline.append(metric)
            continue
        report.deltas.append(
            MetricDelta(
                metric=metric,
                baseline=round(float(baseline_metrics[metric]), 4),
                current=round(float(value), 4),
                allowed_drop=drop,
            )
        )
    return report


def compare_prompt_versions(
    scores_by_version: dict[str, dict[str, float]], allowed_drop: float | None = None
) -> dict[str, Any]:
    """A/B two prompt versions over the same dataset (used by tests/prompts)."""
    versions = list(scores_by_version)
    if len(versions) != 2:
        raise ValueError("compare_prompt_versions expects exactly two versions")
    left, right = versions
    drop = (
        allowed_drop
        if allowed_drop is not None
        else float(settings.evaluation.get("regression", {}).get("allowed_drop", 0.03))
    )
    rows = []
    for metric in sorted(set(scores_by_version[left]) | set(scores_by_version[right])):
        a = float(scores_by_version[left].get(metric, 0.0))
        b = float(scores_by_version[right].get(metric, 0.0))
        delta = MetricDelta(metric=metric, baseline=a, current=b, allowed_drop=drop)
        rows.append({**delta.as_row(), "baseline_version": left, "current_version": right})
    return {
        "versions": [left, right],
        "rows": rows,
        "regressions": [r for r in rows if r["status"] == "REGRESSION"],
        "winner": right if sum(r["difference"] for r in rows) > 0 else left,
    }
