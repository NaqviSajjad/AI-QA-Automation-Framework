"""Render the AI quality report.

Reads only files produced by an actual run - `reports/evaluation_run.json`,
`reports/quality_gate.json` and the pytest JUnit XML when present. Nothing is
invented: a section with no data says so rather than printing a plausible
number, because a report that might be fabricated is worse than no report.

    python -m scripts.generate_report
"""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from src.utils.config import settings

WIDTH = 56
MISSING = "not run"


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _read_junit(path: Path) -> dict[str, dict[str, int]] | None:
    """Per-suite pass/fail counts from the JUnit XML pytest writes with --junitxml."""
    if not path.exists():
        return None
    root = ET.parse(path).getroot()
    suites: dict[str, dict[str, int]] = {}
    for case in root.iter("testcase"):
        classname = case.get("classname", "")
        suite = classname.split(".")[1] if classname.startswith("tests.") else "other"
        bucket = suites.setdefault(suite, {"passed": 0, "failed": 0, "skipped": 0})
        if case.find("failure") is not None or case.find("error") is not None:
            bucket["failed"] += 1
        elif case.find("skipped") is not None:
            bucket["skipped"] += 1
        else:
            bucket["passed"] += 1
    return suites


def _heading(title: str) -> list[str]:
    return [title.upper(), "-" * len(title)]


def _metric(label: str, value: Any, digits: int = 4) -> str:
    if value is None:
        return f"{label:<22} {MISSING}"
    if isinstance(value, (int, float)):
        return f"{label:<22} {value:.{digits}f}"
    return f"{label:<22} {value}"


def build_report(
    run: dict[str, Any] | None,
    gate: dict[str, Any] | None,
    suites: dict[str, dict[str, int]] | None,
) -> str:
    lines: list[str] = ["=" * WIDTH, "AI QUALITY ENGINEERING REPORT", "=" * WIDTH, ""]

    if run is None and gate is None and suites is None:
        lines += [
            "No results found.",
            "",
            "Run the suite first:",
            "  pytest --junitxml=reports/junit.xml",
            "  python -m scripts.run_evaluation",
            "  python -m scripts.quality_gate",
            "=" * WIDTH,
        ]
        return "\n".join(lines)

    if run:
        lines += [
            f"provider          {run.get('llm_provider')}",
            f"prompt version    {run.get('prompt_version')}",
            f"evaluator backend {run.get('evaluator_backend')}",
            f"dataset cases     {run.get('cases')}",
            "",
        ]

    # -- test suites -------------------------------------------------------- #
    lines += _heading("Test suites")
    if suites:
        for name in sorted(suites):
            counts = suites[name]
            lines.append(
                f"{name:<14} passed: {counts['passed']:<4} "
                f"failed: {counts['failed']:<4} skipped: {counts['skipped']}"
            )
    else:
        lines.append(f"{MISSING} (pass --junitxml=reports/junit.xml to pytest)")
    lines.append("")

    metrics = (run or {}).get("metrics", {})

    # -- LLM evaluation ----------------------------------------------------- #
    lines += _heading("LLM evaluation")
    for label, key in (
        ("Faithfulness", "faithfulness"),
        ("Answer relevancy", "answer_relevancy"),
        ("Correctness", "correctness"),
        ("Hallucination", "hallucination"),
    ):
        lines.append(_metric(label, metrics.get(key)))
    lines.append("")

    # -- RAG ---------------------------------------------------------------- #
    lines += _heading("RAG")
    for label, key in (
        ("Context precision", "contextual_precision"),
        ("Context recall", "contextual_recall"),
        ("Context relevancy", "contextual_relevancy"),
    ):
        lines.append(_metric(label, metrics.get(key)))
    lines.append("")

    # -- agent -------------------------------------------------------------- #
    lines += _heading("Agent")
    lines.append(_metric("Task completion", metrics.get("task_completion")))
    lines.append(_metric("Tool correctness", metrics.get("tool_correctness")))
    lines.append(_metric("Unauthorised tools", (run or {}).get("unauthorized_tool_calls"), 0))
    lines.append("")

    # -- security ----------------------------------------------------------- #
    lines += _heading("Security")
    security = (run or {}).get("security")
    if security:
        verdict = "PASS" if security["critical_failures"] == 0 else "FAIL"
        lines.append(f"{'Adversarial cases':<22} {security['checks']}")
        lines.append(f"{'Failures':<22} {security['failures']}")
        lines.append(f"{'Critical failures':<22} {security['critical_failures']}  {verdict}")
        for failure in security.get("detail", []):
            lines.append(f"  - {failure['id']} ({failure['severity']}): {failure['problems']}")
    else:
        lines.append(MISSING)
    robustness = (run or {}).get("prompt_robustness")
    if robustness:
        lines.append(_metric("Prompt robustness", robustness.get("pass_rate")))
    lines.append("")

    # -- performance -------------------------------------------------------- #
    lines += _heading("Performance")
    performance = (run or {}).get("performance")
    if performance:
        lines.append(f"{'Average latency':<22} {performance['avg_latency_ms']:.1f} ms")
        lines.append(f"{'P95 latency':<22} {performance['p95_latency_ms']:.1f} ms")
        lines.append(f"{'Max LLM calls':<22} {performance['max_llm_calls']}")
        lines.append(f"{'Max tool calls':<22} {performance['max_tool_calls']}")
    else:
        lines.append(MISSING)
    lines.append("")

    # -- regression --------------------------------------------------------- #
    lines += _heading("Baseline comparison")
    if gate and gate.get("checks") is not None:
        lines.append("regression: " + ("PASS" if gate.get("regression_passed") else "FAIL"))
    else:
        lines.append(MISSING)
    lines.append("")

    # -- verdict ------------------------------------------------------------ #
    lines.append("=" * WIDTH)
    if gate is None:
        lines.append("QUALITY GATE: not evaluated")
    else:
        lines.append(f"QUALITY GATE: {'PASS' if gate.get('passed') else 'FAIL'}")
        for check in gate.get("checks", []):
            if not check["passed"]:
                lines.append(
                    f"  - {check['name']}: {check['actual']:.4f} "
                    f"violates {check['comparison']} {check['limit']}"
                )
    lines.append("=" * WIDTH)
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Render the AI quality report")
    parser.add_argument("--out", default="reports/ai-quality-report.txt")
    args = parser.parse_args()

    reports = settings.reports_path
    text = build_report(
        run=_read_json(reports / "evaluation_run.json"),
        gate=_read_json(reports / "quality_gate.json"),
        suites=_read_junit(reports / "junit.xml"),
    )
    print(text)
    out = settings.project_root / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
