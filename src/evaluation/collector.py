"""Session-wide collection of evaluation results.

Every `AIEvaluator.evaluate()` call is appended here, and the pytest session
hook flushes the lot to `reports/evaluation_results.json`. That file is the
single input to the quality gate, the AI quality report and the baseline
comparison - so those three always describe the *same* run.
"""

from __future__ import annotations

import json
import threading
from statistics import mean
from typing import Any

from src.utils.config import settings

_LOCK = threading.Lock()
_RESULTS: list[dict[str, Any]] = []
_EVENTS: list[dict[str, Any]] = []


def record_evaluation(result: dict[str, Any]) -> None:
    with _LOCK:
        _RESULTS.append(result)


def record_event(kind: str, **payload: Any) -> None:
    """Non-metric observations: latencies, tool calls, security verdicts."""
    with _LOCK:
        _EVENTS.append({"kind": kind, **payload})


def results() -> list[dict[str, Any]]:
    return list(_RESULTS)


def events(kind: str | None = None) -> list[dict[str, Any]]:
    return [e for e in _EVENTS if kind is None or e["kind"] == kind]


def clear() -> None:
    with _LOCK:
        _RESULTS.clear()
        _EVENTS.clear()


def aggregate() -> dict[str, float]:
    """Mean score per metric across everything evaluated this session."""
    buckets: dict[str, list[float]] = {}
    for result in _RESULTS:
        for name, score in result.get("scores", {}).items():
            buckets.setdefault(name, []).append(float(score["score"]))
    return {name: round(mean(values), 4) for name, values in sorted(buckets.items()) if values}


def summary() -> dict[str, Any]:
    latencies = [float(e["latency_ms"]) for e in events("latency") if "latency_ms" in e]
    latencies.sort()
    p95 = latencies[min(len(latencies) - 1, int(len(latencies) * 0.95))] if latencies else 0.0
    security = events("security")
    return {
        "evaluations": len(_RESULTS),
        "passed": sum(1 for r in _RESULTS if r.get("passed")),
        "metrics": aggregate(),
        "latency": {
            "samples": len(latencies),
            "avg_ms": round(mean(latencies), 2) if latencies else 0.0,
            "p95_ms": round(p95, 2),
            "max_ms": round(max(latencies), 2) if latencies else 0.0,
        },
        "tool_calls": {
            "total": sum(int(e.get("tool_calls", 0)) for e in events("agent")),
            "unauthorized": sum(int(e.get("unauthorized", 0)) for e in events("agent")),
        },
        "security": {
            "checks": len(security),
            "failures": sum(1 for e in security if not e.get("passed", True)),
            "critical_failures": sum(
                1 for e in security
                if not e.get("passed", True) and e.get("severity") in ("critical", "high")
            ),
        },
        "tokens": {
            "total": sum(int(e.get("total_tokens", 0)) for e in events("llm")),
        },
    }


def flush(filename: str = "evaluation_results.json") -> dict[str, Any]:
    payload = {"summary": summary(), "results": results(), "events": events()}
    path = settings.reports_path / filename
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return payload
