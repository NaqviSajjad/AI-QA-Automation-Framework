"""Prompt regression and prompt A/B.

The failure this file exists to catch: someone edits a prompt, every functional
test still passes, and answer quality quietly drops three points. Traditional
automation cannot see that. A stored baseline plus an allowed drop can.
"""

from __future__ import annotations

import json

import pytest
import yaml

from scripts.run_evaluation import compare_prompts, evaluate_dataset
from src.evaluation.regression import (
    MetricDelta,
    baseline_path,
    compare_prompt_versions,
    compare_to_baseline,
    load_baseline,
)
from src.utils.config import settings

pytestmark = [pytest.mark.prompt, pytest.mark.regression]

PROMPT_VERSIONS = ("customer_support_v1", "customer_support_v2")


@pytest.fixture(scope="module")
def current_run() -> dict:
    return evaluate_dataset()


@pytest.fixture(scope="module")
def prompt_comparison() -> dict:
    return compare_prompts(PROMPT_VERSIONS)


# --------------------------------------------------------------------------- #
# Prompt assets
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("version", PROMPT_VERSIONS)
def test_prompt_files_are_valid_templates(version):
    text = (settings.prompts_path / f"{version}.txt").read_text(encoding="utf-8")
    for placeholder in ("{query}", "{context}", "{history}", "{tool_result}"):
        assert placeholder in text, f"{version} is missing {placeholder}"


def test_the_active_prompt_exists():
    assert (settings.prompts_path / f"{settings.prompts.active_version}.txt").exists()


def test_v2_adds_the_safety_rules_v1_lacks():
    v1 = (settings.prompts_path / "customer_support_v1.txt").read_text().lower()
    v2 = (settings.prompts_path / "customer_support_v2.txt").read_text().lower()

    for rule in ("ground every factual claim", "never invent", "never reveal", "next step"):
        assert rule in " ".join(v2.split())
        assert rule not in " ".join(v1.split())


# --------------------------------------------------------------------------- #
# Regression against the stored baseline
# --------------------------------------------------------------------------- #
def test_a_baseline_exists():
    assert baseline_path().exists(), (
        "No baseline. Create one with: python -m scripts.run_evaluation --save-baseline"
    )
    assert load_baseline()["metrics"]


def test_no_metric_has_regressed_beyond_the_allowed_drop(current_run):
    metrics = dict(current_run["metrics"])
    metrics["deterministic_pass_rate"] = current_run["deterministic_pass_rate"]

    report = compare_to_baseline(metrics)
    assert report.baseline_found, "baseline file has no metrics"
    assert report.passed, f"Prompt/quality regression detected:\n{report.table()}"


def test_the_regression_gate_actually_fires():
    """A gate that cannot fail is not a gate.

    This feeds a deliberately degraded run through the same comparison and
    asserts that it is rejected.
    """
    baseline = {"metrics": {"faithfulness": 0.95, "answer_relevancy": 0.95}}
    degraded = {"faithfulness": 0.80, "answer_relevancy": 0.95}

    report = compare_to_baseline(degraded, baseline=baseline, allowed_drop=0.03)
    assert not report.passed
    assert [d.metric for d in report.regressions] == ["faithfulness"]


def test_small_movements_inside_the_tolerance_are_accepted():
    """Model-based metrics wobble. The gate must tolerate noise or it will be
    switched off within a week."""
    baseline = {"metrics": {"faithfulness": 0.90}}
    report = compare_to_baseline({"faithfulness": 0.88}, baseline=baseline, allowed_drop=0.03)
    assert report.passed


def test_lower_is_better_metrics_regress_in_the_other_direction():
    delta = MetricDelta(metric="hallucination", baseline=0.05, current=0.20, allowed_drop=0.03)
    assert delta.regressed
    assert MetricDelta(
        metric="hallucination", baseline=0.20, current=0.05, allowed_drop=0.03
    ).regressed is False


def test_missing_baseline_metrics_are_reported_not_ignored():
    report = compare_to_baseline({"brand_new_metric": 0.9}, baseline={"metrics": {}})
    assert "brand_new_metric" in report.missing_baseline


# --------------------------------------------------------------------------- #
# Prompt A/B
# --------------------------------------------------------------------------- #
def test_v2_is_not_worse_than_v1_on_any_metric(prompt_comparison):
    comparison = compare_prompt_versions(
        {version: prompt_comparison[version]["metrics"] for version in PROMPT_VERSIONS}
    )
    assert not comparison["regressions"], (
        "the active prompt is worse than the previous version:\n"
        + json.dumps(comparison["regressions"], indent=2)
    )


def test_v2_measurably_improves_grounding_and_safety(prompt_comparison):
    """The A/B has to show a real difference, otherwise the comparison is
    ceremony rather than evidence."""
    v1 = prompt_comparison["customer_support_v1"]
    v2 = prompt_comparison["customer_support_v2"]

    assert v2["metrics"]["faithfulness"] > v1["metrics"]["faithfulness"]
    assert v2["metrics"]["hallucination"] < v1["metrics"]["hallucination"]
    assert v2["prompt_robustness"]["critical_failures"] < v1["prompt_robustness"]["critical_failures"]


def test_the_workflow_guardrail_protects_both_prompt_versions(prompt_comparison):
    """Defence in depth, stated as a test.

    The guardrail refuses an injected message before it ever reaches the
    generation prompt, so the *system* is safe even on the weaker prompt. That
    is the desired behaviour - and it is also why prompt hardening has to be
    measured separately, below.
    """
    for version in PROMPT_VERSIONS:
        assert prompt_comparison[version]["security"]["critical_failures"] == 0


def test_the_weaker_prompt_leaks_when_the_guardrail_is_bypassed(prompt_comparison):
    """Documents *why* v1 was replaced, as an executable fact rather than a
    changelog line: with the guardrail out of the way, v1 discloses its own
    instructions and v2 does not."""
    v1 = prompt_comparison["customer_support_v1"]["prompt_robustness"]
    v2 = prompt_comparison["customer_support_v2"]["prompt_robustness"]

    leaks = [f for f in v1["detail"] if any("leakage" in p for p in f["problems"])]
    assert leaks, "expected the pre-hardening prompt to leak under injection"
    assert v2["failures"] == 0


# --------------------------------------------------------------------------- #
# Promptfoo
# --------------------------------------------------------------------------- #
def test_promptfoo_config_is_valid_and_covers_both_versions():
    config_path = settings.project_root / "promptfooconfig.yaml"
    assert config_path.exists()

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert len(config["prompts"]) == 2
    assert config["tests"], "promptfoo config has no test cases"
    assert any("file://" in str(p) for p in config["prompts"])
