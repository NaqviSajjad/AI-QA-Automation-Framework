"""Loading and querying the customer-support test datasets."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field

from src.utils.config import settings


class SupportCase(BaseModel):
    """One row of the customer-support evaluation dataset."""

    id: str
    category: str
    user_query: str
    expected_behavior: str
    expected_tool: str | None = None
    expected_priority: str | None = None
    expected_escalation: bool = False
    expected_source: str | None = None
    # Other articles that are legitimately relevant to the same question.
    # Retrieval ground truth is a set, not a single label.
    acceptable_sources: list[str] = Field(default_factory=list)
    expects_uncertainty: bool = False
    reference_context: str = ""
    reference_answer: str | None = None
    # Behavioural descriptions of what the answer must avoid. Graded by the
    # model-based judge; too fuzzy for a deterministic string check.
    must_not: list[str] = Field(default_factory=list)
    # Literal strings that must never appear. Checked deterministically.
    forbidden_phrases: list[str] = Field(default_factory=list)
    follow_ups: list[str] = Field(default_factory=list)
    # A multi-turn case resolves on the follow-up, so the follow-up needs its own
    # reference. Grading turn two against turn one's expectation measures nothing.
    follow_up_expected_behavior: str | None = None
    tags: list[str] = Field(default_factory=list)

    @property
    def relevant_sources(self) -> set[str]:
        sources = set(self.acceptable_sources)
        if self.expected_source:
            sources.add(self.expected_source)
        return sources


class AdversarialCase(BaseModel):
    """One row of the AI-security dataset."""

    id: str
    attack_type: str
    user_query: str
    expected_behavior: str
    must_not: list[str] = Field(default_factory=list)
    forbidden_tools: list[str] = Field(default_factory=list)
    severity: str = "high"


def _read_json(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def load_support_cases() -> list[SupportCase]:
    rows = _read_json(settings.test_data_path / "customer_support_cases.json")
    return [SupportCase(**row) for row in rows]


@lru_cache(maxsize=1)
def load_adversarial_cases() -> list[AdversarialCase]:
    rows = _read_json(settings.test_data_path / "adversarial_cases.json")
    return [AdversarialCase(**row) for row in rows]


def cases_by_category(category: str) -> list[SupportCase]:
    return [case for case in load_support_cases() if case.category == category]


def case_by_id(case_id: str) -> SupportCase:
    for case in load_support_cases():
        if case.id == case_id:
            return case
    raise KeyError(f"Unknown case id: {case_id}")


def case_ids(cases: list[SupportCase] | list[AdversarialCase]) -> list[str]:
    """pytest ids so a failure names the scenario, not `case12`."""
    return [c.id for c in cases]
