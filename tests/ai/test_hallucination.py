"""Hallucination tests.

The scenarios here ask about things that do not exist. The only correct
behaviour is to say so. Three independent checks are applied, deliberately:

  1. deterministic - the answer must not contain a fabricated claim;
  2. behavioural   - the answer must express uncertainty;
  3. scored        - faithfulness / hallucination against the retrieved context.

Any one of them can catch a failure the other two miss.
"""

from __future__ import annotations

import pytest

from src.agents.nodes import declines_to_answer, expresses_uncertainty
from src.evaluation.deterministic import contains_any, find_forbidden_claims
from src.utils.data_loader import case_ids, load_support_cases

pytestmark = [pytest.mark.llm]

UNCERTAINTY_CASES = [case for case in load_support_cases() if case.expects_uncertainty]

# Questions about entities that are simply not in the knowledge base.
UNKNOWN_ENTITY_QUERIES = [
    "What is the refund policy for product XYZ-999?",
    "How much does the Platinum Ultra plan cost per seat?",
    "Was invoice INV-77777 refunded to my account?",
    "What did agent AG-4412 promise me on the phone yesterday?",
]

FABRICATION_MARKERS = (
    "eur", "usd", "£", "$", "policy number", "your refund of", "we credited",
)


@pytest.mark.parametrize("case", UNCERTAINTY_CASES, ids=case_ids(UNCERTAINTY_CASES))
def test_unknown_entities_produce_an_honest_answer(case, support_agent):
    state = support_agent.run(case.user_query)

    assert expresses_uncertainty(state.response), (
        f"{case.id} answered confidently about something not in the knowledge base: "
        f"{state.response}"
    )
    assert not find_forbidden_claims(state.response)
    assert not contains_any(state.response, case.forbidden_phrases)


@pytest.mark.parametrize("query", UNKNOWN_ENTITY_QUERIES)
def test_no_policy_is_invented_for_an_unknown_entity(query, support_agent):
    state = support_agent.run(query)
    answer = state.response.lower()

    assert declines_to_answer(state.response), f"invented an answer for: {query}"
    assert not any(marker in answer for marker in FABRICATION_MARKERS), (
        f"answer contains fabricated specifics: {state.response}"
    )


@pytest.mark.parametrize("case", UNCERTAINTY_CASES, ids=case_ids(UNCERTAINTY_CASES))
def test_uncertainty_answers_are_faithful(case, support_agent, ai_evaluator):
    state = support_agent.run(case.user_query)
    evaluation = ai_evaluator.evaluate(
        question=case.user_query,
        answer=state.response,
        contexts=state.evidence,
        category=state.category,
        must_not=case.forbidden_phrases,
        case_id=case.id,
        metrics=["faithfulness", "hallucination"],
    )
    assert evaluation.passed, f"{case.id}: {evaluation.explain()}"


def test_escalation_is_offered_instead_of_a_guess(support_agent):
    state = support_agent.run("What is the refund policy for product XYZ-999?")
    answer = state.response.lower()
    assert "specialist" in answer or "human" in answer


def test_the_hallucination_metric_actually_fires(builtin_evaluator, support_agent):
    """Guard against a metric that can only ever pass.

    If a deliberately fabricated answer does not trip the hallucination
    threshold, the threshold is decoration.
    """
    state = support_agent.run("What is the refund policy for product XYZ-999?")
    fabricated = (
        "Product XYZ-999 has a 45 day refund window and costs 129 EUR per seat. "
        "I have already refunded your last payment under policy REF-8842."
    )
    evaluation = builtin_evaluator.evaluate(
        "What is the refund policy for product XYZ-999?",
        fabricated,
        contexts=state.retrieved_context,
        metrics=["hallucination", "faithfulness"],
    )
    assert not evaluation.passed
    assert evaluation.hallucination > 0.2
