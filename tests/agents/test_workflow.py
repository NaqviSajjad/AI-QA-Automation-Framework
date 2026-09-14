"""The workflow as a whole: paths, budgets, determinism and runtime parity."""

from __future__ import annotations

import pytest

from src.agents.graph import LANGGRAPH_AVAILABLE, SupportAgent
from src.utils.config import settings

pytestmark = [pytest.mark.agent]


def test_answering_path_visits_every_stage(support_agent):
    state = support_agent.run("I was charged twice this month.")
    assert state.node_path == ["classify", "retrieve", "tool", "generate", "validate"]
    assert state.completed


def test_clarification_path_skips_retrieval_and_tools(support_agent):
    state = support_agent.run("Hello, can you help me?")
    assert state.node_path == ["classify", "clarify", "validate"]
    assert state.selected_tool is None
    assert state.retrieved_context == []


def test_escalation_path_short_circuits(support_agent):
    state = support_agent.run("I want to speak to a human manager about a fraudulent charge.")
    assert state.node_path == ["classify", "escalate"]
    assert state.requires_escalation


def test_call_budgets_are_respected(support_agent):
    for query in [
        "I was charged twice this month.",
        "My card was declined.",
        "How do I cancel my subscription?",
    ]:
        state = support_agent.run(query)
        assert state.llm_calls <= settings.agent.max_llm_calls
        assert len(state.tool_calls) <= settings.agent.max_tool_calls
        assert state.retry_count <= settings.agent.max_retries


def test_the_same_question_gives_the_same_answer(support_agent):
    """Determinism is what makes regression testing of an LLM system possible
    at all. With the mock provider this must hold exactly."""
    if settings.llm.provider != "mock":
        pytest.skip("only guaranteed with the deterministic mock provider")

    first = support_agent.run("I was charged twice this month.")
    second = support_agent.run("I was charged twice this month.")
    assert first.response == second.response
    assert first.category == second.category
    assert first.node_path == second.node_path
    assert first.conversation_id != second.conversation_id


def test_each_run_gets_its_own_identifiers(support_agent):
    first = support_agent.run("I cannot sign in.")
    second = support_agent.run("I cannot sign in.")
    assert first.conversation_id != second.conversation_id
    assert first.trace_id != second.trace_id


def test_conversation_history_is_carried_into_the_answer(support_agent):
    history = [
        {"role": "user", "content": "I have a payment problem."},
        {"role": "assistant", "content": "Could you tell me a little more about what happened?"},
    ]
    state = support_agent.run("It was declined yesterday.", history=history)
    assert state.category == "payment"
    assert state.response


@pytest.mark.skipif(not LANGGRAPH_AVAILABLE, reason="LangGraph extra not installed")
def test_langgraph_and_builtin_runtimes_agree():
    """The built-in executor exists so the framework runs without the optional
    LangGraph extra. If the two runtimes could diverge, every agent test would
    depend on which one happened to be installed."""
    graph_agent = SupportAgent(use_langgraph=True)
    plain_agent = SupportAgent(use_langgraph=False)
    assert graph_agent.runtime == "langgraph"
    assert plain_agent.runtime == "builtin"

    def normalise(state) -> str:
        # Every run gets a fresh conversation id, and the escalation message
        # quotes it - so compare the answer with the id masked out.
        return (state.response or "").replace(state.conversation_id, "<conversation-id>")

    for query in [
        "I was charged twice this month.",
        "Hello, can you help me?",
        "I want to speak to a human manager about a fraudulent charge.",
    ]:
        left = graph_agent.run(query)
        right = plain_agent.run(query)
        assert left.category == right.category
        assert left.selected_tool == right.selected_tool
        assert left.node_path == right.node_path
        assert normalise(left) == normalise(right)


def test_a_trace_is_recorded_for_every_run(support_agent, tracer):
    state = support_agent.run("How long does a refund take?")
    span = tracer.find(state.trace_id) or tracer.last()

    assert span is not None
    assert span.metadata["prompt_version"] == settings.prompts.active_version
    assert span.metrics["llm_calls"] >= 1
    assert span.output["category"] == state.category
