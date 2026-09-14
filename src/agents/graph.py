"""The customer-support agent workflow.

    START -> classify -> route ─┬─ retrieve -> tool -> generate ─┐
                               ├─ clarify ────────────────────── ┤
                               └─ escalate ──────────────► END   │
                                                                 v
                                                              validate
                                                          ┌──────┼──────┐
                                                        pass   retry  escalate
                                                          |      |       |
                                                         END  generate  END

Runs on LangGraph when the optional `ai` extra is installed, and on an
equivalent built-in executor otherwise. Both drive the *same* node functions,
so agent tests assert on behaviour, not on which runtime executed it.
"""

from __future__ import annotations

from typing import Any

from src.agents import nodes
from src.agents.state import AgentState
from src.llm.base import LLMProvider
from src.observability.langfuse_client import get_tracer
from src.utils.config import settings
from src.utils.logging import get_logger

logger = get_logger(__name__)

try:  # pragma: no cover - optional
    from langgraph.graph import END, START, StateGraph

    LANGGRAPH_AVAILABLE = True
except ImportError:  # pragma: no cover
    StateGraph = None  # type: ignore[assignment]
    START = "__start__"  # type: ignore[assignment]
    END = "__end__"  # type: ignore[assignment]
    LANGGRAPH_AVAILABLE = False


class SupportAgent:
    """Public entry point used by the demo app, the API and the agent tests."""

    def __init__(self, llm: LLMProvider | None = None, use_langgraph: bool | None = None) -> None:
        from src.llm.factory import get_llm_provider

        self.llm = llm or get_llm_provider()
        self.use_langgraph = (
            LANGGRAPH_AVAILABLE if use_langgraph is None else (use_langgraph and LANGGRAPH_AVAILABLE)
        )
        self._compiled = self._build_langgraph() if self.use_langgraph else None

    @property
    def runtime(self) -> str:
        return "langgraph" if self._compiled is not None else "builtin"

    # -- LangGraph wiring -------------------------------------------------- #
    def _build_langgraph(self) -> Any:  # pragma: no cover - exercised when installed
        def wrap(fn):
            def node(state: AgentState) -> dict[str, Any]:
                return fn(state).model_dump()

            return node

        graph = StateGraph(AgentState)
        graph.add_node("classify", wrap(lambda s: nodes.classify_node(s, self.llm)))
        graph.add_node("retrieve", wrap(nodes.retrieve_node))
        graph.add_node("tool", wrap(nodes.tool_node))
        graph.add_node("generate", wrap(lambda s: nodes.generate_node(s, self.llm)))
        graph.add_node("clarify", wrap(nodes.clarify_node))
        graph.add_node("refuse", wrap(nodes.refuse_node))
        graph.add_node("validate", wrap(nodes.validate_node))
        graph.add_node("retry", wrap(nodes.retry_node))
        graph.add_node("escalate", wrap(nodes.escalate_node))

        graph.add_edge(START, "classify")
        graph.add_conditional_edges(
            "classify",
            nodes.route,
            {
                "retrieve": "retrieve",
                "clarify": "clarify",
                "escalate": "escalate",
                "refuse": "refuse",
            },
        )
        graph.add_edge("retrieve", "tool")
        graph.add_edge("tool", "generate")
        graph.add_edge("generate", "validate")
        graph.add_edge("clarify", "validate")
        graph.add_edge("refuse", "validate")
        graph.add_conditional_edges(
            "validate",
            nodes.should_retry,
            {"pass": END, "retry": "retry", "escalate": "escalate"},
        )
        graph.add_edge("retry", "generate")
        graph.add_edge("escalate", END)
        return graph.compile()

    # -- built-in executor (identical semantics, no extra dependency) ------ #
    def _run_builtin(self, state: AgentState) -> AgentState:
        state = nodes.classify_node(state, self.llm)
        branch = nodes.route(state)

        if branch == "escalate":
            return nodes.escalate_node(state)

        if branch == "refuse":
            state = nodes.refuse_node(state)
        elif branch == "clarify":
            state = nodes.clarify_node(state)
        else:
            state = nodes.retrieve_node(state)
            state = nodes.tool_node(state)
            state = nodes.generate_node(state, self.llm)

        while True:
            state = nodes.validate_node(state)
            decision = nodes.should_retry(state)
            if decision == "pass":
                return state
            if decision == "escalate":
                return nodes.escalate_node(state)
            state = nodes.retry_node(state)
            state = nodes.generate_node(state, self.llm)

    # -- public API -------------------------------------------------------- #
    def run(
        self,
        query: str,
        history: list[dict[str, str]] | None = None,
        conversation_id: str | None = None,
        account_id: str | None = None,
    ) -> AgentState:
        state = AgentState(
            user_query=query,
            conversation_history=history or [],
            **({"conversation_id": conversation_id} if conversation_id else {}),
            **({"account_id": account_id} if account_id else {}),
        )
        tracer = get_tracer()
        with tracer.trace(
            name="customer_support_agent",
            input={"query": query, "history_turns": len(history or [])},
            metadata={
                "runtime": self.runtime,
                "provider": self.llm.name,
                "prompt_version": settings.prompts.active_version,
                "conversation_id": state.conversation_id,
            },
        ) as span:
            if self._compiled is not None:  # pragma: no cover - optional runtime
                raw = self._compiled.invoke(state)
                result = AgentState(**raw) if isinstance(raw, dict) else raw
            else:
                result = self._run_builtin(state)

            span.update(
                output={"response": result.response, "category": result.category},
                metrics={
                    "latency_ms": result.latency_ms,
                    "llm_calls": result.llm_calls,
                    "tool_calls": len(result.tool_calls),
                    "retries": result.retry_count,
                },
            )
            result.trace_id = span.trace_id or result.trace_id

        logger.info(
            "agent_run",
            extra={
                "conversation_id": result.conversation_id,
                "category": result.category,
                "tool": result.selected_tool,
                "escalated": result.requires_escalation,
                "path": result.node_path,
            },
        )
        return result


_AGENT: SupportAgent | None = None


def get_agent(refresh: bool = False) -> SupportAgent:
    global _AGENT
    if _AGENT is None or refresh:
        _AGENT = SupportAgent()
    return _AGENT
