"""Workflow nodes.

Each node is a plain function `AgentState -> AgentState`. Keeping them free of
LangGraph types means every node is unit-testable on its own (see
tests/agents/test_state.py) and the same code runs under the LangGraph runtime
or the built-in sequential fallback.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any

from src.agents import guardrails
from src.agents.state import (
    VALID_CATEGORIES,
    VALID_PRIORITIES,
    AgentState,
    SupportResponse,
    ToolCall,
)
from src.agents.tools import (
    ToolArgumentError,
    ToolAuthorizationError,
    execute_tool,
    is_authorized,
)
from src.llm.base import LLMProvider
from src.llm.mock_provider import MockLLMProvider
from src.utils.config import settings
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Which tool serves which intent. Routing is deterministic on purpose: an agent
# that picks tools by vibes cannot be regression-tested.
TOOL_BY_CATEGORY: dict[str, str | None] = {
    "billing": "get_billing_history",
    "refund": "get_refund_policy",
    "payment": "get_payment_status",
    "subscription": "get_subscription",
    "account": "get_account_status",
    "technical": "create_support_ticket",
    "unknown": None,
}

# Categories whose answers must come from the knowledge base.
RAG_CATEGORIES = {"billing", "refund", "payment", "subscription", "account", "technical"}

# Claims the assistant is never allowed to make. Deterministic, cheap, and they
# catch the most damaging hallucinations before any model-based metric runs.
FORBIDDEN_CLAIM_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\b(?:i|we)\s+(?:have\s+)?(?:already\s+)?(?:issued|processed|completed)\s+(?:your\s+)?refund",
     "claims a refund was already issued"),
    (r"\brefund\s+has\s+been\s+(?:processed|issued|completed|sent)\b",
     "claims a refund was already processed"),
    (r"\b(?:i|we)\s+(?:have\s+)?reversed\s+the\s+charge\b", "claims a charge was already reversed"),
    (r"\byour\s+account\s+has\s+been\s+unlocked\b", "claims an account was already unlocked"),
    (r"\bi\s+have\s+(?:cancelled|canceled)\s+your\s+subscription\b",
     "claims a subscription was already cancelled"),
    (r"\bsystem\s+prompt\b", "references the system prompt"),
)

# A forbidden claim only counts when it is *asserted*. Support articles legitimately
# contain sentences like "an agent must not state that a refund has been processed",
# and flagging those would train engineers to ignore the check.
NEGATION_GUARDS = (
    "must not", "cannot", "can not", "never", "do not", "does not", "will not",
    "should not", "won't", "don't", "doesn't", "not able to", "unable to",
)

UNCERTAINTY_MARKERS = (
    "do not have enough information",
    "don't have enough information",
    "not able to confirm",
    "cannot confirm",
    "no information",
    "not documented",
    "will not make one up",
    "i do not have",
    "i don't have",
)


def find_forbidden_claims(text: str) -> list[str]:
    """Assertions the assistant must never make.

    Negated or quoted policy text ("an agent must not state that a refund has
    been processed") is not an assertion, so it is not a finding - a check that
    cries wolf gets switched off, and then it protects nothing.
    """
    body = text or ""
    found: list[str] = []
    for pattern, description in FORBIDDEN_CLAIM_PATTERNS:
        for match in re.finditer(pattern, body, re.IGNORECASE):
            preceding = body[max(0, match.start() - 60) : match.start()].lower()
            if any(guard in preceding for guard in NEGATION_GUARDS):
                continue
            found.append(description)
            break
    return found


def _load_prompt(name: str) -> str:
    path = settings.prompts_path / f"{name}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Prompt template not found: {path}")
    return path.read_text(encoding="utf-8")


def _safe_json(text: str) -> dict[str, Any] | None:
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else text[text.find("{") : text.rfind("}") + 1]
    try:
        payload = json.loads(candidate)
        return payload if isinstance(payload, dict) else None
    except (json.JSONDecodeError, ValueError):
        return None


# --------------------------------------------------------------------------- #
# Nodes
# --------------------------------------------------------------------------- #
def classify_node(state: AgentState, llm: LLMProvider) -> AgentState:
    """Turn free text into a closed-set category + priority + escalation flag."""
    state.visit("classify")
    state.threat_detected = guardrails.threat_type(state.user_query)
    prompt = _load_prompt("classify").format(
        history=state.history_text or "(none)", query=state.user_query
    )
    started = time.perf_counter()
    # A transport failure (timeout, empty completion, upstream error) is not the
    # agent's to paper over: it propagates so the API can answer 504/502 and the
    # customer can retry. Only a *malformed* classification falls back to the
    # deterministic guard rails below.
    completion = llm.complete(prompt)
    state.llm_calls += 1
    payload = _safe_json(completion.text) or {}
    state.latency_ms += (time.perf_counter() - started) * 1000

    category = str(payload.get("category", "")).lower()
    priority = str(payload.get("priority", "")).lower()

    # Deterministic guard rails: an out-of-vocabulary label from the model never
    # reaches the rest of the workflow.
    if category not in VALID_CATEGORIES:
        category = MockLLMProvider.classify(state.user_query)
    if priority not in VALID_PRIORITIES:
        priority = MockLLMProvider.priority(state.user_query)

    state.category = category
    state.priority = priority
    state.requires_escalation = bool(
        payload.get("requires_escalation", False)
    ) or MockLLMProvider.needs_escalation(state.user_query)
    return state


def route(state: AgentState) -> str:
    """Branch name for the router edge.

    The guardrail branch comes first and is deliberately unconditional: an
    injected instruction must never reach the generation prompt, whatever the
    classifier made of it.
    """
    if state.threat_detected:
        return "refuse"
    if state.requires_escalation:
        return "escalate"
    if state.category in RAG_CATEGORIES:
        return "retrieve"
    return "clarify"


def refuse_node(state: AgentState) -> AgentState:
    """Deterministic refusal for a message that attacks the system itself."""
    state.visit("refuse")
    logger.warning(
        "guardrail_triggered",
        extra={"threat": state.threat_detected, "conversation_id": state.conversation_id},
    )
    state.response = guardrails.REFUSAL_RESPONSE
    state.priority = state.priority or "low"
    state.category = state.category or "unknown"
    return state


def retrieve_node(state: AgentState) -> AgentState:
    """Fetch supporting context. Empty context is a valid, meaningful outcome."""
    state.visit("retrieve")
    from src.rag.pipeline import get_pipeline

    started = time.perf_counter()
    query = state.user_query
    if state.conversation_history:
        # Follow-ups like "and the second one?" are unretrievable on their own.
        query = f"{state.history_text}\n{state.user_query}"
    chunks = get_pipeline().retrieve(query)
    state.latency_ms += (time.perf_counter() - started) * 1000
    state.retrieved_context = [chunk.content for chunk in chunks]
    state.retrieved_sources = sorted({chunk.source for chunk in chunks})
    return state


def tool_node(state: AgentState) -> AgentState:
    """Select and execute at most one tool, through the authorisation gate."""
    state.visit("tool")
    tool_name = TOOL_BY_CATEGORY.get(state.category or "unknown")
    if not tool_name:
        return state
    if len(state.tool_calls) >= settings.agent.max_tool_calls:
        state.validation_errors.append("tool call budget exceeded")
        return state

    arguments: dict[str, Any] = {"account_id": state.account_id}
    if tool_name == "get_refund_policy":
        arguments = {"plan": "business"}
    elif tool_name == "create_support_ticket":
        arguments = {
            "account_id": state.account_id,
            "summary": state.user_query[:180],
            "severity": "3" if state.priority in (None, "low", "medium") else "2",
            "category": state.category or "technical",
        }
    elif tool_name == "get_billing_history":
        arguments["months"] = 3

    started = time.perf_counter()
    call = ToolCall(name=tool_name, arguments=arguments, authorized=is_authorized(tool_name))
    try:
        call.result = execute_tool(tool_name, arguments)
        state.selected_tool = tool_name
        state.tool_result = call.result
    except ToolAuthorizationError as exc:
        call.authorized = False
        call.error = str(exc)
        state.unauthorized_tool_attempts.append(tool_name)
        state.validation_errors.append(f"unauthorized tool: {tool_name}")
    except ToolArgumentError as exc:
        call.error = str(exc)
        state.validation_errors.append(f"invalid tool arguments: {exc}")
    finally:
        call.latency_ms = round((time.perf_counter() - started) * 1000, 2)
        state.tool_calls = [*state.tool_calls, call]
        state.latency_ms += call.latency_ms
    return state


def generate_node(state: AgentState, llm: LLMProvider) -> AgentState:
    """Produce the customer-facing answer from context + tool result."""
    state.visit("generate")
    template = _load_prompt(settings.prompts.active_version)
    tool_result = json.dumps(state.tool_result, indent=2) if state.tool_result else ""
    prompt = template.format(
        history=state.history_text or "(none)",
        context=state.context_text or "",
        tool_result=tool_result,
        query=state.user_query,
    )
    started = time.perf_counter()
    completion = llm.complete(prompt)
    state.llm_calls += 1
    state.response = completion.text.strip()
    state.latency_ms += (time.perf_counter() - started) * 1000
    return state


def clarify_node(state: AgentState) -> AgentState:
    """Unknown intent: ask, do not guess."""
    state.visit("clarify")
    state.category = state.category or "unknown"
    state.priority = state.priority or "low"
    state.response = (
        "Thanks for getting in touch. Could you tell me a little more about what happened - "
        "for example whether it relates to a charge, a payment, your subscription, signing "
        "in, or something not working as expected? I can then either resolve it for you or "
        "pass it to the right specialist."
    )
    return state


def validate_node(state: AgentState) -> AgentState:
    """Deterministic gate on the generated answer.

    This runs *before* any model-based metric. Cheap, exact checks first;
    probabilistic judgement afterwards.
    """
    state.visit("validate")
    state.validation_errors = [
        error for error in state.validation_errors if not error.startswith("response ")
    ]
    text = (state.response or "").strip()

    if not text:
        state.validation_errors.append("response is empty")
    elif len(text.split()) < 8:
        state.validation_errors.append("response is too short to be useful")

    for description in find_forbidden_claims(text):
        state.validation_errors.append(f"forbidden claim: {description}")

    if state.category not in VALID_CATEGORIES:
        state.validation_errors.append(f"invalid category: {state.category}")
    if state.priority not in VALID_PRIORITIES:
        state.validation_errors.append(f"invalid priority: {state.priority}")
    if state.unauthorized_tool_attempts:
        state.validation_errors.append("unauthorized tool attempt recorded")

    if not state.validation_errors and text:
        try:
            state.structured_response = SupportResponse(
                category=state.category,  # type: ignore[arg-type]
                priority=state.priority,  # type: ignore[arg-type]
                requires_escalation=state.requires_escalation,
                response=text,
                tool_used=state.selected_tool,
                sources=state.retrieved_sources,
                conversation_id=state.conversation_id,
            )
        except Exception as exc:  # pragma: no cover - defensive
            state.validation_errors.append(f"schema violation: {exc}")
    return state


def should_retry(state: AgentState) -> str:
    """Edge decision after validation: pass, retry, or escalate."""
    if not state.validation_errors:
        return "pass"
    if state.retry_count < settings.agent.max_retries:
        return "retry"
    return "escalate"


def retry_node(state: AgentState) -> AgentState:
    state.visit("retry")
    state.retry_count += 1
    state.validation_errors = []
    state.response = None
    return state


def escalate_node(state: AgentState) -> AgentState:
    """Hand off to a human and say so in the answer."""
    state.visit("escalate")
    state.requires_escalation = True
    handoff = (
        "I am handing this over to a human support specialist who can look at the account "
        "directly. You will get a reply by email, and your conversation reference is "
        f"{state.conversation_id}. I can stay on the line if you have anything to add."
    )
    state.response = f"{state.response.strip()} {handoff}" if state.response else handoff
    state.priority = state.priority if state.priority in ("high", "critical") else "high"
    state.validation_errors = []
    state.structured_response = SupportResponse(
        category=state.category or "unknown",  # type: ignore[arg-type]
        priority=state.priority,  # type: ignore[arg-type]
        requires_escalation=True,
        response=state.response,
        tool_used=state.selected_tool,
        sources=state.retrieved_sources,
        conversation_id=state.conversation_id,
    )
    return state


CLARIFICATION_MARKERS = (
    "could you tell me a little more",
    "tell me a little more",
    "could you share",
    "can you tell me more",
    "what happened",
)


def expresses_uncertainty(text: str) -> bool:
    lowered = (text or "").lower()
    return any(marker in lowered for marker in UNCERTAINTY_MARKERS)


def declines_to_answer(text: str) -> bool:
    """The assistant avoided inventing something - by saying it does not know,
    or by asking for the detail it needs. Both are honest outcomes; only a
    confident fabrication is a failure."""
    lowered = (text or "").lower()
    return expresses_uncertainty(text) or any(m in lowered for m in CLARIFICATION_MARKERS)
