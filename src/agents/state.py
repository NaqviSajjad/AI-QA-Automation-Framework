"""Agent state and the structured contract the AI must satisfy.

`SupportResponse` is the deterministic half of AI testing: whatever the model
says in prose, the envelope around it has a fixed shape, a closed set of
categories and a boolean escalation flag - and those can be asserted exactly.
"""

from __future__ import annotations

import uuid
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Category(str, Enum):
    BILLING = "billing"
    REFUND = "refund"
    PAYMENT = "payment"
    SUBSCRIPTION = "subscription"
    ACCOUNT = "account"
    TECHNICAL = "technical"
    UNKNOWN = "unknown"


class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


VALID_CATEGORIES = tuple(c.value for c in Category)
VALID_PRIORITIES = tuple(p.value for p in Priority)


class ToolCall(BaseModel):
    """A single tool invocation, recorded whether or not it was allowed."""

    model_config = ConfigDict(extra="forbid")

    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    result: dict[str, Any] | None = None
    authorized: bool = True
    error: str | None = None
    latency_ms: float = 0.0


class SupportResponse(BaseModel):
    """The structured output the support AI must produce."""

    model_config = ConfigDict(extra="forbid")

    category: Category
    priority: Priority
    requires_escalation: bool
    response: str
    tool_used: str | None = None
    sources: list[str] = Field(default_factory=list)
    conversation_id: str | None = None

    @field_validator("response")
    @classmethod
    def response_must_be_substantive(cls, value: str) -> str:
        text = value.strip()
        if not text:
            raise ValueError("response must not be empty")
        if len(text.split()) < 5:
            raise ValueError("response must contain at least 5 words")
        return text


class AgentState(BaseModel):
    """State carried through the LangGraph workflow.

    Kept as a Pydantic model rather than a bare dict so that every state
    transition is itself testable (tests/agents/test_state.py).
    """

    model_config = ConfigDict(validate_assignment=True)

    user_query: str
    account_id: str = "ACC-1001"
    conversation_history: list[dict[str, str]] = Field(default_factory=list)
    conversation_id: str = Field(default_factory=lambda: f"conv_{uuid.uuid4().hex[:12]}")
    trace_id: str = Field(default_factory=lambda: f"trace_{uuid.uuid4().hex[:12]}")

    category: str | None = None
    priority: str | None = None
    retrieved_context: list[str] = Field(default_factory=list)
    retrieved_sources: list[str] = Field(default_factory=list)
    selected_tool: str | None = None
    tool_result: dict[str, Any] | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)

    response: str | None = None
    structured_response: SupportResponse | None = None

    retry_count: int = 0
    requires_escalation: bool = False
    validation_errors: list[str] = Field(default_factory=list)
    unauthorized_tool_attempts: list[str] = Field(default_factory=list)
    threat_detected: str | None = None

    llm_calls: int = 0
    latency_ms: float = 0.0
    node_path: list[str] = Field(default_factory=list)

    # -- helpers ----------------------------------------------------------- #
    def visit(self, node: str) -> None:
        self.node_path = [*self.node_path, node]

    @property
    def context_text(self) -> str:
        return "\n\n".join(self.retrieved_context)

    @property
    def history_text(self) -> str:
        return "\n".join(f"{turn['role']}: {turn['content']}" for turn in self.conversation_history)

    @property
    def evidence(self) -> list[str]:
        """Everything the answer is allowed to be grounded in: retrieved context
        plus any tool result. This is what faithfulness is measured against."""
        import json as _json

        evidence = list(self.retrieved_context)
        if self.tool_result:
            evidence.append(_json.dumps(self.tool_result, indent=2))
        return evidence

    @property
    def tool_names(self) -> list[str]:
        return [call.name for call in self.tool_calls]

    @property
    def completed(self) -> bool:
        return bool(self.response) and not self.validation_errors

    def to_public_dict(self) -> dict[str, Any]:
        """What the API returns - internal reasoning is deliberately excluded."""
        return {
            "conversation_id": self.conversation_id,
            "trace_id": self.trace_id,
            "category": self.category,
            "priority": self.priority,
            "requires_escalation": self.requires_escalation,
            "response": self.response,
            "tool_used": self.selected_tool,
            "sources": self.retrieved_sources,
            "latency_ms": round(self.latency_ms, 2),
            "llm_calls": self.llm_calls,
            "tool_calls": len(self.tool_calls),
            "retry_count": self.retry_count,
            "node_path": self.node_path,
        }
