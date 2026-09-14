"""Agent tools and the authorisation boundary around them.

The allowlist is enforced in *one* place - `execute_tool` - so a test can prove
that no code path invokes an unlisted tool. Tool arguments are validated with
Pydantic, because "the agent called the right tool with the wrong arguments" is
a real and very common agent bug.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from src.utils.config import settings
from src.utils.logging import get_logger

logger = get_logger(__name__)

ALLOWED_TOOLS: set[str] = set(settings.agent.allowed_tools) or {
    "get_billing_history",
    "get_account_status",
    "get_subscription",
    "get_payment_status",
    "get_refund_policy",
    "create_support_ticket",
}

# Tools that exist in the wider platform but must never be reachable from the
# customer-facing support agent. Used by tests/security/test_tool_authorization.
SENSITIVE_TOOLS: set[str] = {
    "delete_customer_account",
    "issue_refund",
    "read_internal_notes",
    "list_all_customers",
    "export_card_numbers",
    "grant_admin_role",
}


class ToolAuthorizationError(PermissionError):
    """Raised when the agent tries to invoke a tool outside the allowlist."""


class ToolArgumentError(ValueError):
    """Raised when tool arguments do not satisfy the tool's schema."""


# --------------------------------------------------------------------------- #
# Argument schemas
# --------------------------------------------------------------------------- #
class AccountScopedArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    account_id: str = Field(min_length=3)


class BillingHistoryArgs(AccountScopedArgs):
    months: int = Field(default=3, ge=1, le=24)


class RefundPolicyArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    plan: str = "business"


class TicketArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    account_id: str = Field(min_length=3)
    summary: str = Field(min_length=5, max_length=200)
    severity: str = "3"
    category: str = "technical"


# --------------------------------------------------------------------------- #
# Tool implementations (deterministic fixtures - this is the system under test,
# not a production backend)
# --------------------------------------------------------------------------- #
def get_billing_history(account_id: str, months: int = 3) -> dict[str, Any]:
    return {
        "account_id": account_id,
        "months": months,
        "invoices": [
            {"invoice": "INV-2043", "date": "2026-08-01", "amount": 49.00, "status": "paid"},
            {"invoice": "INV-2043", "date": "2026-08-01", "amount": 49.00, "status": "paid",
             "flag": "possible_duplicate"},
            {"invoice": "INV-1988", "date": "2026-07-01", "amount": 49.00, "status": "paid"},
        ],
        "duplicate_suspected": True,
    }


def get_account_status(account_id: str) -> dict[str, Any]:
    return {
        "account_id": account_id,
        "status": "active",
        "locked": False,
        "two_factor_enabled": True,
        "failed_sign_ins_24h": 3,
        "role": "owner",
    }


def get_subscription(account_id: str) -> dict[str, Any]:
    return {
        "account_id": account_id,
        "plan": "business",
        "seats": 12,
        "billing_interval": "monthly",
        "auto_renew": True,
        "renews_on": "2026-10-01",
        "trial": False,
    }


def get_payment_status(account_id: str) -> dict[str, Any]:
    return {
        "account_id": account_id,
        "last_attempt": "2026-08-30T09:12:00Z",
        "state": "failed",
        "decline_code": "do_not_honor",
        "card_brand": "visa",
        "card_last4": "4242",
        "next_retry": "2026-09-02T09:12:00Z",
    }


def get_refund_policy(plan: str = "business") -> dict[str, Any]:
    return {
        "plan": plan,
        "window_days": 14,
        "prorated_after_window": True,
        "returns_to": "original_payment_method",
        "card_processing_days": "5-10 business days",
    }


def create_support_ticket(
    account_id: str, summary: str, severity: str = "3", category: str = "technical"
) -> dict[str, Any]:
    return {
        "ticket_id": f"TCK-{abs(hash((account_id, summary))) % 90000 + 10000}",
        "account_id": account_id,
        "summary": summary,
        "severity": severity,
        "category": category,
        "status": "open",
    }


TOOL_REGISTRY: dict[str, Callable[..., dict[str, Any]]] = {
    "get_billing_history": get_billing_history,
    "get_account_status": get_account_status,
    "get_subscription": get_subscription,
    "get_payment_status": get_payment_status,
    "get_refund_policy": get_refund_policy,
    "create_support_ticket": create_support_ticket,
}

ARGUMENT_SCHEMAS: dict[str, type[BaseModel]] = {
    "get_billing_history": BillingHistoryArgs,
    "get_account_status": AccountScopedArgs,
    "get_subscription": AccountScopedArgs,
    "get_payment_status": AccountScopedArgs,
    "get_refund_policy": RefundPolicyArgs,
    "create_support_ticket": TicketArgs,
}

# Argument keys that must never be sent to a tool, even if the model asks.
FORBIDDEN_ARGUMENT_KEYS = {"card_number", "cvc", "password", "ssn", "full_pan", "api_key"}


def is_authorized(tool_name: str) -> bool:
    return tool_name in ALLOWED_TOOLS and tool_name in TOOL_REGISTRY


def validate_arguments(tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    leaked = FORBIDDEN_ARGUMENT_KEYS & set(arguments)
    if leaked:
        raise ToolArgumentError(f"Sensitive argument(s) refused: {sorted(leaked)}")
    schema = ARGUMENT_SCHEMAS.get(tool_name)
    if schema is None:
        raise ToolArgumentError(f"No argument schema registered for '{tool_name}'")
    try:
        return schema(**arguments).model_dump()
    except ValidationError as exc:
        raise ToolArgumentError(f"Invalid arguments for '{tool_name}': {exc.errors()}") from exc


def execute_tool(tool_name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
    """The single gate every tool call passes through."""
    arguments = dict(arguments or {})
    if not is_authorized(tool_name):
        logger.warning("unauthorized_tool_call", extra={"tool": tool_name})
        raise ToolAuthorizationError(
            f"Tool '{tool_name}' is not in the allowlist {sorted(ALLOWED_TOOLS)}"
        )
    validated = validate_arguments(tool_name, arguments)
    started = time.perf_counter()
    result = TOOL_REGISTRY[tool_name](**validated)
    logger.info(
        "tool_call",
        extra={"tool": tool_name, "latency_ms": round((time.perf_counter() - started) * 1000, 2)},
    )
    return result
