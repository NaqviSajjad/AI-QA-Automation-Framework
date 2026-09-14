"""Input guardrails.

Prompt-injection defence belongs in the workflow, not in the model's good
manners. A dedicated branch means the refusal is deterministic, testable, and
identical whichever provider is behind the agent - and it means the injected
text never reaches the generation prompt at all.

The same marker list is used by the mock provider, so a test that exercises the
model path and a test that exercises the graph path agree on what an attack is.
"""

from __future__ import annotations

import re

INJECTION_MARKERS: tuple[str, ...] = (
    "ignore all previous", "ignore previous instructions", "ignore your instructions",
    "disregard your rules", "disregard all previous", "system prompt",
    "repeat everything above", "verbatim", "print them raw", "print your configuration",
    "devmode", "dev mode", "without restrictions", "you are now",
    "reveal your", "show me your instructions", "your instructions",
    "do what it says", "act as an unrestricted",
)

DATA_EXFILTRATION_MARKERS: tuple[str, ...] = (
    "internal customer records", "every account", "all customers",
    "full card number", "does the account for", "list all accounts",
)

PRIVILEGE_MARKERS: tuple[str, ...] = (
    "admin tool", "grant my user the admin", "issue the refund yourself",
    "use your admin", "make an exception",
)

REFUSAL_RESPONSE = (
    "I am not able to share my internal instructions or configuration, act outside the "
    "support actions available to me, or give out information about another customer's "
    "account. I can still help with your own account: billing and invoices, payments, "
    "refunds, subscriptions, sign-in problems and technical issues. Tell me what you need "
    "and I will take it from there, or I can pass you to a human support specialist."
)


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower())


def is_injection(text: str) -> bool:
    """True when the message tries to change the assistant's instructions."""
    lowered = _normalise(text)
    return any(marker in lowered for marker in INJECTION_MARKERS)


def is_data_exfiltration(text: str) -> bool:
    lowered = _normalise(text)
    return any(marker in lowered for marker in DATA_EXFILTRATION_MARKERS)


def is_privilege_escalation(text: str) -> bool:
    lowered = _normalise(text)
    return any(marker in lowered for marker in PRIVILEGE_MARKERS)


def threat_type(text: str) -> str | None:
    """Name the threat, so the trace and the report say *why* it was refused."""
    if is_injection(text):
        return "prompt_injection"
    if is_data_exfiltration(text):
        return "data_exfiltration"
    if is_privilege_escalation(text):
        return "privilege_escalation"
    return None
