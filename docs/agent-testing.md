# Agent testing

An agent is a state machine that a language model steers. So it is tested like a state
machine — transitions, budgets, authorisation — with model-based judgement reserved for
the one thing that genuinely needs it: did the customer's problem get solved.

## The workflow

```
START -> classify -> route ─┬─ refuse   (guardrail) ─────────► validate
                            ├─ escalate ────────────────────► END
                            ├─ clarify ─────────────────────► validate
                            └─ retrieve -> tool -> generate ─► validate
                                                                │
                                                    ┌───────────┼──────────┐
                                                  pass        retry     escalate
                                                    │           │          │
                                                   END      generate      END
```

State is a Pydantic model (`src/agents/state.py`), which is what makes each node testable
in isolation.

## What is tested, and how

### Routing (`test_routing.py`)

Every dataset case must land in its expected category, plus representative intents and the
router's own branch decisions. Two properties matter beyond individual cases:

* **classification is always inside the enum.** Whatever the model returns, an
  out-of-vocabulary label never reaches the rest of the workflow — `classify_node` falls
  back to deterministic guard rails.
* **escalation wins over everything.** A customer asking for a human gets one, whatever
  the category.

### Tool selection and arguments (`test_tools.py`)

"The agent called the right tool" is half an assertion. The other half is where the
defects are:

* correct tool per intent, scored with `tool_correctness`;
* no tool at all for an unclassified request;
* at most one tool call per turn;
* arguments valid, typed, and carrying the right context (a ticket's severity, a billing
  lookup's window);
* invalid arguments rejected by the Pydantic schema — missing, out of range, unexpected
  extra keys;
* **sensitive argument keys refused outright** (`card_number`, `cvc`, `password`,
  `api_key`), regardless of what the model asked for;
* every registered tool has an argument schema — a structural test that stops a new tool
  being added without one.

### Authorisation (`test_tools.py`, `security/test_tool_authorization.py`)

```python
ALLOWED_TOOLS = {
    "get_billing_history", "get_account_status", "get_subscription",
    "get_payment_status", "get_refund_policy", "create_support_ticket",
}
```

The allowlist is enforced in exactly one place — `execute_tool` — so a test can prove no
code path bypasses it:

* every sensitive tool (`delete_customer_account`, `issue_refund`, `export_card_numbers`,
  `grant_admin_role`, …) raises;
* an unknown tool name raises;
* the allowlist and the registry agree, and do not overlap the sensitive set;
* **no run over the entire dataset ever attempts an unauthorised tool**;
* the allowlist contains no destructive operation — the only writing tool a
  customer-facing agent has is raising a ticket;
* an angry, escalating customer does not unlock extra tools.

And the sharpest one: the mock provider is asked to *request* a forbidden tool, and the
test asserts that asking is not the same as getting.

### State transitions (`test_state.py`)

Each node with a hand-built state:

| Node | Asserted |
|---|---|
| `classify` | category and priority populated, path recorded |
| `retrieve` | context and sources populated |
| `tool` | call recorded, result attached, budget enforced |
| `validate` | empty answer rejected; forbidden claim rejected; **quoted policy allowed** |
| `retry` | counter incremented, failed answer cleared |
| `escalate` | flag set, priority raised, reference quoted |

Plus the structured-output contract: missing fields, wrong types, invalid enums, empty
responses, malformed JSON, and unexpected extra fields — each rejected, each with its own
test.

The "quoted policy allowed" case is the interesting one. Quoting the rule is not breaking
it, and a validator that cannot tell the difference will be switched off.

### Workflow (`test_workflow.py`)

* each path visits exactly the expected nodes;
* LLM, tool and retry budgets hold — cost control is a correctness property for agents;
* **the same question gives the same answer** with the mock provider, which is what makes
  regression testing of an LLM system possible at all;
* every run gets its own conversation and trace ids;
* conversation history is carried into the follow-up;
* **the LangGraph runtime and the built-in executor agree** — same category, tool, path
  and answer, so no agent test depends on which runtime happened to be installed;
* a trace is recorded for every run, carrying the prompt version and call counts.

### Task completion (`test_task_completion.py`)

The metric that asks whether support actually happened. An answer can be relevant and
faithful and still leave the customer with nothing to do.

`task_completion` combines behaviour match, an actionable next step, and the correct
escalation decision. Alongside it:

* escalation decisions match the dataset exactly, across all 35 cases;
* **every answer offers a way forward** — resolve, ask, or escalate. No dead ends.

A multi-turn case is graded on the turn that resolves it, against its own
`follow_up_expected_behavior`. Grading turn two against turn one's expectation measures
nothing.

## Testing a non-deterministic component deterministically

Three techniques, in order of preference:

1. **Make the parts that can be deterministic, deterministic.** Routing is keyword-based
   with an explicit tie-break order. An agent that picks tools by vibes cannot be
   regression-tested.
2. **Assert on properties, not strings.** Category in an enum, tool in an allowlist,
   answer non-empty and above a word count, budgets respected.
3. **Score what is left**, with thresholds and reasons rather than equality.

The mock provider makes (1) and (2) exact. With a real model, (1) and (2) still hold —
the guard rails are in the workflow, not in the model — and (3) moves, which is why the
model-based backends run nightly and not on every pull request.
