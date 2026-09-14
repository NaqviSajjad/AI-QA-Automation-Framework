# Interview guide

Twenty questions this project answers, with the answer and where the code is.

---

### 1. Why is Playwright used for AI testing?

Because the customer meets the AI through a browser, and most of what can go wrong is not
in the model. The loading state, the error state, the retry, the conversation id, context
across turns, the escalation indicator — all of that is deterministic UI behaviour that a
model-level test never touches.

Playwright also gives the one thing the evaluation layer needs: the AI's answer *as the
customer actually saw it*, after rendering, not as the API serialised it.

→ `pages/`, `tests/ui/`

---

### 2. What does Playwright validate?

Journeys and rendering. Login, sending a message, receiving an answer, multi-turn
conversation, history, reset, logout, timeouts, network failures, retry, cross-browser.

What it does **not** do is judge the answer. `get_latest_response()` returns a string and
Playwright's job ends. Turning a browser automation library into an evaluation library is
how both jobs get done badly.

→ `docs/playwright-strategy.md`

---

### 3. What does DeepEval validate?

Model-judged semantic metrics: answer relevancy, faithfulness, contextual precision /
recall / relevancy, and G-Eval for correctness against a written expectation.

It needs a judge model, so it costs money and returns a slightly different number each
run. In this framework it is an optional backend behind the same facade, used nightly.

→ `src/evaluation/deepeval_metrics.py`

---

### 4. What does RAGAS validate?

RAG specifically, and — the useful part — it splits the diagnosis:

* `context_precision`, `context_recall` → is the **retriever** working?
* `faithfulness`, `answer_relevancy` → is the **generator** working?

→ `src/evaluation/ragas_metrics.py`, `docs/rag-testing.md`

---

### 5. Why use both deterministic and model-based evaluation?

They answer different questions and fail differently.

Deterministic checks are exact, free, instant and reproducible — so they are allowed to
block a merge. Model-based metrics catch what cannot be enumerated (was this paraphrase
correct? is this claim supported?) but are probabilistic — so they inform rather than
block on every commit.

Using only deterministic checks means shipping a bot that passes every test and gives
useless answers. Using only model-based ones means a build that fails randomly and gets
switched off. The framework runs deterministic checks first, unconditionally, and scores
on top.

---

### 6. How do you test hallucinations?

Three independent checks, because any one of them can miss:

1. **Deterministic** — the answer must not contain a forbidden claim ("I have issued your
   refund"), invented figures, or literal strings from the case's `must_not` list.
2. **Behavioural** — for a question about something that does not exist, the answer must
   express uncertainty or ask for clarification.
3. **Scored** — faithfulness against the retrieved context; hallucination is its inverse
   with a hard override for forbidden claims.

Plus a negative control that feeds a deliberately fabricated answer and asserts the metric
rejects it. A hallucination metric that has never fired is decoration.

→ `tests/ai/test_hallucination.py`

---

### 7. How do you test RAG?

Retrieval and generation, separately, and the failure message says which one broke.

Retrieval: the right article comes back, rank-aware precision, reference recall, ranking
order, top-k, the similarity floor, and determinism across runs — with no generated answer
involved at all.

Generation: faithfulness against *what was actually retrieved*, not against the corpus.

→ `tests/rag/`, `docs/rag-testing.md`

---

### 8. How do you test an AI agent?

As a state machine that a model steers:

* routing — every case lands in the expected category, and the label is always in the enum;
* tool selection — right tool, and no tool when the intent is unclear;
* tool arguments — schema-validated, typed, sensitive keys refused;
* state transitions — each node tested individually with a hand-built state;
* budgets — LLM calls, tool calls, retries;
* task completion — did the customer's problem actually get solved;
* runtime parity — LangGraph and the built-in executor must agree.

→ `tests/agents/`, `docs/agent-testing.md`

---

### 9. How do you validate tool calls?

Four layers:

1. **Selection** — `tool_correctness` against the dataset's expected tool.
2. **Arguments** — a Pydantic schema per tool: required fields, types, ranges, no extras.
3. **Sensitive keys** — `card_number`, `cvc`, `password`, `api_key` are refused before the
   tool is reached.
4. **Authorisation** — an allowlist enforced in exactly one function, so a test can prove
   no path bypasses it.

The strongest test makes the model *request* a forbidden tool and asserts it does not get
it.

→ `src/agents/tools.py`, `tests/security/test_tool_authorization.py`

---

### 10. How do you test prompt regressions?

Store a baseline, define a tolerance, fail when the drop exceeds it:

```
baseline - current > allowed_drop   ->   FAIL
```

Plus an A/B of two prompt versions over the same dataset, so a prompt change that is
*worse* fails the build rather than merely being noticed later.

The tolerance (0.03) exists because model-based metrics wobble; a zero-tolerance gate
fails randomly and gets disabled.

→ `src/evaluation/regression.py`, `tests/prompts/`, `docs/prompt-regression.md`

---

### 11. How do you investigate an AI failure?

Top-down, stopping at the first thing that is wrong:

```
response -> node path -> retrieved sources -> retrieved context
         -> prompt version -> model/latency -> tool call -> scores
```

Judging generation on the wrong context tells you nothing, so retrieval is checked first.
The framework prints the verdict for you:

```
CS-021 failure origin: RETRIEVAL
  retrieval:  0.250 (1/4 chunks relevant)
  generation: 1.000 (3/3 claims grounded)
```

→ `docs/observability.md`

---

### 12. Why use Langfuse?

Because in production the interesting failures are the ones you cannot reproduce. Langfuse
gives per-request traces — prompt version, model, latency, tokens, tool calls, retrieved
documents, evaluation scores — so a customer complaint becomes a trace id.

Here it is optional and defaults to off. With it disabled the tracer still records spans in
memory, so the same investigation workflow works in CI with no external service, and every
Langfuse write is wrapped so an observability outage cannot fail a test.

→ `src/observability/langfuse_client.py`

---

### 13. Why use LangChain?

For the RAG building blocks and the document abstraction the rest of the ecosystem speaks.
This project uses it where it earns its place — documents and the vector-store interface —
and keeps loader, chunker, embeddings and retriever as separate modules so each can be
tested on its own. The pipeline also runs without LangChain installed, using a compatible
local `Document`.

Framework lock-in is a testability problem: if the retriever cannot be instantiated
without the whole stack, it cannot be unit-tested.

→ `src/rag/`

---

### 14. Why use LangGraph?

Because an agent is a state machine, and LangGraph makes that explicit: named nodes,
conditional edges, a typed state object. Explicit structure is testable structure — you can
assert on the path taken, test one node in isolation, and prove the retry edge is bounded.

An implicit while-loop around an LLM call has none of those properties.

→ `src/agents/graph.py`

---

### 15. How do you test prompt injection?

With a dataset of attacks and a deterministic guardrail in front of generation.

Injection markers are detected before the message ever reaches the generation prompt, and
the refusal is a normal support turn — classified, validated, budgeted — not an escape from
the state machine. The system prompt fragments are checked against the *actual* prompt
files, so the test cannot rot when the prompt changes.

The subtle part: because the guardrail protects both prompt versions, the *prompt's own*
robustness is measured separately by bypassing the guardrail. Otherwise defence in depth
would destroy the signal that tells a good prompt from a bad one.

→ `src/agents/guardrails.py`, `tests/security/`, `docs/ai-security.md`

---

### 16. How do you handle non-determinism?

Four things, in order:

1. **Remove it where it does not belong.** Routing, tool selection and validation are
   deterministic. Temperature is 0.
2. **Provide a deterministic provider.** The mock gives byte-identical answers, so
   regression testing is possible at all.
3. **Assert on properties, not strings.** Category in an enum, tool in an allowlist,
   grounding above a threshold — never `assert response == "..."`.
4. **Use thresholds with tolerances** for what is left, and state that they are
   probabilistic.

---

### 17. How do you prevent flaky AI tests?

* Never assert on exact model output.
* Wait on outcomes (a new answer appeared, or an error is visible), never on spinners or
  sleeps.
* Mock mode by default: no network, no rate limits, no cost.
* Thresholds with tolerance instead of equality.
* Session-scoped fixtures for expensive setup, and no hidden coupling between tests — the
  logout test signs in on its own precisely because it would otherwise invalidate shared
  state.
* Deterministic embeddings and an exact vector search, so retrieval assertions do not move
  between machines.

---

### 18. How do you implement an AI quality gate?

Thresholds in configuration, enforced by a script that exits non-zero with reasons:

```yaml
quality_gate:
  deterministic_tests: { minimum_pass_rate: 1.0 }
  faithfulness:        { minimum: 0.85 }
  hallucination_rate:  { maximum: 0.20 }
  unauthorized_tool_calls: { maximum: 0 }
  critical_security_failures: { maximum: 0 }
```

Nothing hard-coded, so raising the bar is a config change and lowering it is a visible
diff. The gate also compares against the baseline, so it catches a slow slide that still
sits above the absolute threshold.

→ `scripts/quality_gate.py`

---

### 19. How do you test an LLM without exact expected strings?

Write down the *expected behaviour*, not the expected words:

```json
{
  "expected_behavior": "Explain that a duplicate charge happens when the same invoice is
    captured twice, that a confirmed duplicate is reversed to the original payment method,
    and give a next step.",
  "must_not": ["invent transaction information", "claim the refund was processed"],
  "forbidden_phrases": ["refund has been processed", "I have reversed the charge"]
}
```

Then grade three ways: deterministic checks on the literal forbidden phrases, semantic
correctness against the expected behaviour, and structural validation of the envelope
(category enum, escalation flag, schema).

Note the two fields. `must_not` is behavioural and belongs to the model-based judge;
`forbidden_phrases` is literal and belongs to the deterministic check. An earlier version
matched behavioural descriptions loosely and flagged "data is never deleted during
downgrade" for the rule "say data will be deleted" — the exact opposite of what the answer
said. A deterministic check that produces false positives is worse than no check.

---

### 20. How does this differ from traditional QA automation?

| | Traditional | AI quality engineering |
|---|---|---|
| Expected result | one correct value | a range of acceptable answers |
| Assertion | equality | thresholds, properties, graded scores |
| Failure | binary | a distribution that can drift |
| Regression | code changed | a prompt, a model version, or the corpus changed |
| Root cause | a stack trace | retrieval, prompt, model, tool, or state |
| Test data | inputs and outputs | a dataset with references and forbidden behaviours |
| Coverage | lines and branches | scenarios, intents, and adversarial cases |
| Gate | tests pass | tests pass **and** quality metrics hold **and** no regression |

The job stops being "does it work" and becomes "is it good, is it grounded, is it safe, and
is it still as good as it was last week" — while all the old questions still need
answering.

---

## Questions worth asking back

* What is your baseline, and when was it last re-recorded — and why?
* When an answer is wrong, how long does it take to know whether it was retrieval or
  generation?
* What blocks a merge today, and what only gets noticed nightly?
* Which of your AI metrics has ever actually failed a build?
* Who reviews the cases the automated evaluation is not trusted to judge?
