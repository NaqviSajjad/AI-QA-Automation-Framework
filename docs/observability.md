# Observability

## Why a test framework needs tracing

A failing UI test tells you the answer was bad. It does not tell you *why*. In a pipeline
of retrieval, prompting, generation and tool calls, "why" has at least six candidates, and
guessing is expensive.

Tracing turns a failure into a diagnosis.

## What is captured

Every agent run opens a span carrying:

| Field | Use |
|---|---|
| `trace_id` | correlates the UI failure, the API log and the agent run |
| `conversation_id` | groups the turns of one conversation |
| prompt version | which prompt produced this — the first question after a regression |
| provider, model, runtime | mock vs real, LangGraph vs built-in |
| input / output | the question and the answer |
| latency, llm_calls, tool_calls, retries | cost and budget |
| scores | evaluation results attached to the same span |

## Optional means optional

```env
LANGFUSE_ENABLED=false     # the default
```

With Langfuse disabled the tracer is not a no-op — it still records spans **in memory**.
That is what lets the failure-investigation workflow below be exercised in CI with no
external service, and it means `tests/agents/test_workflow.py` can assert on the trace:

```python
def test_a_trace_is_recorded_for_every_run(support_agent, tracer):
    state = support_agent.run("How long does a refund take?")
    span = tracer.find(state.trace_id) or tracer.last()
    assert span.metadata["prompt_version"] == settings.prompts.active_version
    assert span.metrics["llm_calls"] >= 1
```

Enable it with credentials and the same code path writes to Langfuse; every write is
wrapped so an observability outage can never fail a test.

Structured JSON logging (`src/utils/logging.py`) carries the same identifiers, so CI logs
can be grepped by `conversation_id` or `trace_id`.

## The failure-investigation workflow

```
Playwright test failed
        |
        v
AI response captured        <- what did the customer actually see?
        |
        v
Trace / AgentState          <- node_path: which branch ran?
        |
        v
Retriever                   <- retrieved_sources: right article?
        |                      NO  -> RETRIEVAL problem. Stop here.
        v
Retrieved context           <- is the answer's claim even present?
        |
        v
Prompt                      <- which version? did it change recently?
        |
        v
LLM                         <- provider, model, latency, finish_reason
        |
        v
Tools                       <- right tool? right arguments? did it error?
        |
        v
Evaluation scores           <- which metric failed, and by how much?
```

Read top-down and stop at the first thing that is wrong. The order matters: judging
generation on the wrong context tells you nothing, which is why
`diagnose_rag_failure` reports `RETRIEVAL` when both halves look bad.

## Mapping symptoms to causes

| Symptom | Likely cause | Where to look |
|---|---|---|
| `contextual_precision` low, faithfulness fine | wrong retrieval | `retrieved_sources`, chunking, IDF weighting |
| Retrieval fine, faithfulness low | generator ignoring context | prompt version, `extract_claims` output |
| Answer correct but irrelevant | over-retrieval, padding | `contextual_relevancy`, the relative score floor |
| `tool_correctness` 0 | routing | `state.category`, `TOOL_BY_CATEGORY` |
| Tool error in the span | bad arguments | `ToolCall.arguments`, the Pydantic schema |
| Hallucination 1.0 with faithfulness high | forbidden claim asserted | `find_forbidden_claims` |
| Everything scores low at once | prompt regression | baseline comparison, `--compare-prompts` |
| Latency spike, budgets exceeded | retry loop | `retry_count`, `node_path` |

## Worked example

`test_answer_is_relevant_and_grounded[CS-021]` fails.

1. The failure message already names the metric: `contextual_relevancy=0.250`.
2. `sources: ['accounts.md', 'subscriptions.md', 'technical_support.md']` — the password
   reset question pulled in two unrelated articles.
3. `generation: 1.000 (3/3 claims grounded)` — the generator was faithful to what it was
   given.
4. Verdict: **RETRIEVAL**. Do not touch the prompt.
5. Fix: the retriever was returning a fixed top-k regardless of score. A relative floor at
   45% of the best hit trims the near-misses.

That is a real fix from this repository's own development, and it is why the relative
floor exists.

## The published run

`scripts/generate_report.py` renders the report from files that an actual run produced —
`reports/evaluation_run.json`, `reports/quality_gate.json`, `reports/junit.xml`. A section
with no data prints `not run` rather than a plausible number. A report that might be
fabricated is worse than no report.
