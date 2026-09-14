# Architecture

## The shape of the problem

An AI feature is a pipeline, and a pipeline fails in more places than a form does:

```
USER -> UI -> API -> AGENT -> RETRIEVER -> LLM -> TOOLS -> RESPONSE
```

A traditional suite covers the first three arrows and stops. Everything downstream is
where AI-specific defects live: the retriever finds the wrong article, the generator
ignores a good one, the agent calls the right tool with the wrong arguments, a prompt
edit costs three points of faithfulness. None of those break a selector.

So the framework is built in two layers that answer different questions and never
substitute for one another.

## Layer A - deterministic

Everything that has a right answer. Fast, exact, reproducible, and allowed to block a
merge on its own.

| Concern | Where |
|---|---|
| User journeys, rendering, error states | `tests/ui/` |
| HTTP contracts, schemas, auth | `tests/api/` |
| Structured AI output (enums, required fields, types) | `tests/agents/test_state.py` |
| Tool allowlist and argument schemas | `src/agents/tools.py`, `tests/agents/test_tools.py` |
| Forbidden claims, PII, prompt leakage | `src/evaluation/deterministic.py` |
| Call and retry budgets | `tests/performance/`, `tests/agents/test_workflow.py` |

## Layer B - AI evaluation

Everything that has a *better or worse* answer rather than a right one. Scored against
thresholds in `config/evaluation.yaml`.

```
Playwright / agent
        |
   AI response
        |
        v
  AIEvaluator.evaluate()
        |
   +----+-----+--------------+
   |          |              |
builtin    deepeval        ragas
(lexical)  (LLM judge)   (RAG metrics)
   |          |              |
   +----+-----+--------------+
        |
   MetricScore  -> threshold -> pass/fail + reason
```

One facade (`src/evaluation/evaluator.py`), three interchangeable backends, one result
type. A test never knows which backend ran; it asserts on the score and prints the
reason on failure.

## Module map

```
src/
  app/            the application under test - FastAPI + a chat UI with data-testids
  llm/
    base.py       LLMProvider contract + LLMResponse (text, latency, tokens)
    mock_provider deterministic, context-grounded, with triggerable failures
    openai_provider  any OpenAI-compatible endpoint
    factory.py    registry; adding a provider is one register_provider() call
  rag/
    loader        markdown -> documents (LangChain Document when installed)
    chunker       heading-aware, then size-aware
    embeddings    deterministic hashed bag-of-words, or OpenAI embeddings
    retriever     TF-IDF weighted vector store (FAISS or numpy) + score floors
    pipeline      retrieve -> generate, returning the evidence alongside the answer
  agents/
    state.py      Pydantic AgentState + SupportResponse contract
    guardrails.py prompt-injection detection, applied before generation
    nodes.py      classify / route / retrieve / tool / generate / validate / retry / escalate / refuse
    graph.py      LangGraph runtime, plus an equivalent built-in executor
    tools.py      allowlist, argument schemas, the single execution gate
  evaluation/
    deterministic exact checks, no model
    judge         built-in metric implementations
    deepeval_metrics / ragas_metrics   optional model-based backends
    evaluator     the facade tests call
    collector     session-wide results, flushed to reports/
    regression    baseline comparison and prompt A/B
  observability/  Langfuse tracing, in-memory when disabled
```

## Key design decisions

**The provider is an interface, not an import.** Every layer above `src/llm/base.py`
depends on `LLMProvider`. That is what makes credential-free CI possible and what would
make swapping to a different vendor a factory change.

**The mock is a first-class implementation, not a stub.** It grounds answers in the
context it is given, refuses to answer about entities the context never mentions, and
can be told to time out. Those behaviours are what the evaluation layer is written to
detect, so the framework can demonstrate that its metrics *fire*, not just that they
pass.

**The agent state is a Pydantic model.** Every transition is therefore assertable on its
own, which is why `tests/agents/test_state.py` can test nodes individually and
`test_workflow.py` only has to cover the paths between them.

**Guardrails live in the graph, not the prompt.** An injected instruction is refused
before it reaches the generation prompt. Prompt hardening is still measured - separately,
in `evaluate_prompt_robustness` - because defence in depth means the prompt is no longer
the only control, and a test that only checks the system would stop being able to tell a
good prompt from a bad one.

**Two runtimes, one set of nodes.** LangGraph runs the workflow when the optional extra
is installed; an equivalent built-in executor runs it otherwise.
`test_langgraph_and_builtin_runtimes_agree` asserts they cannot diverge, so no agent test
depends on which one happened to be installed.

**Evidence travels with the answer.** `AgentState.evidence` and `RAGResult.chunks` carry
the retrieved context and tool output next to the response. That is what makes a failure
diagnosable rather than merely visible.

## Data flow for one answer

```
POST /api/chat
   -> SupportAgent.run(query, history)
      -> classify_node        LLM call 1, guarded by enum validation
      -> guardrails           injection? -> refuse, and stop
      -> route                escalate | retrieve | clarify | refuse
      -> retrieve_node        TF-IDF search, top-k, absolute + relative floor
      -> tool_node            one tool, through the authorisation gate
      -> generate_node        LLM call 2, prompt = active version
      -> validate_node        deterministic checks; pass | retry | escalate
   <- AgentState.to_public_dict()   internal reasoning excluded
```

Every run opens a trace span carrying the prompt version, provider, runtime, latency,
call counts and the final category - see [observability.md](observability.md).
