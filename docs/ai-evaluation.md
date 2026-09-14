# AI evaluation

## The rule

> Deterministic assertions first. Model-based judgement on top. Never the other way round.

Anything that can be checked exactly *is* checked exactly — enums, schemas, tool
allowlists, forbidden claims, PII patterns, call budgets. Only what genuinely has no
right answer is scored: relevance, grounding, semantic correctness, completion.

A scored metric never replaces a deterministic one. It sits above it.

## Metrics

All defined in `src/evaluation/judge.py`, thresholds in `config/evaluation.yaml`.

### `answer_relevancy` (≥ 0.85)

Three observable components:

* **coverage** — how much of the question's *important* content the answer picks up.
  Question terms are weighted by inverse document frequency against the support corpus,
  so "trial" counts and "more" and "time" do not. Also satisfied by a single sentence that
  directly answers the question, because a good answer need not echo the whole question.
* **on_topic** — the share of *factual* sentences connected to the question or its intent.
  Greetings and closing next steps are excluded: penalising them would score answer
  structure rather than relevance, and every well-formed support answer has both.
* **actionable** — a support answer that states policy and stops is not a resolution.

### `faithfulness` (≥ 0.85)

The share of factual claims supported by the retrieved context (plus any tool result —
tool output is evidence too).

Claim extraction matters more than the scoring. Questions, courtesy openers, offers of
help ("I can check…") and self-referential process statements are not factual assertions,
so they are not scored. An answer that asserts nothing — a clarification, or an honest
"I don't know" — is faithful by definition, and RAGAS treats it the same way.

Each claim is normalised by stripping a short leading clause ("Looking at the records on
your account, …") before matching, because the framing is the assistant's words, not the
claim.

### `hallucination` (≤ 0.20)

`1 - faithfulness`, with a hard override: if the answer contains a forbidden claim
("I have issued your refund"), the score is 1.0 regardless.

The forbidden-claim detector has a negation guard. Support articles legitimately contain
"an agent must not state that a refund has been processed", and an earlier version flagged
that sentence — the exact opposite of what it said. A deterministic check that produces
false positives is worse than no check, because engineers learn to ignore it.

### `correctness` (≥ 0.80)

Semantic equivalence to the case's `expected_behavior`, never string equality:

```
Expected: explain that duplicate billing should be investigated and give next steps
Actual:   a semantically equivalent answer
Result:   PASS
```

Scored as IDF-weighted recall of the reference's content plus embedding similarity. With
`EVALUATOR_BACKEND=deepeval` this becomes G-Eval — an LLM grading against the same
criteria.

**Reference-based metrics are only as good as the reference.** Several dataset entries
were rewritten during development because the expected behaviour described an outcome the
correct answer did not use the words for. That is a real lesson, not a workaround: if you
cannot write the reference, you cannot grade the answer.

### Retrieval metrics

* `contextual_relevancy` (≥ 0.50) — share of surviving chunks that are on topic.
  Retrieval over-fetches on purpose and then trims with a relative score floor, so a
  near-miss in the tail is normal. Calibrated against this corpus; re-check it when the
  corpus grows.
* `contextual_precision` (≥ 0.80) — mean average precision over the ranked sources.
  MAP rewards getting the right article *high* without demanding every slot be relevant.
  Ground truth is a **set** of acceptable sources, because a question about mid-cycle plan
  changes is legitimately answered by both `subscriptions.md` and `billing.md`.
* `contextual_recall` (≥ 0.80) — how much of the reference context the retrieval covers.

### Agent metrics

* `tool_correctness` (≥ 0.90) — exact match against the expected tool.
* `task_completion` (≥ 0.85) — outcome achieved: right behaviour, an actionable next
  step, and the correct escalation decision. An answer can be relevant and faithful and
  still leave the customer with nothing to do.

## Backends

| `EVALUATOR_BACKEND` | Needs | Deterministic | Used for |
|---|---|---|---|
| `builtin` | nothing | yes | pull-request gate |
| `deepeval` | `.[eval]` + API key | no | nightly, deeper semantics |
| `ragas` | `.[eval]` + API key | no | nightly RAG diagnosis |
| `auto` | — | depends | picks the best usable backend |

Every backend returns the same `MetricScore`, so a test never knows which one ran:

```python
evaluation = ai_evaluator.evaluate(
    question=case.user_query,
    answer=answer,
    contexts=state.evidence,
    reference=case.expected_behavior,
    metrics=["answer_relevancy", "faithfulness", "correctness", "hallucination"],
)
assert evaluation.passed, evaluation.explain()
```

## Why the default backend is deterministic

An LLM judge costs money and returns a slightly different number each run. Both are fine
for investigation and fatal for a gate — a CI check that fails randomly gets disabled.
The built-in scorers implement the same metric *definitions* with a reproducible scorer,
so the pull-request gate is free, offline and stable, and the nightly run adds semantic
depth on top.

The trade is stated plainly: the built-in scorers measure lexical grounding and coverage,
not deep semantics. A paraphrase that shares no vocabulary with its reference will score
lower than it deserves. When the built-in and model-based backends disagree, that
disagreement is itself worth looking at.

## Not trusting the judge

LLM-as-a-judge has known failure modes, and none of them are hypothetical:

* **evaluator bias** — judges prefer verbose, confident, self-similar answers;
* **model limitations** — the judge can be wrong about the domain;
* **non-determinism** — the same pair scores differently across runs;
* **prompt sensitivity** — rewording the rubric moves the numbers;
* **judge disagreement** — two judges, two verdicts.

So the framework combines four things:

```
Deterministic assertions      exact, blocking
        +
Reference-based evaluation    graded against a written expectation
        +
LLM-as-a-judge                semantic depth, nightly
        +
Human review                  critical scenarios: refunds, escalation, security
```

## Making metrics prove they can fail

Every metric family has a **negative control** — a test that feeds a deliberately bad
answer and asserts the metric rejects it:

* `test_a_grounded_answer_scores_above_an_ungrounded_one`
* `test_the_hallucination_metric_actually_fires`
* `test_an_off_topic_answer_is_scored_down`
* `test_the_regression_gate_actually_fires`
* `test_the_leakage_detector_actually_detects_leakage`
* `test_the_pii_detector_actually_detects_pii`

A suite that only ever shows green has not been shown to detect anything. This is the
cheapest insurance in the whole framework.

## Choosing metrics per case

`scripts/run_evaluation.py::_metrics_for` decides which metrics apply:

* an answering case gets everything;
* an uncertainty case is scored on faithfulness, hallucination, correctness and completion
  — **not relevancy**, because a correct refusal deliberately does not restate the
  question;
* a clarification case is scored on correctness and completion only.

Applying every metric to every case produces a number that looks rigorous and means
nothing, and it fails the *right* behaviour.
