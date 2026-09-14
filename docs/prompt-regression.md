# Prompt regression

## The failure being defended against

Someone improves a prompt. Every functional test still passes — the button works, the API
returns 200, the schema validates. Nobody notices that faithfulness dropped from 0.98 to
0.79 and the assistant started padding answers with reassurance nothing supports.

Traditional automation cannot see that. A baseline and a tolerance can.

## The rule

```
baseline_score - current_score > allowed_drop   ->   FAIL
```

`allowed_drop` is 0.03, in `config/evaluation.yaml`. For lower-is-better metrics
(`hallucination`) the comparison is inverted, which `MetricDelta` handles explicitly.

The tolerance is not laziness. Model-based metrics wobble between runs; a zero-tolerance
gate fails randomly and gets switched off within a week, at which point it protects
nothing. `test_small_movements_inside_the_tolerance_are_accepted` pins that intent.

## The baseline

```bash
python -m scripts.run_evaluation --save-baseline
```

`data/baselines/baseline.json` records the metrics plus the context they were measured in
— prompt version, provider, evaluator backend, case count, timestamp. A baseline without
that context is a number with no meaning.

```json
{
  "generated_at": "2026-09-01T01:06:47+00:00",
  "prompt_version": "customer_support_v2",
  "llm_provider": "mock",
  "evaluator_backend": "auto",
  "cases": 35,
  "metrics": {
    "answer_relevancy": 0.9822,
    "faithfulness": 1.0,
    "correctness": 0.9689,
    "hallucination": 0.0,
    "contextual_precision": 0.8444,
    "contextual_recall": 0.9764,
    "task_completion": 0.9829,
    "tool_correctness": 1.0,
    "security_pass_rate": 1.0,
    "prompt_robustness": 1.0,
    "deterministic_pass_rate": 1.0
  }
}
```

Re-baseline **only** when a change is understood and intended, and say so in the commit.
Re-baselining to make a red build green is how a quality gate becomes decoration.

The comparison output names the metric and the direction:

```
Metric                    Baseline   Current      Diff  Status
--------------------------------------------------------------
answer_relevancy            0.9822    0.9822   +0.0000  STABLE
faithfulness                1.0000    0.7865   -0.2135  REGRESSION
```

## Prompt A/B

Two versions live in `prompts/`, and the framework runs the *same* dataset through both:

```bash
python -m scripts.run_evaluation --compare-prompts
```

| Metric | `customer_support_v1` | `customer_support_v2` |
|---|---|---|
| answer relevancy | 0.8739 | 0.9822 |
| faithfulness | 0.7865 | 1.0000 |
| hallucination | 0.2135 | 0.0000 |
| task completion | 0.9489 | 0.9829 |
| prompt robustness | 0.8333 | 1.0000 |

The difference is caused by three rules v2 has and v1 does not: ground every factual
claim, never claim an action has been taken, finish with a concrete next step — plus the
non-disclosure rule. The numbers show exactly those three things: less grounding, more
hallucination, fewer next steps, and instruction leakage under injection.

`test_v2_is_not_worse_than_v1_on_any_metric` turns this into a gate, so replacing the
active prompt with something weaker fails the build.

## Measuring the prompt separately from the system

An important subtlety. The workflow guardrail (`src/agents/guardrails.py`) refuses an
injected message *before* generation, so at the system level **both** prompt versions are
safe — which is exactly what defence in depth should give you.

But it also means an agent-level security test can no longer distinguish a hardened prompt
from a weak one. So prompt robustness is measured separately, by sending the adversarial
dataset straight through each prompt template to the model with the guardrail bypassed
(`evaluate_prompt_robustness`). That keeps the signal, and shows what the system would
fall back to if the guardrail were ever bypassed.

Two tests state both halves:

* `test_the_workflow_guardrail_protects_both_prompt_versions` — the system is safe either
  way;
* `test_the_weaker_prompt_leaks_when_the_guardrail_is_bypassed` — and v1 is still the
  worse prompt.

## Promptfoo

`promptfooconfig.yaml` covers the interactive half — a matrix of prompt × model × case
that a human reads:

```bash
npx promptfoo@latest eval -c promptfooconfig.yaml
npx promptfoo@latest view
```

It calls a real model, so it belongs in the nightly pipeline or on a developer's machine,
never on every pull request. `tests/prompts/test_prompt_regression.py` validates the
config structurally so it cannot silently rot.

Division of labour:

| | Promptfoo | This framework |
|---|---|---|
| Purpose | explore, compare, iterate | gate, block, regress |
| Runs | on demand, nightly | every commit |
| Needs a key | yes | no |
| Output | an HTML matrix for a human | an exit code for CI |
