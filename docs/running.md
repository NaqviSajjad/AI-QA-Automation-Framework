# Running the project

Everything below runs with **no API key and no network**. `LLM_PROVIDER=mock` is the
default, and the mock is a real implementation of the provider interface.

---

## 0. Prerequisites

| Needed | Check | Notes |
|---|---|---|
| Python **3.11+** | `python3 --version` | 3.10 will fail — the code uses 3.11 syntax |
| pip | `python3 -m pip --version` | |
| Node (optional) | `npx --version` | only for Promptfoo |

On macOS, if `python3` reports 3.10 or older:

```bash
brew install python@3.12
python3.12 --version          # use this in place of `python3` below
```

---

## 1. First-time setup

```bash
cd ~/Documents/AI-QA-Automation-Framework

python3 -m venv .venv
source .venv/bin/activate                 # Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install -e ".[ai]"                    # core + LangChain / LangGraph / FAISS
python -m playwright install chromium     # ~150 MB, one time

cp .env.example .env                      # defaults already work
```

**Install options**

| Command | Gets you |
|---|---|
| `pip install -e .` | core only — Playwright, pytest, FastAPI, the agent's built-in runtime |
| `pip install -e ".[ai]"` | + LangChain, LangGraph, FAISS, OpenAI client ← **use this** |
| `pip install -e ".[ai,eval,obs,dev]"` | + DeepEval, RAGAS, Langfuse, ruff, mypy |

The `[ai]` extra is optional by design: without it the agent runs on an equivalent
built-in executor and the vector store falls back to exact numpy search. A test asserts
the two runtimes cannot diverge.

---

## 2. Verify the setup

Three commands, ~30 seconds total.

```bash
pytest -m smoke                  # critical path
pytest                           # everything
python -m scripts.quality_gate   # the CI gate, with an exit code
```

A healthy full run:

```
483 passed in ~18s
AI evaluation: 111/151 passed  metrics={'answer_relevancy': 0.97, 'faithfulness': 0.99, ...}
```

> `111/151` is expected, not a problem. Roughly forty of those evaluations are
> **negative controls** — deliberately bad answers fed in to prove each metric can
> actually fail. The pytest count (`483 passed`) is the real verdict.

And the gate:

```
[PASS] deterministic_tests            1.0000  >= 1.0
[PASS] answer_relevancy               0.9822  >= 0.85
[PASS] faithfulness                   1.0000  >= 0.85
...
AI QUALITY GATE: PASS
```

`echo $?` → `0` on pass, `1` on fail.

---

## 3. Everyday commands

`make help` lists them all. The ones that matter:

| Task | Make | Raw |
|---|---|---|
| Everything | `make test` | `pytest` |
| Critical path | `make smoke` | `pytest -m smoke` |
| Browser tests | `make ui` | `pytest tests/ui` |
| API contracts | `make api` | `pytest tests/api` |
| Answer quality | `make ai` | `pytest tests/ai` |
| Retrieval + generation | `make rag` | `pytest tests/rag` |
| Agent behaviour | `make agent` | `pytest tests/agents` |
| Security | `make security` | `pytest tests/security` |
| Prompt regression | `make prompt` | `pytest tests/prompts` |
| The 3 showcase scenarios | `make flagship` | `pytest -m flagship` |
| Lint + types | `make lint typecheck` | `ruff check . && mypy src api pages fixtures` |
| The lot, as CI runs it | `make all` | — |

Useful pytest flags:

```bash
pytest -k "billing"              # by name
pytest -x                        # stop at the first failure
pytest -n auto                   # parallel (pytest-xdist)
pytest -vv                       # full assertion output
pytest --lf                      # only what failed last time
```

---

## 4. Watching it run

Playwright is headless by default. To watch:

```bash
HEADLESS=false pytest tests/ui -k chatbot        # see the browser
HEADLESS=false SLOW_MO=400 pytest -m flagship    # slowed down, good for a demo
```

To use the app by hand:

```bash
uvicorn src.app.main:app --port 8000 --reload
open http://127.0.0.1:8000            # customer@example.com / Passw0rd!
```

The test suite starts its own server on a free port, so this is only for poking at it
yourself. Point the tests at an already-running instance with `APP_AUTO_START=false`.

---

## 5. The evaluation workflow

```bash
python -m scripts.run_evaluation                    # score all 35 cases
python -m scripts.run_evaluation --compare-prompts  # v1 vs v2, same dataset
python -m scripts.quality_gate                      # thresholds + baseline, exit code
python -m scripts.generate_report                   # the readable report
```

Outputs land in `reports/`:

| File | What |
|---|---|
| `evaluation_run.json` | every case, every score |
| `quality_gate.json` | each threshold check |
| `ai-quality-report.txt` | the human-readable report |
| `pytest-report.html` | the pytest HTML report |
| `junit.xml` | with `--junitxml=reports/junit.xml` |

### Re-recording the baseline

```bash
python -m scripts.run_evaluation --save-baseline
```

Do this **only** when a metric change is understood and intended, and say why in the
commit. Re-baselining to turn a build green is how a quality gate becomes decoration.

---

## 6. Cross-browser

```bash
python -m playwright install firefox webkit
pytest tests/ui tests/e2e --browser chromium --browser firefox --browser webkit
```

Chromium is what CI runs on every pull request; all three run nightly.

---

## 7. Running against a real model

```bash
pip install -e ".[eval]"

export LLM_PROVIDER=openai
export OPENAI_API_KEY=sk-...
export EVALUATOR_BACKEND=deepeval      # or ragas

pytest -m "llm or rag"
python -m scripts.run_evaluation
```

Expect lower and less stable scores than mock mode — that is the point of running it.
Tests marked `requires_mock_provider` skip automatically, because fault injection and
byte-identical determinism only mean something against the mock.

Any OpenAI-compatible endpoint works via `OPENAI_BASE_URL` (Azure, vLLM, Ollama's compat
layer, OpenRouter).

### Langfuse tracing

```bash
pip install -e ".[obs]"
export LANGFUSE_ENABLED=true
export LANGFUSE_PUBLIC_KEY=pk-... LANGFUSE_SECRET_KEY=sk-...
pytest -m flagship
```

With it off (the default) the tracer still records spans in memory, so the
failure-investigation workflow in `docs/observability.md` works without an account.

### Promptfoo

```bash
export OPENAI_API_KEY=sk-...
npx promptfoo@latest eval -c promptfooconfig.yaml
npx promptfoo@latest view
```

Calls a real model, so it belongs nightly or on your machine — never on every commit.

---

## 8. Configuration

| Where | For |
|---|---|
| `.env` | secrets and per-machine settings (never committed) |
| `config/config.yaml` | app, browser, LLM, RAG, agent settings |
| `config/evaluation.yaml` | every metric threshold and the quality gate |

Environment variables always win over YAML. The ones you will actually use:

```bash
LLM_PROVIDER=mock|openai
EVALUATOR_BACKEND=builtin|deepeval|ragas|auto
PROMPT_VERSION=customer_support_v1|customer_support_v2
HEADLESS=true|false
SLOW_MO=0
LOG_LEVEL=INFO|ERROR          # ERROR quiets the JSON logs
APP_AUTO_START=true|false
```

---

## 9. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `SyntaxError` on `str \| None` | Python 3.10 or older | use 3.11+ |
| `Executable doesn't exist … chromium` | browsers not installed | `python -m playwright install chromium` |
| `ModuleNotFoundError: src` | not installed, or wrong directory | `pip install -e ".[ai]"` from the project root |
| `Application under test failed to start` | port or import error | run `uvicorn src.app.main:app --port 8000` to see the real traceback |
| `no baseline` in prompt tests | `data/baselines/baseline.json` missing | `python -m scripts.run_evaluation --save-baseline` |
| Everything slow on first run | the vector index is built once per session | subsequent tests reuse it |
| Gate fails only on `contextual_precision` | retrieval drifted | check `reports/evaluation_run.json` → `sources` per case |
| DeepEval/RAGAS silently not used | package or key missing | `pytest` header prints the resolved backend |
| Logs too noisy | JSON logging | `LOG_LEVEL=ERROR pytest` |

The pytest header always tells you what is actually running:

```
AI QA framework: provider=mock prompt=customer_support_v2 evaluator=builtin langfuse=off
```

---

## 10. Putting it on GitHub

```bash
cd ~/Documents/AI-QA-Automation-Framework
git init && git add -A
git commit -m "AI-QA-Automation-Framework: Playwright + AI evaluation framework"
gh repo create AI-QA-Automation-Framework --public --source=. --push
```

`.gitignore` already excludes `.env`, `reports/`, caches and virtualenvs. The GitHub
Actions workflow runs on the first push: lint, types, tests, evaluation, quality gate,
and it publishes the AI quality report to the job summary — no secrets required.

Before pushing, confirm nothing sensitive is staged:

```bash
git status --short | grep -E "\.env$|key|secret" || echo "clean"
```
