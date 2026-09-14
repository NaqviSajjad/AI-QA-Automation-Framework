.PHONY: help install install-full browsers test smoke ui api ai rag agent security prompt perf flagship \
        cross-browser lint typecheck evaluate baseline compare gate report promptfoo clean all

PYTHON ?= python
PYTEST ?= $(PYTHON) -m pytest

help:
	@echo "Setup"
	@echo "  make install         core + LangChain/LangGraph/FAISS"
	@echo "  make install-full    also DeepEval, RAGAS, Langfuse, dev tools"
	@echo "  make browsers        install Playwright browsers"
	@echo ""
	@echo "Tests"
	@echo "  make test            everything (mock mode, no credentials)"
	@echo "  make smoke           critical path only"
	@echo "  make ui api ai rag agent security prompt perf flagship"
	@echo "  make cross-browser   chromium + firefox + webkit"
	@echo ""
	@echo "Quality"
	@echo "  make lint typecheck"
	@echo "  make evaluate        score the whole dataset"
	@echo "  make compare         prompt v1 vs v2"
	@echo "  make baseline        record a new baseline (do this deliberately)"
	@echo "  make gate            quality gate, exit code"
	@echo "  make report          render the AI quality report"
	@echo "  make all             lint + types + tests + evaluate + gate + report"

install:
	$(PYTHON) -m pip install -e ".[ai]"

install-full:
	$(PYTHON) -m pip install -e ".[ai,eval,obs,dev]"

browsers:
	$(PYTHON) -m playwright install chromium

test:
	$(PYTEST) --junitxml=reports/junit.xml

smoke:
	$(PYTEST) -m smoke

ui:
	$(PYTEST) tests/ui
api:
	$(PYTEST) tests/api
ai:
	$(PYTEST) tests/ai
rag:
	$(PYTEST) tests/rag
agent:
	$(PYTEST) tests/agents
security:
	$(PYTEST) tests/security
prompt:
	$(PYTEST) tests/prompts
perf:
	$(PYTEST) tests/performance
flagship:
	$(PYTEST) -m flagship

cross-browser:
	$(PYTEST) tests/ui tests/e2e --browser chromium --browser firefox --browser webkit

lint:
	ruff check .

typecheck:
	mypy src api pages fixtures

evaluate:
	$(PYTHON) -m scripts.run_evaluation

compare:
	$(PYTHON) -m scripts.run_evaluation --compare-prompts

baseline:
	$(PYTHON) -m scripts.run_evaluation --save-baseline

gate:
	$(PYTHON) -m scripts.quality_gate

report:
	$(PYTHON) -m scripts.generate_report

promptfoo:
	npx --yes promptfoo@latest eval -c promptfooconfig.yaml

clean:
	rm -rf reports/* .pytest_cache .mypy_cache .ruff_cache test-results
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	@touch reports/.gitkeep

all: lint typecheck test evaluate gate report
