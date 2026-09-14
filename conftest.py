"""Root pytest configuration.

Registers every fixture module as a plugin, keeps Playwright artefacts under
`reports/`, and - at the end of the session - flushes the collected evaluation
results so the quality gate and the AI quality report describe exactly the run
that just happened.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation import collector  # noqa: E402
from src.utils.config import settings  # noqa: E402
from src.utils.logging import configure_logging  # noqa: E402

pytest_plugins = [
    "fixtures.app_server",
    "fixtures.browser",
    "fixtures.api",
    "fixtures.llm",
    "fixtures.rag",
    "fixtures.evaluation",
]


def pytest_configure(config: pytest.Config) -> None:
    configure_logging()
    settings.reports_path.mkdir(parents=True, exist_ok=True)
    (settings.reports_path / "artifacts").mkdir(parents=True, exist_ok=True)
    config.stash[_RUN_META] = {
        "llm_provider": settings.llm.provider,
        "prompt_version": settings.prompts.active_version,
        "langfuse_enabled": settings.observability.langfuse_enabled,
    }


_RUN_META = pytest.StashKey[dict]()


def pytest_report_header(config: pytest.Config) -> list[str]:
    meta = config.stash.get(_RUN_META, {})
    from src.evaluation.evaluator import get_evaluator

    return [
        f"AI QA framework: provider={meta.get('llm_provider')} "
        f"prompt={meta.get('prompt_version')} "
        f"evaluator={get_evaluator().backend} "
        f"langfuse={'on' if meta.get('langfuse_enabled') else 'off'}",
    ]


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Persist evaluation results for the quality gate and the report."""
    if collector.results() or collector.events():
        payload = collector.flush()
        summary = payload["summary"]
        print(
            f"\nAI evaluation: {summary['passed']}/{summary['evaluations']} passed  "
            f"metrics={summary['metrics']}"
        )


@pytest.fixture(scope="session", autouse=True)
def _reset_collector() -> None:
    collector.clear()


# --------------------------------------------------------------------------- #
# Playwright artefact configuration
# --------------------------------------------------------------------------- #
def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--eval-backend",
        action="store",
        default=None,
        help="Override the evaluation backend (builtin | deepeval | ragas | auto)",
    )


@pytest.fixture(scope="session", autouse=True)
def _apply_eval_backend(request: pytest.FixtureRequest) -> None:
    backend = request.config.getoption("--eval-backend")
    if backend:
        from src.evaluation.evaluator import get_evaluator

        get_evaluator(backend, refresh=True)
