"""Centralised, typed configuration.

Precedence: environment variables > config/*.yaml > code defaults.
Loaded once and cached so every layer (Playwright fixtures, API client, agent,
evaluators) reads exactly the same settings.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"


def _load_dotenv() -> None:
    """Load .env if python-dotenv is available; silently skip otherwise."""
    env_file = PROJECT_ROOT / ".env"
    if not env_file.exists():
        return
    try:
        from dotenv import load_dotenv

        load_dotenv(env_file, override=False)
    except ImportError:  # pragma: no cover - optional dependency
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())


def _yaml(name: str) -> dict[str, Any]:
    path = CONFIG_DIR / name
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _env_str(key: str, default: str) -> str:
    value = os.getenv(key)
    return value if value else default


def _env_bool(key: str, default: bool) -> bool:
    value = os.getenv(key)
    if not value:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(key: str, default: int) -> int:
    value = os.getenv(key)
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_float(key: str, default: float) -> float:
    value = os.getenv(key)
    if not value:
        return default
    try:
        return float(value)
    except ValueError:
        return default


# --------------------------------------------------------------------------- #
# Typed sections
# --------------------------------------------------------------------------- #
class AppConfig(BaseModel):
    base_url: str = "http://127.0.0.1:8000"
    api_base_url: str = "http://127.0.0.1:8000/api"
    auto_start: bool = True
    startup_timeout_s: int = 30


class AuthConfig(BaseModel):
    email: str = "customer@example.com"
    password: str = "Passw0rd!"


class PlaywrightConfig(BaseModel):
    headless: bool = True
    browser: str = "chromium"
    viewport: dict[str, int] = Field(default_factory=lambda: {"width": 1280, "height": 900})
    default_timeout_ms: int = 10000
    navigation_timeout_ms: int = 15000
    slow_mo_ms: int = 0
    video: str = "retain-on-failure"
    screenshot: str = "only-on-failure"
    trace: str = "retain-on-failure"


class LLMConfig(BaseModel):
    provider: str = "mock"
    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_tokens: int = 600
    timeout_s: int = 30
    max_retries: int = 2
    api_key: str | None = None
    base_url: str | None = None


class RAGConfig(BaseModel):
    knowledge_base_dir: str = "knowledge-base"
    chunk_size: int = 700
    chunk_overlap: int = 120
    top_k: int = 4
    min_similarity: float = 0.05
    relative_floor: float = 0.45
    embedding_backend: str = "deterministic"
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536


class AgentConfig(BaseModel):
    max_retries: int = 2
    max_tool_calls: int = 3
    max_llm_calls: int = 5
    escalation_priorities: list[str] = Field(default_factory=lambda: ["high", "critical"])
    allowed_tools: list[str] = Field(default_factory=list)


class PromptConfig(BaseModel):
    active_version: str = "customer_support_v2"
    directory: str = "prompts"


class ObservabilityConfig(BaseModel):
    langfuse_enabled: bool = False
    service_name: str = "ai-qa-automation-framework"
    public_key: str | None = None
    secret_key: str | None = None
    host: str = "https://cloud.langfuse.com"


class Settings(BaseModel):
    app: AppConfig = Field(default_factory=AppConfig)
    auth: AuthConfig = Field(default_factory=AuthConfig)
    playwright: PlaywrightConfig = Field(default_factory=PlaywrightConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    rag: RAGConfig = Field(default_factory=RAGConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    prompts: PromptConfig = Field(default_factory=PromptConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)
    evaluation: dict[str, Any] = Field(default_factory=dict)
    project_root: Path = PROJECT_ROOT

    # -- convenience ------------------------------------------------------- #
    @property
    def knowledge_base_path(self) -> Path:
        return self.project_root / self.rag.knowledge_base_dir

    @property
    def prompts_path(self) -> Path:
        return self.project_root / self.prompts.directory

    @property
    def reports_path(self) -> Path:
        path = self.project_root / "reports"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def test_data_path(self) -> Path:
        return self.project_root / "test-data"

    @property
    def baselines_path(self) -> Path:
        path = self.project_root / "data" / "baselines"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def metric_threshold(self, metric: str, default: float = 0.8) -> float:
        node = self.evaluation.get("metrics", {}).get(metric, {})
        for key in ("threshold", "minimum", "max"):
            if key in node:
                return float(node[key])
        return default

    def gate(self, name: str) -> dict[str, Any]:
        return dict(self.evaluation.get("quality_gate", {}).get(name, {}))


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    _load_dotenv()
    raw = _yaml("config.yaml")
    evaluation = _yaml("evaluation.yaml")

    app = AppConfig(**raw.get("app", {}))
    app.base_url = _env_str("APP_BASE_URL", app.base_url).rstrip("/")
    app.api_base_url = _env_str("API_BASE_URL", app.api_base_url).rstrip("/")
    app.auto_start = _env_bool("APP_AUTO_START", app.auto_start)

    auth = AuthConfig(**raw.get("auth", {}))
    auth.email = _env_str("TEST_USER_EMAIL", auth.email)
    auth.password = _env_str("TEST_USER_PASSWORD", auth.password)

    pw = PlaywrightConfig(**raw.get("playwright", {}))
    pw.headless = _env_bool("HEADLESS", pw.headless)
    pw.browser = _env_str("BROWSER", pw.browser)
    pw.slow_mo_ms = _env_int("SLOW_MO", pw.slow_mo_ms)

    llm = LLMConfig(**raw.get("llm", {}))
    llm.provider = _env_str("LLM_PROVIDER", llm.provider).lower()
    llm.model = _env_str("LLM_MODEL", llm.model)
    llm.temperature = _env_float("LLM_TEMPERATURE", llm.temperature)
    llm.api_key = os.getenv("OPENAI_API_KEY") or None
    llm.base_url = os.getenv("OPENAI_BASE_URL") or None

    rag = RAGConfig(**raw.get("rag", {}))
    agent = AgentConfig(**raw.get("agent", {}))
    prompts = PromptConfig(**raw.get("prompts", {}))
    prompts.active_version = _env_str("PROMPT_VERSION", prompts.active_version)

    obs = ObservabilityConfig(**raw.get("observability", {}))
    obs.langfuse_enabled = _env_bool("LANGFUSE_ENABLED", obs.langfuse_enabled)
    obs.public_key = os.getenv("LANGFUSE_PUBLIC_KEY") or None
    obs.secret_key = os.getenv("LANGFUSE_SECRET_KEY") or None
    obs.host = _env_str("LANGFUSE_HOST", obs.host)

    if "evaluator" in evaluation:
        evaluation["evaluator"]["backend"] = _env_str(
            "EVALUATOR_BACKEND", evaluation["evaluator"].get("backend", "auto")
        )

    return Settings(
        app=app,
        auth=auth,
        playwright=pw,
        llm=llm,
        rag=rag,
        agent=agent,
        prompts=prompts,
        observability=obs,
        evaluation=evaluation,
    )


settings = get_settings()
