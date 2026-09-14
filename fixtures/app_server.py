"""Session-scoped application-under-test server.

The suite boots the app itself so `pytest` is the only command anyone needs to
run - locally or in CI. Set `APP_AUTO_START=false` to point the tests at an
already-running instance instead.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from collections.abc import Iterator

import httpx
import pytest

from src.utils.config import settings
from src.utils.logging import get_logger

logger = get_logger(__name__)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_health(base_url: str, timeout_s: int) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            response = httpx.get(f"{base_url}/api/health", timeout=2.0)
            if response.status_code == 200:
                return True
        except httpx.HTTPError:
            pass
        time.sleep(0.25)
    return False


@pytest.fixture(scope="session")
def app_server() -> Iterator[str]:
    """Yield the base URL of a running application under test."""
    if not settings.app.auto_start:
        base_url = settings.app.base_url
        if not _wait_for_health(base_url, settings.app.startup_timeout_s):
            pytest.fail(f"APP_AUTO_START=false but no app is reachable at {base_url}")
        yield base_url
        return

    port = _free_port()
    base_url = f"http://127.0.0.1:{port}"
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "src.app.main:app",
         "--host", "127.0.0.1", "--port", str(port), "--log-level", "warning"],
        cwd=settings.project_root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    try:
        if not _wait_for_health(base_url, settings.app.startup_timeout_s):
            process.terminate()
            output = (process.stdout.read().decode() if process.stdout else "")[-4000:]
            pytest.fail(f"Application under test failed to start on {base_url}\n{output}")

        # Keep every layer pointing at the port we actually got.
        settings.app.base_url = base_url
        settings.app.api_base_url = f"{base_url}/api"
        logger.info("app_under_test_started", extra={"base_url": base_url})
        yield base_url
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:  # pragma: no cover
            process.kill()
