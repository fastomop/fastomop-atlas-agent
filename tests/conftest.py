"""Shared pytest fixtures and configuration.

The `integration` marker is declared in `pyproject.toml`. Tests using it are
skipped by default in CI; run them locally with `uv run pytest -m integration`.
"""

from __future__ import annotations

import os

import pytest

# The repo-root-level test files (tests/test_*.py) are ad-hoc smoke scripts
# inherited from the project's earlier iteration. They run service calls at
# module load and use `sys.exit(...)` — pytest can't collect them cleanly.
# Skip collection here; they remain runnable directly as
# `uv run python tests/test_<name>.py` once the required env vars and a
# real Milvus / Ollama / Azure environment are in place. Properly converting
# them to pytest-native tests is tracked as a follow-up.
collect_ignore_glob = [
    "test_agent.py",
    "test_direct_ollama.py",
    "test_ollama_json.py",
    "test_ollama_plain.py",
    "test_parser_simple.py",
    "test_pipeline_focused.py",
    "test_quick_fixes.py",
    "test_refined_reasoning.py",
    "test_sle_challenge.py",
    "test_sle_streaming.py",
]

# Env vars required for any integration test that talks to real services.
# When unset, integration tests are skipped rather than crashing on a
# missing Milvus / Ollama / Azure connection.
_INTEGRATION_ENV_VARS = ("MILVUS_HOST",)


@pytest.fixture(autouse=True)
def _skip_integration_if_env_missing(request: pytest.FixtureRequest) -> None:
    """Auto-skip integration tests when the required env vars aren't set."""
    if "integration" not in request.keywords:
        return
    missing = [name for name in _INTEGRATION_ENV_VARS if not os.getenv(name)]
    if missing:
        pytest.skip(f"integration test skipped: missing env var(s) {', '.join(missing)}")
