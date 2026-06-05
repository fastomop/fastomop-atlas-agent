# Contributing

Thanks for working on OMOP Atlas Agent. This guide covers the local dev loop
and the conventions CI enforces.

## Setup

Python 3.10+ and [uv](https://docs.astral.sh/uv/) are required.

```bash
uv sync --all-extras
uvx prek install
```

`uvx prek install` wires the pre-commit hooks declared in `prek.toml` into
your git config — they run ruff-check, ruff-format, and basic file-hygiene
checks on every commit.

## Running tests

By default, only unit tests run — the existing top-level `tests/test_*.py`
scripts are inherited integration smoke scripts that hit live Milvus / Ollama
/ Azure endpoints and call `sys.exit(...)` at module scope, so pytest skips
their collection (`collect_ignore_glob` in `tests/conftest.py`). The unit
suite under `tests/unit/` runs end-to-end with no external services:

```bash
# Unit tests only (the CI default).
uv run pytest

# Run a specific test file.
uv run pytest tests/unit/test_config.py -v
```

The integration scripts can still be exercised manually once you have a
working `.env` (Milvus, embedding model, LLM credentials, vignette server):

```bash
# Run individual scripts as before.
uv run python tests/test_agent.py
uv run python tests/test_sle_challenge.py
```

Converting them to pytest-native integration tests with the
`@pytest.mark.integration` marker is a tracked follow-up.

## Lint, format, type-check

```bash
uv run ruff check src tests             # lint
uv run ruff format src tests            # format (auto-fix)
uv run ruff format --check src tests    # format (verify only)
uv run ty check src                     # type check (advisory; ty is alpha)
```

`uvx prek run --all-files` runs ruff in the same configuration CI uses.

## Building

```bash
uv build   # produces dist/atlas_agent-0.1.0.tar.gz and *.whl via uv_build
```

## Logging

The agent uses Python's standard `logging` module. Set the level via the
`LOG_LEVEL` env var (default `INFO`) or the per-subcommand `--log-level`
flag:

```bash
LOG_LEVEL=DEBUG uv run atlas_agent query "..."
uv run atlas_agent query --log-level DEBUG "..."
```

## Commit conventions

Use [conventional commit](https://www.conventionalcommits.org/) prefixes:
`fix:`, `feat:`, `chore:`, `refactor:`, `docs:`, `ci:`, `test:`. Keep the
subject under ~70 characters; the body explains the *why*.

## Opening a PR

Stack on top of in-flight PRs when work is sequential — set the new PR's base
to the previous PR's branch (e.g. `--base feat/typer-cli`). GitHub auto-
retargets to `main` once the parent merges.
