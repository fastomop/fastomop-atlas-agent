# OMOP Atlas Agent

Multi-agent system for automated ATLAS concept set construction from natural language clinical descriptions.

## Overview

OMOP Atlas Agent translates free-text clinical phenotype definitions into structured ATLAS-compatible concept sets through a seven-agent pipeline:

1. **Clinical Parser**: Decomposes clinical descriptions into structured entities (conditions, drugs, measurements, procedures)
2. **Concept Finder**: Performs semantic search over 4.1M OMOP concepts using MedEmbed embeddings and Milvus vector database
3. **Relationship Reasoner**: Traverses OMOP vocabulary relationships to validate and enrich candidate concepts
4. **Set Builder**: Assembles concept sets with inclusion/exclusion logic and descendant handling
5. **Validator**: Checks concept sets for clinical correctness and completeness
6. **Corrector**: Resolves validation errors through iterative refinement
7. **Orchestrator**: Coordinates the full pipeline with structured handoffs between agents

## Requirements

- Python 3.10+
- UV package manager
- Milvus vector database (standalone or Docker)
- LLM provider (one of: Azure OpenAI, OpenAI, Anthropic, or Ollama)

### Logging

The agent uses Python's standard `logging` module. Set the verbosity via the
`LOG_LEVEL` env var (`DEBUG` | `INFO` | `WARNING` | `ERROR` | `CRITICAL`,
default `INFO`):

```bash
LOG_LEVEL=DEBUG uv run atlas_agent query "Patients with diabetes"
```

CLI usage banners and the final concept-set summary still go to stdout via
`print()` — they are part of the program's primary output, not diagnostic
logs that should be filtered by `LOG_LEVEL`.

## Installation

```bash
# Clone repository
git clone https://github.com/fastomop/omop-atlas-agent.git
cd omop-atlas-agent

# Install dependencies with UV
uv sync

# Set up environment variables
cp .env.example .env
# Edit .env with your credentials
```

## Configuration

### Environment Variables

Create a `.env` file from `.env.example` with credentials for your chosen LLM provider and Milvus connection details.

### Model Configuration

Edit `config.toml` to configure model providers and agent assignments:

```toml
[models]
default_provider = "azure"

[models.providers.azure]
provider = "azure"
model_id = "gpt-4.1"
api_version = "2025-01-01-preview"

[agents.orchestrator]
model_provider = "azure"
```

All seven agents can be assigned independently to different providers for cost optimization.

### Milvus Configuration

```toml
[milvus]
host = "localhost"
port = 19530
collection_name = "omop_clinical_concepts"

[embedding]
model = "abhinand/MedEmbed-large-v0.1"
```

The Milvus collection must be pre-loaded with OMOP concept embeddings.

## Usage

### Custom Patient Vignettes

Process your own clinical phenotype definitions from `.md` files:

```bash
# Single vignette
uv run atlas_agent vignettes my_phenotype.md

# Multiple files
uv run atlas_agent vignettes case1.md case2.md case3.md

# All .md files in a directory (recursive)
uv run atlas_agent vignettes vignettes/

# Custom output directory
uv run atlas_agent vignettes vignettes/ -o results/
```

Results are written to `output/vignettes/<name>/` with:
- `concept_set.json` — ATLAS-importable concept set
- `summary.json` — statistics (domains, vocabularies, counts)
- `explanation.txt` — human-readable rationale
- `vignette.md` — copy of the source vignette

### Mind Meets Machines Challenges

Run the [OHDSI Mind Meets Machines](https://github.com/ohdsi-studies/MindMeetsMachines) benchmark:

```bash
# Single challenge
uv run atlas_agent challenge C01

# Full challenge suite
uv run atlas_agent challenges
```

### Inline Query

```bash
uv run atlas_agent query "Patients with type 2 diabetes who received bariatric surgery"
```

Each subcommand also accepts a CLI-level `--log-level` flag that takes
precedence over the `LOG_LEVEL` env var:

```bash
uv run atlas_agent query --log-level DEBUG "..."
```

## Development

The repository uses the Astral stack:

- **uv** for package management and builds (`uv sync`, `uv build`).
- **Ruff** for linting and formatting (`uv run ruff check`, `uv run ruff format`).
- **prek** as a drop-in modern replacement for pre-commit (`uvx prek install`, `uvx prek run --all-files`).
- **ty** for type checking (alpha; advisory only — `uv run ty check src`).
- **zensical** for the documentation site (`uv sync --extra docs && uv run zensical serve`).

Install everything in one go:

```bash
uv sync --all-extras
uvx prek install
```

## Architecture

```
Clinical Description
        │
        ▼
┌─────────────────┐
│ Clinical Parser  │  Parse entities
└────────┬────────┘
         ▼
┌─────────────────┐     ┌────────────┐
│ Concept Finder   │────▶│   Milvus   │  Semantic search
└────────┬────────┘     │  (4.1M     │
         │              │  concepts) │
         │              └────────────┘
         ▼
┌─────────────────┐
│  Relationship   │  OMOP hierarchy traversal
│   Reasoner      │
└────────┬────────┘
         ▼
┌─────────────────┐
│  Set Builder     │  Assemble concept set
└────────┬────────┘
         ▼
┌─────────────────┐
│   Validator      │◀──┐  Validate clinical correctness
└────────┬────────┘   │
         │ errors     │
         ▼            │
┌─────────────────┐   │
│   Corrector      │───┘  Fix validation errors
└────────┬────────┘
         ▼
   ATLAS Concept Set (JSON)
```

## Part of FastOMOP

This project is a component of the [FastOMOP](https://github.com/fastomop) framework for automated real-world evidence generation using multi-agent architectures on OMOP CDM databases.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Contact

k24118093@kcl.ac.uk
