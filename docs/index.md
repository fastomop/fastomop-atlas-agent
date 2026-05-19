# OMOP Atlas Agent

Multi-agent system for automated ATLAS concept-set construction from
natural-language clinical descriptions.

## Quick start

See the [README](https://github.com/fastomop/fastomop-atlas-agent#installation)
for the full installation, configuration, and usage flow.

## Architecture overview

OMOP Atlas Agent translates free-text clinical phenotype definitions into
structured ATLAS-compatible concept sets through a seven-agent pipeline:

1. **Clinical Parser** — decomposes the description into structured entities
   (conditions, drugs, measurements, procedures).
2. **Concept Finder** — semantic search over 4.1M OMOP concepts using
   MedEmbed embeddings backed by Milvus.
3. **Relationship Reasoner** — traverses OMOP vocabulary hierarchies to
   validate and enrich candidate concepts.
4. **Set Builder** — assembles concept sets with inclusion / exclusion
   logic and descendant handling.
5. **Validator** — checks the assembled set for clinical correctness and
   completeness.
6. **Corrector** — resolves validation errors through iterative refinement.
7. **Orchestrator** — coordinates the full pipeline with structured
   hand-offs between agents.

## Documentation

This site is built with [zensical](https://zensical.org/) and deployed
automatically on pushes to `main`. To preview locally:

```bash
uv sync --extra docs
uv run zensical serve
```
