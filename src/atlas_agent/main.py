"""Inline-query entry point for ATLAS concept set creation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from .agents import OrchestratorAgent

logger = logging.getLogger(__name__)

# The CONCEPT SET SUMMARY block uses plain print() — it's the program's
# primary output, not diagnostic logs that should be filtered by LOG_LEVEL.
_DIVIDER = "=" * 80


def query(clinical_description: str, export_path: Optional[Path | str] = None) -> int:
    """Run the orchestrator on a single inline clinical description.

    Returns an integer exit code suitable for CLI dispatch.
    """
    orchestrator = OrchestratorAgent()

    concept_set, _atlas_json = orchestrator.create_concept_set(
        clinical_description=clinical_description,
        validate=True,
        export_path=str(export_path) if export_path else None,
    )

    print("\n" + _DIVIDER)
    print("📊 CONCEPT SET SUMMARY")
    print(_DIVIDER + "\n")

    explanation = orchestrator.explain_concept_set(concept_set)
    print(explanation)

    print("\n" + _DIVIDER)
    print("✅ Ready to import into ATLAS")
    print(_DIVIDER + "\n")

    return 0
