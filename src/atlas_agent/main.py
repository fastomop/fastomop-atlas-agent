"""Main entry point for ATLAS concept set creation."""

import logging
import sys

from ._logging import setup_logging
from .agents import OrchestratorAgent

logger = logging.getLogger(__name__)

# Usage banners use plain print() — they are user-facing CLI output, not
# diagnostic logs that should be filtered by LOG_LEVEL.
_DIVIDER = "=" * 80


def main():
    """Run the ATLAS concept set creation pipeline."""
    setup_logging()

    if len(sys.argv) < 2:
        print('Usage: python -m atlas_agent.main "<clinical description>" [output.json]')
        print("\nExample:")
        print(
            '  python -m atlas_agent.main "Patients with type 2 diabetes mellitus who have received bariatric surgery"'
        )
        sys.exit(1)

    clinical_description = sys.argv[1]
    export_path = sys.argv[2] if len(sys.argv) > 2 else None

    # Create orchestrator and run pipeline
    orchestrator = OrchestratorAgent()

    concept_set, atlas_json = orchestrator.create_concept_set(
        clinical_description=clinical_description,
        validate=True,
        export_path=export_path,
    )

    # Human-readable summary goes to stdout — this is the program's primary
    # output, not log noise.
    print("\n" + _DIVIDER)
    print("📊 CONCEPT SET SUMMARY")
    print(_DIVIDER + "\n")

    explanation = orchestrator.explain_concept_set(concept_set)
    print(explanation)

    print("\n" + _DIVIDER)
    print("✅ Ready to import into ATLAS")
    print(_DIVIDER + "\n")


if __name__ == "__main__":
    main()
