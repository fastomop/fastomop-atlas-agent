"""Main entry point for ATLAS concept set creation."""
import sys
from pathlib import Path
from .agents import OrchestratorAgent


def main():
    """Run the ATLAS concept set creation pipeline."""
    if len(sys.argv) < 2:
        print("Usage: python -m atlas_agent.main \"<clinical description>\" [output.json]")
        print("\nExample:")
        print('  python -m atlas_agent.main "Patients with type 2 diabetes mellitus who have received bariatric surgery"')
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

    # Print human-readable explanation
    print("\n" + "="*80)
    print("📊 CONCEPT SET SUMMARY")
    print("="*80 + "\n")

    explanation = orchestrator.explain_concept_set(concept_set)
    print(explanation)

    print("\n" + "="*80)
    print("✅ Ready to import into ATLAS")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
