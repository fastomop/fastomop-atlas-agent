"""Quick test of ATLAS agent with simple examples."""
from src.atlas_agent.agents import OrchestratorAgent
import json

# Initialize orchestrator
print("🚀 Initializing ATLAS agent...")
orchestrator = OrchestratorAgent()

# Test with a simple clinical description
clinical_description = """
Patients with type 2 diabetes mellitus who are currently taking metformin.
Exclude patients with diabetic ketoacidosis.
"""

print(f"\n{'='*80}")
print("Testing ATLAS Concept Set Creation")
print(f"{'='*80}\n")
print(f"Clinical Description:\n{clinical_description}")
print(f"\n{'='*80}\n")

try:
    # Create concept set
    concept_set, atlas_json = orchestrator.create_concept_set(
        clinical_description=clinical_description,
        validate=True,
        export_path="output/test_concept_set.json",
    )

    # Print explanation
    print("\n" + orchestrator.explain_concept_set(concept_set))

    # Show JSON structure
    print(f"\n{'='*80}")
    print("ATLAS JSON Preview (first item):")
    print(f"{'='*80}")
    if atlas_json['items']:
        print(json.dumps(atlas_json['items'][0], indent=2))

    print(f"\n✅ Test completed successfully!")
    print(f"Exported to: output/test_concept_set.json")

except Exception as e:
    print(f"\n❌ Test failed with error:")
    print(f"   {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
