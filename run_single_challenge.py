"""Run a single Mind Meets Machines challenge.

Usage:
    python run_single_challenge.py C01  # Run SLE challenge
    python run_single_challenge.py C02  # Run Rheumatoid Arthritis challenge
    etc.
"""
import sys
import json
import time
from datetime import datetime
from pathlib import Path
import requests

from src.atlas_agent.agents import OrchestratorAgent


# Challenge definitions
CHALLENGES = {
    "C01": {
        "name": "Systemic Lupus Erythematosus (SLE)",
        "url": "https://raw.githubusercontent.com/ohdsi-studies/MindMeetsMachines/main/C01/Systemic%20Lupus%20Erythematous%20(SLE).md",
    },
    "C02": {
        "name": "Rheumatoid Arthritis",
        "url": "https://raw.githubusercontent.com/ohdsi-studies/MindMeetsMachines/main/C02/Rheumatoid%20Arthritis.md",
    },
    "C03": {
        "name": "Diabetic Macular Edema (DME)",
        "url": "https://raw.githubusercontent.com/ohdsi-studies/MindMeetsMachines/main/C03/Diabetic%20Macular%20Edema%20(DME).md",
    },
    "C04": {
        "name": "Acute Proximal Lower Extremity Deep Vein Thrombosis",
        "url": "https://raw.githubusercontent.com/ohdsi-studies/MindMeetsMachines/main/C04/Acute%20Proximal%20Lower%20Extremity%20Deep%20Vein%20Thrombosis.md",
    },
    "C05": {
        "name": "Ovarian Cancer",
        "url": "https://raw.githubusercontent.com/ohdsi-studies/MindMeetsMachines/main/C05/Ovarian%20Cancer.md",
    },
    "C06": {
        "name": "Non-Infectious Posterior-Segment Uveitis",
        "url": "https://raw.githubusercontent.com/ohdsi-studies/MindMeetsMachines/main/C06/Non-Infectious%20posterior-segment%20uveitis.md",
    },
    "C07": {
        "name": "Systemic Sclerosis (SSc)",
        "url": "https://raw.githubusercontent.com/ohdsi-studies/MindMeetsMachines/main/C07/Systemic%20Sclerosis%20(SSc).md",
    },
}


def main():
    # Parse command line arguments
    if len(sys.argv) < 2:
        print("Usage: python run_single_challenge.py <challenge_id>")
        print("\nAvailable challenges:")
        for cid, info in CHALLENGES.items():
            print(f"  {cid}: {info['name']}")
        sys.exit(1)

    challenge_id = sys.argv[1].upper()

    if challenge_id not in CHALLENGES:
        print(f"❌ Unknown challenge ID: {challenge_id}")
        print("\nAvailable challenges:")
        for cid, info in CHALLENGES.items():
            print(f"  {cid}: {info['name']}")
        sys.exit(1)

    challenge = CHALLENGES[challenge_id]

    print("="*80)
    print(f"🏥 CHALLENGE {challenge_id}: {challenge['name']}")
    print("="*80)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Source: {challenge['url']}")
    print("="*80 + "\n")

    # Fetch vignette
    print("📥 Fetching vignette from GitHub...")
    response = requests.get(challenge["url"])
    response.raise_for_status()
    vignette = response.text
    print(f"✓ Fetched {len(vignette)} characters\n")

    # Create output directory
    output_dir = Path(f"output/challenges/{challenge_id}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save vignette
    vignette_path = output_dir / "vignette.md"
    with open(vignette_path, "w") as f:
        f.write(vignette)
    print(f"💾 Saved vignette to {vignette_path}\n")

    # Initialize orchestrator
    print("🔧 Initializing ATLAS orchestrator agent...")
    orchestrator = OrchestratorAgent()
    print("✓ Orchestrator ready\n")

    # Run concept set creation
    print("⏳ Running ATLAS agent (this may take 3-5 minutes)...\n")
    print("="*80 + "\n")

    start_time = time.time()

    try:
        concept_set, atlas_json = orchestrator.create_concept_set(
            clinical_description=vignette,
            validate=True,
            export_path=str(output_dir / "concept_set.json"),
        )

        elapsed_time = time.time() - start_time

        # Print results
        print("\n" + "="*80)
        print("📊 CONCEPT SET SUMMARY")
        print("="*80)
        print(f"Name: {concept_set.name}")
        print(f"Total concepts: {len(concept_set.items)}")

        # Domain breakdown
        domain_counts = {}
        for item in concept_set.items:
            domain = item.concept.domain_id
            domain_counts[domain] = domain_counts.get(domain, 0) + 1

        print("\nConcepts by domain:")
        for domain, count in sorted(domain_counts.items()):
            print(f"  {domain}: {count}")

        # Include/exclude breakdown
        included = sum(1 for item in concept_set.items if not item.is_excluded)
        excluded = sum(1 for item in concept_set.items if item.is_excluded)
        print(f"\nIncluded: {included}")
        print(f"Excluded: {excluded}")

        # Show some example concepts
        print("\n" + "="*80)
        print("EXAMPLE CONCEPTS (first 10)")
        print("="*80)
        for i, item in enumerate(concept_set.items[:10], 1):
            status = "-" if item.is_excluded else "+"
            descendants = " [+descendants]" if item.include_descendants else ""
            print(f"{i:2d}. {status} [{item.concept.concept_id}] {item.concept.concept_name}{descendants}")
            print(f"     {item.concept.domain_id} / {item.concept.vocabulary_id} / {item.concept.concept_class_id}")

        if len(concept_set.items) > 10:
            print(f"\n... and {len(concept_set.items) - 10} more concepts")

        # Save explanation
        print("\n" + "="*80)
        print("GENERATING EXPLANATION")
        print("="*80)
        explanation = orchestrator.explain_concept_set(concept_set)
        explanation_path = output_dir / "explanation.txt"
        with open(explanation_path, "w") as f:
            f.write(explanation)
        print(explanation)

        # Save summary
        summary = {
            "challenge_id": challenge_id,
            "challenge_name": challenge["name"],
            "timestamp": datetime.now().isoformat(),
            "elapsed_seconds": round(elapsed_time, 2),
            "concept_set_name": concept_set.name,
            "total_concepts": len(concept_set.items),
            "included_concepts": included,
            "excluded_concepts": excluded,
            "concepts_by_domain": domain_counts,
        }

        summary_path = output_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        print("\n" + "="*80)
        print("✅ CHALLENGE COMPLETED SUCCESSFULLY")
        print("="*80)
        print(f"⏱️  Time: {elapsed_time:.1f} seconds")
        print(f"\n📁 Output saved to: {output_dir}/")
        print(f"   - concept_set.json (ATLAS import format)")
        print(f"   - summary.json (statistics)")
        print(f"   - explanation.txt (human-readable)")
        print(f"   - vignette.md (original challenge)")
        print("\n" + "="*80)

    except Exception as e:
        print("\n" + "="*80)
        print("❌ CHALLENGE FAILED")
        print("="*80)
        print(f"Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

        # Save error log
        error_path = output_dir / "error.txt"
        with open(error_path, "w") as f:
            f.write(f"Challenge {challenge_id} failed at {datetime.now().isoformat()}\n\n")
            f.write(f"Error: {type(e).__name__}: {e}\n\n")
            f.write("Traceback:\n")
            f.write(traceback.format_exc())

        sys.exit(1)


if __name__ == "__main__":
    main()
