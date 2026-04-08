"""Run all Mind Meets Machines challenges and generate concept sets.

This script processes all challenge vignettes from the official repository:
https://github.com/ohdsi-studies/MindMeetsMachines
"""
import os
import json
import time
import traceback
from datetime import datetime
from pathlib import Path
import requests

from src.atlas_agent.agents import OrchestratorAgent


# Challenge definitions from Mind Meets Machines repository
CHALLENGES = [
    {
        "id": "C01",
        "name": "Systemic Lupus Erythematosus (SLE)",
        "url": "https://raw.githubusercontent.com/ohdsi-studies/MindMeetsMachines/main/C01/Systemic%20Lupus%20Erythematous%20(SLE).md",
    },
    {
        "id": "C02",
        "name": "Rheumatoid Arthritis",
        "url": "https://raw.githubusercontent.com/ohdsi-studies/MindMeetsMachines/main/C02/Rheumatoid%20Arthritis.md",
    },
    {
        "id": "C03",
        "name": "Diabetic Macular Edema (DME)",
        "url": "https://raw.githubusercontent.com/ohdsi-studies/MindMeetsMachines/main/C03/Diabetic%20Macular%20Edema%20(DME).md",
    },
    {
        "id": "C04",
        "name": "Acute Proximal Lower Extremity Deep Vein Thrombosis",
        "url": "https://raw.githubusercontent.com/ohdsi-studies/MindMeetsMachines/main/C04/Acute%20Proximal%20Lower%20Extremity%20Deep%20Vein%20Thrombosis.md",
    },
    {
        "id": "C05",
        "name": "Ovarian Cancer",
        "url": "https://raw.githubusercontent.com/ohdsi-studies/MindMeetsMachines/main/C05/Ovarian%20Cancer.md",
    },
    {
        "id": "C06",
        "name": "Non-Infectious Posterior-Segment Uveitis",
        "url": "https://raw.githubusercontent.com/ohdsi-studies/MindMeetsMachines/main/C06/Non-Infectious%20posterior-segment%20uveitis.md",
    },
    {
        "id": "C07",
        "name": "Systemic Sclerosis (SSc)",
        "url": "https://raw.githubusercontent.com/ohdsi-studies/MindMeetsMachines/main/C07/Systemic%20Sclerosis%20(SSc).md",
    },
]


def fetch_vignette(url: str) -> str:
    """Fetch vignette content from GitHub."""
    print(f"   📥 Fetching vignette from GitHub...")
    response = requests.get(url)
    response.raise_for_status()
    return response.text


def run_challenge(challenge: dict, orchestrator: OrchestratorAgent, output_dir: Path):
    """Run a single challenge and save results."""
    challenge_id = challenge["id"]
    challenge_name = challenge["name"]

    print(f"\n{'='*80}")
    print(f"🏥 CHALLENGE {challenge_id}: {challenge_name}")
    print(f"{'='*80}\n")

    try:
        # Fetch vignette
        vignette = fetch_vignette(challenge["url"])
        print(f"   ✓ Fetched {len(vignette)} characters\n")

        # Create output directory for this challenge
        challenge_dir = output_dir / challenge_id
        challenge_dir.mkdir(parents=True, exist_ok=True)

        # Save vignette
        vignette_path = challenge_dir / "vignette.md"
        with open(vignette_path, "w") as f:
            f.write(vignette)
        print(f"   💾 Saved vignette to {vignette_path}\n")

        # Run orchestrator
        print(f"   ⏳ Running ATLAS agent (this may take 3-5 minutes)...\n")
        start_time = time.time()

        concept_set, atlas_json = orchestrator.create_concept_set(
            clinical_description=vignette,
            validate=True,
            export_path=str(challenge_dir / "concept_set.json"),
        )

        elapsed_time = time.time() - start_time

        # Save summary report
        summary = {
            "challenge_id": challenge_id,
            "challenge_name": challenge_name,
            "timestamp": datetime.now().isoformat(),
            "elapsed_seconds": round(elapsed_time, 2),
            "concept_set_name": concept_set.name,
            "total_concepts": len(concept_set.items),
            "concepts_by_domain": {},
            "concepts_by_vocabulary": {},
            "included_concepts": 0,
            "excluded_concepts": 0,
        }

        # Calculate statistics
        for item in concept_set.items:
            # Domain stats
            domain = item.concept.domain_id
            summary["concepts_by_domain"][domain] = summary["concepts_by_domain"].get(domain, 0) + 1

            # Vocabulary stats
            vocab = item.concept.vocabulary_id
            summary["concepts_by_vocabulary"][vocab] = summary["concepts_by_vocabulary"].get(vocab, 0) + 1

            # Include/exclude stats
            if item.is_excluded:
                summary["excluded_concepts"] += 1
            else:
                summary["included_concepts"] += 1

        # Save summary
        summary_path = challenge_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        # Save explanation
        explanation = orchestrator.explain_concept_set(concept_set)
        explanation_path = challenge_dir / "explanation.txt"
        with open(explanation_path, "w") as f:
            f.write(explanation)

        print(f"\n   {'='*76}")
        print(f"   ✅ CHALLENGE {challenge_id} COMPLETED SUCCESSFULLY")
        print(f"   {'='*76}")
        print(f"   📊 Total concepts: {summary['total_concepts']}")
        print(f"   ✓  Included: {summary['included_concepts']}")
        print(f"   ✗  Excluded: {summary['excluded_concepts']}")
        print(f"   ⏱️  Time: {elapsed_time:.1f}s")
        print(f"\n   📁 Output saved to: {challenge_dir}/")
        print(f"      - concept_set.json (ATLAS import format)")
        print(f"      - summary.json (statistics)")
        print(f"      - explanation.txt (human-readable)")
        print(f"      - vignette.md (original challenge)")

        return {
            "success": True,
            "challenge_id": challenge_id,
            "summary": summary,
        }

    except Exception as e:
        print(f"\n   {'='*76}")
        print(f"   ❌ CHALLENGE {challenge_id} FAILED")
        print(f"   {'='*76}")
        print(f"   Error: {type(e).__name__}: {e}")
        print(f"\n   Traceback:")
        traceback.print_exc()

        # Save error log
        error_path = challenge_dir / "error.txt"
        with open(error_path, "w") as f:
            f.write(f"Challenge {challenge_id} failed at {datetime.now().isoformat()}\n\n")
            f.write(f"Error: {type(e).__name__}: {e}\n\n")
            f.write("Traceback:\n")
            f.write(traceback.format_exc())

        return {
            "success": False,
            "challenge_id": challenge_id,
            "error": str(e),
        }


def main():
    """Run all challenges."""
    print("="*80)
    print("🚀 MIND MEETS MACHINES - ATLAS AGENT CHALLENGE RUNNER")
    print("="*80)
    print(f"\nTotal challenges: {len(CHALLENGES)}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"\n{'='*80}\n")

    # Create output directory
    output_dir = Path("output/challenges")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize orchestrator once (reuse across challenges)
    print("🔧 Initializing ATLAS orchestrator agent...")
    orchestrator = OrchestratorAgent()
    print("   ✓ Orchestrator ready\n")

    # Run all challenges
    results = []
    start_time = time.time()

    for i, challenge in enumerate(CHALLENGES, 1):
        print(f"\n{'#'*80}")
        print(f"# PROGRESS: Challenge {i}/{len(CHALLENGES)}")
        print(f"{'#'*80}")

        result = run_challenge(challenge, orchestrator, output_dir)
        results.append(result)

        # Brief pause between challenges to avoid overwhelming the system
        if i < len(CHALLENGES):
            print(f"\n   ⏸️  Pausing 5 seconds before next challenge...")
            time.sleep(5)

    # Generate final report
    total_time = time.time() - start_time

    print(f"\n{'='*80}")
    print(f"📋 FINAL REPORT")
    print(f"{'='*80}\n")

    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]

    print(f"Total challenges: {len(CHALLENGES)}")
    print(f"✅ Successful: {len(successful)}")
    print(f"❌ Failed: {len(failed)}")
    print(f"⏱️  Total time: {total_time/60:.1f} minutes\n")

    if successful:
        print(f"Successful challenges:")
        for r in successful:
            summary = r["summary"]
            print(f"   ✓ {r['challenge_id']}: {summary['total_concepts']} concepts " +
                  f"({summary['elapsed_seconds']:.1f}s)")

    if failed:
        print(f"\nFailed challenges:")
        for r in failed:
            print(f"   ✗ {r['challenge_id']}: {r['error']}")

    # Save consolidated report
    report_path = output_dir / "consolidated_report.json"
    report = {
        "timestamp": datetime.now().isoformat(),
        "total_challenges": len(CHALLENGES),
        "successful": len(successful),
        "failed": len(failed),
        "total_time_seconds": round(total_time, 2),
        "results": results,
    }

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n📁 Consolidated report saved to: {report_path}")
    print(f"📁 Individual results in: {output_dir}/")

    print(f"\n{'='*80}")
    print(f"🏁 ALL CHALLENGES COMPLETED")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
