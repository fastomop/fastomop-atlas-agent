"""Mind Meets Machines challenge runners.

Both `run_single_challenge(id)` and `run_all_challenges(output_dir)` were
previously top-level scripts at the repo root. They now live as package
functions invoked by the Typer CLI in `atlas_agent.cli`.

The print() calls in these runners are intentional CLI UX (progress
banners, emoji status) — they are user-facing terminal output, not
diagnostic logs that should be filtered by LOG_LEVEL.
"""

from __future__ import annotations

import json
import time
import traceback
from datetime import datetime
from pathlib import Path

import requests

from ..agents import OrchestratorAgent

# Challenge catalogue from the OHDSI MindMeetsMachines repository.
CHALLENGES: dict[str, dict[str, str]] = {
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


def _fetch_vignette(url: str) -> str:
    """Fetch vignette content from GitHub."""
    print("   📥 Fetching vignette from GitHub...")
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return response.text


def _run_one(challenge_id: str, challenge: dict, orchestrator: OrchestratorAgent, output_dir: Path) -> dict:
    """Run a single challenge and write outputs under ``output_dir/<challenge_id>/``."""
    challenge_name = challenge["name"]

    print(f"\n{'=' * 80}")
    print(f"🏥 CHALLENGE {challenge_id}: {challenge_name}")
    print(f"{'=' * 80}\n")

    challenge_dir = output_dir / challenge_id
    challenge_dir.mkdir(parents=True, exist_ok=True)

    try:
        vignette = _fetch_vignette(challenge["url"])
        print(f"   ✓ Fetched {len(vignette)} characters\n")

        vignette_path = challenge_dir / "vignette.md"
        vignette_path.write_text(vignette)
        print(f"   💾 Saved vignette to {vignette_path}\n")

        print("   ⏳ Running ATLAS agent (this may take 3-5 minutes)...\n")
        start_time = time.time()

        concept_set, _atlas_json = orchestrator.create_concept_set(
            clinical_description=vignette,
            validate=True,
            export_path=str(challenge_dir / "concept_set.json"),
        )

        elapsed_time = time.time() - start_time

        summary: dict = {
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

        for item in concept_set.items:
            domain = item.concept.domain_id
            summary["concepts_by_domain"][domain] = summary["concepts_by_domain"].get(domain, 0) + 1
            vocab = item.concept.vocabulary_id
            summary["concepts_by_vocabulary"][vocab] = summary["concepts_by_vocabulary"].get(vocab, 0) + 1
            if item.is_excluded:
                summary["excluded_concepts"] += 1
            else:
                summary["included_concepts"] += 1

        (challenge_dir / "summary.json").write_text(json.dumps(summary, indent=2))

        explanation = orchestrator.explain_concept_set(concept_set)
        (challenge_dir / "explanation.txt").write_text(explanation)

        print(f"\n   {'=' * 76}")
        print(f"   ✅ CHALLENGE {challenge_id} COMPLETED SUCCESSFULLY")
        print(f"   {'=' * 76}")
        print(f"   📊 Total concepts: {summary['total_concepts']}")
        print(f"   ✓  Included: {summary['included_concepts']}")
        print(f"   ✗  Excluded: {summary['excluded_concepts']}")
        print(f"   ⏱️  Time: {elapsed_time:.1f}s")
        print(f"\n   📁 Output saved to: {challenge_dir}/")

        return {"success": True, "challenge_id": challenge_id, "summary": summary}

    except Exception as e:
        print(f"\n   {'=' * 76}")
        print(f"   ❌ CHALLENGE {challenge_id} FAILED")
        print(f"   {'=' * 76}")
        print(f"   Error: {type(e).__name__}: {e}")
        traceback.print_exc()

        (challenge_dir / "error.txt").write_text(
            f"Challenge {challenge_id} failed at {datetime.now().isoformat()}\n\n"
            f"Error: {type(e).__name__}: {e}\n\n"
            f"Traceback:\n{traceback.format_exc()}"
        )

        return {"success": False, "challenge_id": challenge_id, "error": str(e)}


def run_single_challenge(challenge_id: str) -> int:
    """Run one Mind Meets Machines challenge by ID and return an exit code."""
    challenge_id = challenge_id.upper()

    if challenge_id not in CHALLENGES:
        print(f"❌ Unknown challenge ID: {challenge_id}")
        print("\nAvailable challenges:")
        for cid, info in CHALLENGES.items():
            print(f"  {cid}: {info['name']}")
        return 1

    output_dir = Path("output/challenges")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"🏥 CHALLENGE {challenge_id}: {CHALLENGES[challenge_id]['name']}")
    print("=" * 80)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Source: {CHALLENGES[challenge_id]['url']}")
    print("=" * 80 + "\n")

    print("🔧 Initializing ATLAS orchestrator agent...")
    orchestrator = OrchestratorAgent()
    print("   ✓ Orchestrator ready\n")

    result = _run_one(challenge_id, CHALLENGES[challenge_id], orchestrator, output_dir)
    return 0 if result["success"] else 1


def run_all_challenges(output_dir: Path = Path("output/challenges")) -> int:
    """Run every Mind Meets Machines challenge in sequence and return an exit code."""
    print("=" * 80)
    print("🚀 MIND MEETS MACHINES - ATLAS AGENT CHALLENGE RUNNER")
    print("=" * 80)
    print(f"\nTotal challenges: {len(CHALLENGES)}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"\n{'=' * 80}\n")

    output_dir.mkdir(parents=True, exist_ok=True)

    print("🔧 Initializing ATLAS orchestrator agent...")
    orchestrator = OrchestratorAgent()
    print("   ✓ Orchestrator ready\n")

    results = []
    start_time = time.time()
    items = list(CHALLENGES.items())

    for i, (cid, challenge) in enumerate(items, 1):
        print(f"\n{'#' * 80}")
        print(f"# PROGRESS: Challenge {i}/{len(items)}")
        print(f"{'#' * 80}")

        results.append(_run_one(cid, challenge, orchestrator, output_dir))

        if i < len(items):
            print("\n   ⏸️  Pausing 5 seconds before next challenge...")
            time.sleep(5)

    total_time = time.time() - start_time
    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]

    print(f"\n{'=' * 80}")
    print("📋 FINAL REPORT")
    print(f"{'=' * 80}\n")
    print(f"Total challenges: {len(items)}")
    print(f"✅ Successful: {len(successful)}")
    print(f"❌ Failed: {len(failed)}")
    print(f"⏱️  Total time: {total_time / 60:.1f} minutes\n")

    if successful:
        print("Successful challenges:")
        for r in successful:
            summary = r["summary"]
            print(f"   ✓ {r['challenge_id']}: {summary['total_concepts']} concepts ({summary['elapsed_seconds']:.1f}s)")
    if failed:
        print("\nFailed challenges:")
        for r in failed:
            print(f"   ✗ {r['challenge_id']}: {r['error']}")

    report = {
        "timestamp": datetime.now().isoformat(),
        "total_challenges": len(items),
        "successful": len(successful),
        "failed": len(failed),
        "total_time_seconds": round(total_time, 2),
        "results": results,
    }
    report_path = output_dir / "consolidated_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"\n📁 Consolidated report saved to: {report_path}")
    print(f"📁 Individual results in: {output_dir}/")

    print(f"\n{'=' * 80}")
    print("🏁 ALL CHALLENGES COMPLETED")
    print(f"{'=' * 80}\n")

    return 0 if not failed else 1
