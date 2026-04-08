"""Run custom patient vignettes from local .md files.

Usage:
    python run_vignettes.py vignette.md                     # Single file
    python run_vignettes.py vig1.md vig2.md vig3.md         # Multiple files
    python run_vignettes.py vignettes/                      # All .md files in a directory
    python run_vignettes.py vignettes/ -o results/          # Custom output directory
"""
from __future__ import annotations

import argparse
import json
import time
import traceback
from datetime import datetime
from pathlib import Path

from src.atlas_agent.agents import OrchestratorAgent


def collect_vignette_paths(inputs: list[str]) -> list[Path]:
    """Resolve input arguments to a list of .md file paths."""
    paths = []
    for inp in inputs:
        p = Path(inp)
        if p.is_dir():
            found = sorted(p.glob("**/*.md"))
            if not found:
                print(f"Warning: no .md files found in {p}")
            paths.extend(found)
        elif p.is_file() and p.suffix == ".md":
            paths.append(p)
        elif p.is_file():
            print(f"Warning: skipping non-markdown file {p}")
        else:
            print(f"Warning: {p} does not exist, skipping")
    return paths


def run_vignette(
    vignette_path: Path,
    orchestrator: OrchestratorAgent,
    output_dir: Path,
) -> dict:
    """Process a single vignette file."""
    name = vignette_path.stem
    print(f"\n{'=' * 80}")
    print(f"  {name}")
    print(f"  Source: {vignette_path}")
    print(f"{'=' * 80}\n")

    try:
        vignette = vignette_path.read_text(encoding="utf-8")
        print(f"  Read {len(vignette)} characters\n")

        challenge_dir = output_dir / name
        challenge_dir.mkdir(parents=True, exist_ok=True)

        # Copy source vignette into output for reproducibility
        (challenge_dir / "vignette.md").write_text(vignette, encoding="utf-8")

        start_time = time.time()
        concept_set, atlas_json = orchestrator.create_concept_set(
            clinical_description=vignette,
            validate=True,
            export_path=str(challenge_dir / "concept_set.json"),
        )
        elapsed = time.time() - start_time

        # Statistics
        domain_counts: dict[str, int] = {}
        vocab_counts: dict[str, int] = {}
        included = excluded = 0
        for item in concept_set.items:
            domain_counts[item.concept.domain_id] = domain_counts.get(item.concept.domain_id, 0) + 1
            vocab_counts[item.concept.vocabulary_id] = vocab_counts.get(item.concept.vocabulary_id, 0) + 1
            if item.is_excluded:
                excluded += 1
            else:
                included += 1

        summary = {
            "vignette": name,
            "source_path": str(vignette_path),
            "timestamp": datetime.now().isoformat(),
            "elapsed_seconds": round(elapsed, 2),
            "concept_set_name": concept_set.name,
            "total_concepts": len(concept_set.items),
            "included_concepts": included,
            "excluded_concepts": excluded,
            "concepts_by_domain": domain_counts,
            "concepts_by_vocabulary": vocab_counts,
        }

        (challenge_dir / "summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8",
        )

        explanation = orchestrator.explain_concept_set(concept_set)
        (challenge_dir / "explanation.txt").write_text(explanation, encoding="utf-8")

        print(f"  Total concepts: {summary['total_concepts']}")
        print(f"  Included: {included}  Excluded: {excluded}")
        print(f"  Time: {elapsed:.1f}s")
        print(f"  Output: {challenge_dir}/")
        return {"success": True, "vignette": name, "summary": summary}

    except Exception as e:
        print(f"\n  FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()

        challenge_dir = output_dir / name
        challenge_dir.mkdir(parents=True, exist_ok=True)
        (challenge_dir / "error.txt").write_text(
            f"Failed at {datetime.now().isoformat()}\n\n"
            f"Error: {type(e).__name__}: {e}\n\n"
            f"Traceback:\n{traceback.format_exc()}",
            encoding="utf-8",
        )
        return {"success": False, "vignette": name, "error": str(e)}


def main():
    parser = argparse.ArgumentParser(
        description="Run custom patient vignettes through the ATLAS agent",
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="One or more .md files or directories containing .md files",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=Path("output/vignettes"),
        help="Output directory (default: output/vignettes)",
    )
    args = parser.parse_args()

    paths = collect_vignette_paths(args.inputs)
    if not paths:
        print("No vignette files found.")
        raise SystemExit(1)

    print("=" * 80)
    print("ATLAS AGENT — Custom Vignettes")
    print("=" * 80)
    print(f"Vignettes: {len(paths)}")
    for p in paths:
        print(f"  - {p}")
    print(f"Output:    {args.output}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=" * 80)

    orchestrator = OrchestratorAgent()

    results = []
    start_time = time.time()
    for i, path in enumerate(paths, 1):
        print(f"\n{'#' * 80}")
        print(f"# {i}/{len(paths)}")
        print(f"{'#' * 80}")
        results.append(run_vignette(path, orchestrator, args.output))

    total_time = time.time() - start_time
    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]

    print(f"\n{'=' * 80}")
    print("RESULTS")
    print(f"{'=' * 80}")
    print(f"Total: {len(results)}  Passed: {len(successful)}  Failed: {len(failed)}")
    print(f"Time:  {total_time / 60:.1f} minutes")

    if failed:
        print("\nFailed:")
        for r in failed:
            print(f"  - {r['vignette']}: {r['error']}")

    # Consolidated report
    report = {
        "timestamp": datetime.now().isoformat(),
        "total_vignettes": len(results),
        "successful": len(successful),
        "failed": len(failed),
        "total_time_seconds": round(total_time, 2),
        "results": results,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    report_path = args.output / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nReport: {report_path}")


if __name__ == "__main__":
    main()
