"""Orchestrator Agent - Coordinates the ATLAS concept set creation workflow."""
from typing import Optional
from .clinical_parser import ClinicalParserAgent
from .concept_finder import ConceptFinderAgent
from .relationship_reasoner import RelationshipReasonerAgent
from .set_builder import SetBuilderAgent
from .validator import ValidatorAgent
from .corrector import CorrectorAgent
from ..models import ConceptSet
from ..tools import export_to_atlas_json


class OrchestratorAgent:
    """
    Orchestrates the end-to-end ATLAS concept set creation workflow.

    Workflow:
    1. Parse clinical description → extract entities
    2. Find OMOP concepts for each entity
    3. Use relationships to enrich and validate concepts
    4. Build concept set with ATLAS rules
    5. Validate concept set thoroughly
    6. If validation fails, attempt a single correction and re-validate
    7. Export to ATLAS JSON format
    """

    def __init__(self):
        self.parser = ClinicalParserAgent()
        self.finder = ConceptFinderAgent()
        self.reasoner = RelationshipReasonerAgent()
        self.builder = SetBuilderAgent()
        self.validator = ValidatorAgent()
        self.corrector = CorrectorAgent()

    def create_concept_set(
        self,
        clinical_description: str,
        validate: bool = True,
        export_path: Optional[str] = None,
    ) -> tuple[ConceptSet, dict]:
        """
        Create an ATLAS concept set from a clinical description.

        Args:
            clinical_description: Natural language clinical description (2-3 lines)
            validate: Whether to run thorough validation (default: True)
            export_path: Optional path to export ATLAS JSON

        Returns:
            Tuple of (ConceptSet, ATLAS JSON dict)
        """
        print(f"\n{'='*80}")
        print("🏥 ATLAS Concept Set Creation Pipeline")
        print(f"{'='*80}\n")

        # Step 1: Parse clinical description
        print("📋 Step 1: Parsing clinical description...")
        print(f"Input: {clinical_description[:100]}...")

        parsed = self.parser.parse(clinical_description)

        print(f"\n✓ Extracted {len(parsed.entities)} entities:")
        for entity in parsed.entities:
            print(f"  • {entity.text} ({entity.entity_type}, domain={entity.domain})")

        print(f"\nStrategy: {parsed.concept_set_strategy}")

        # Step 2: Find OMOP concepts for each entity
        print(f"\n🔍 Step 2: Finding OMOP concepts...")

        concept_matches = []
        for entity in parsed.entities:
            print(f"\n  Searching for: '{entity.text}' (domain={entity.domain})")

            matches = self.finder.find_concepts(
                entity=entity,
                top_k=10,
                min_similarity=0.6,
            )

            if matches:
                print(f"  ✓ Found {len(matches)} candidate(es):")
                for match in matches:
                    rel_count = len(match.relationship_types)
                    rel_suffix = f" [{rel_count} relationships]" if rel_count > 0 else ""
                    print(f"    - [{match.concept_id}] {match.concept_name} (similarity: {match.similarity_score:.3f}){rel_suffix}")

                # Step 3: Use relationship reasoning to enrich and validate
                print(f"\n  🧠 Applying relationship reasoning...")
                enriched_matches = self.reasoner.reason_about_concepts(
                    entity=entity,
                    candidate_concepts=matches,
                    all_entities=parsed.entities,
                )

                if enriched_matches:
                    print(f"  ✓ Selected {len(enriched_matches)} concept(s) after relationship validation:")
                    for match in enriched_matches:
                        print(f"    → [{match.concept_id}] {match.concept_name}")
                    concept_matches.append((entity, enriched_matches))
                else:
                    print(f"  ⚠ No concepts passed relationship validation, using top candidate")
                    concept_matches.append((entity, [matches[0]]))
            else:
                print(f"  ⚠ No matches found for '{entity.text}'")

        # Step 4: Build concept set with ATLAS rules
        print(f"\n🏗️  Step 4: Building concept set with ATLAS rules...")

        concept_set = self.builder.build_concept_set(
            concept_matches=concept_matches,
            description=clinical_description,
        )

        print(f"\n✓ Built concept set: '{concept_set.name}'")
        print(f"  Items: {len(concept_set.items)}")

        # Count inclusions vs exclusions
        inclusions = sum(1 for item in concept_set.items if not item.is_excluded)
        exclusions = sum(1 for item in concept_set.items if item.is_excluded)
        with_descendants = sum(1 for item in concept_set.items if item.include_descendants)

        print(f"  Inclusions: {inclusions}, Exclusions: {exclusions}")
        print(f"  With descendants: {with_descendants}")

        # Step 5: Validate concept set
        if validate:
            print(f"\n✅ Step 5: Validating concept set...")

            # First validation attempt
            concept_set = self.validator.validate(concept_set, parsed_description=parsed)

            # If validation has notes, attempt a single correction
            if concept_set.validation_notes:
                print(f"  ⚠️ Validation produced {len(concept_set.validation_notes)} notes. Attempting a single correction...")

                # Attempt to correct the concept set
                corrected_set = self.corrector.correct_concept_set(concept_set, parsed)

                # Re-validate the corrected set
                print(f"\n✅ Re-validating the corrected concept set...")
                concept_set = self.validator.validate(corrected_set, parsed_description=parsed)

            print(f"\n✓ Validation complete:")
            if concept_set.validation_notes:
                for note in concept_set.validation_notes[:5]:  # Show first 5
                    print(f"  • {note}")
                if len(concept_set.validation_notes) > 5:
                    print(f"  ... and {len(concept_set.validation_notes) - 5} more notes")
            else:
                print("  • No validation issues found")

            print(f"\nCoverage: {concept_set.coverage_summary}")

        # Step 6: Export to ATLAS JSON
        print(f"\n📤 Step 6: Exporting to ATLAS JSON...")

        atlas_json = export_to_atlas_json(concept_set)

        if export_path:
            import json
            with open(export_path, 'w') as f:
                json.dump(atlas_json, f, indent=2)
            print(f"✓ Exported to: {export_path}")
        else:
            print(f"✓ ATLAS JSON ready ({len(atlas_json['items'])} items)")

        # Summary
        print(f"\n{'='*80}")
        print("✨ Concept Set Creation Complete!")
        print(f"{'='*80}\n")

        return concept_set, atlas_json

    def explain_concept_set(self, concept_set: ConceptSet) -> str:
        """
        Generate a human-readable explanation of the concept set.

        Args:
            concept_set: The concept set to explain

        Returns:
            Human-readable explanation
        """
        lines = [
            f"Concept Set: {concept_set.name}",
            f"Description: {concept_set.description}",
            "",
            f"Total Concepts: {len(concept_set.items)}",
            "",
            "Included Concepts:",
        ]

        # Group by domain
        by_domain = {}
        for item in concept_set.items:
            if item.is_excluded:
                continue
            domain = item.concept.domain_id
            if domain not in by_domain:
                by_domain[domain] = []
            by_domain[domain].append(item)

        for domain, items in sorted(by_domain.items()):
            lines.append(f"\n{domain} ({len(items)} concepts):")
            for item in items:
                desc_flag = " [+descendants]" if item.include_descendants else ""
                lines.append(f"  • [{item.concept.concept_id}] {item.concept.concept_name}{desc_flag}")

        # Exclusions
        exclusions = [item for item in concept_set.items if item.is_excluded]
        if exclusions:
            lines.append("\nExcluded Concepts:")
            for item in exclusions:
                lines.append(f"  • [{item.concept.concept_id}] {item.concept.concept_name}")

        # Validation summary
        if concept_set.validation_notes:
            lines.append("\nValidation Notes:")
            for note in concept_set.validation_notes[:3]:
                lines.append(f"  • {note}")

        if concept_set.coverage_summary:
            lines.append(f"\nCoverage: {concept_set.coverage_summary}")

        return "\n".join(lines)
