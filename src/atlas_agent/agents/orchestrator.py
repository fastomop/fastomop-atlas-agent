"""Orchestrator Agent - Coordinates the ATLAS concept set creation workflow."""

import logging
from typing import Optional

from ..models import ConceptSet
from ..tools import export_to_atlas_json
from .clinical_parser import ClinicalParserAgent
from .concept_finder import ConceptFinderAgent
from .corrector import CorrectorAgent
from .relationship_reasoner import RelationshipReasonerAgent
from .set_builder import SetBuilderAgent
from .validator import ValidatorAgent

logger = logging.getLogger(__name__)


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
        logger.info("ATLAS Concept Set Creation Pipeline starting")

        # Step 1: Parse clinical description
        logger.info("Step 1: Parsing clinical description")
        logger.debug("Input preview: %s...", clinical_description[:100])

        parsed = self.parser.parse(clinical_description)

        logger.info("Extracted %d entities", len(parsed.entities))
        for entity in parsed.entities:
            logger.debug("  entity: %s (%s, domain=%s)", entity.text, entity.entity_type, entity.domain)

        logger.info("Strategy: %s", parsed.concept_set_strategy)

        # Step 2: Find OMOP concepts for each entity
        logger.info("Step 2: Finding OMOP concepts")

        concept_matches = []
        for entity in parsed.entities:
            logger.info("Searching for: '%s' (domain=%s)", entity.text, entity.domain)

            matches = self.finder.find_concepts(
                entity=entity,
                top_k=10,
                min_similarity=0.6,
            )

            if matches:
                logger.info("Found %d candidate(s)", len(matches))
                for match in matches:
                    rel_count = len(match.relationship_types)
                    rel_suffix = f" [{rel_count} relationships]" if rel_count > 0 else ""
                    logger.debug(
                        "  [%s] %s (similarity: %.3f)%s",
                        match.concept_id,
                        match.concept_name,
                        match.similarity_score,
                        rel_suffix,
                    )

                # Step 3: Use relationship reasoning to enrich and validate
                logger.info("Applying relationship reasoning")
                enriched_matches = self.reasoner.reason_about_concepts(
                    entity=entity,
                    candidate_concepts=matches,
                    all_entities=parsed.entities,
                )

                if enriched_matches:
                    logger.info("Selected %d concept(s) after relationship validation", len(enriched_matches))
                    for match in enriched_matches:
                        logger.debug("  → [%s] %s", match.concept_id, match.concept_name)
                    concept_matches.append((entity, enriched_matches))
                else:
                    logger.warning("No concepts passed relationship validation, using top candidate")
                    concept_matches.append((entity, [matches[0]]))
            else:
                logger.warning("No matches found for '%s'", entity.text)

        # Step 4: Build concept set with ATLAS rules
        logger.info("Step 4: Building concept set with ATLAS rules")

        concept_set = self.builder.build_concept_set(
            concept_matches=concept_matches,
            description=clinical_description,
        )

        logger.info("Built concept set: '%s'", concept_set.name)
        logger.info("  Items: %d", len(concept_set.items))

        # Count inclusions vs exclusions
        inclusions = sum(1 for item in concept_set.items if not item.is_excluded)
        exclusions = sum(1 for item in concept_set.items if item.is_excluded)
        with_descendants = sum(1 for item in concept_set.items if item.include_descendants)

        logger.info("  Inclusions: %d, Exclusions: %d", inclusions, exclusions)
        logger.info("  With descendants: %d", with_descendants)

        # Step 5: Validate concept set
        if validate:
            logger.info("Step 5: Validating concept set")

            # First validation attempt
            concept_set = self.validator.validate(concept_set, parsed_description=parsed)

            # If validation has notes, attempt a single correction
            if concept_set.validation_notes:
                logger.warning(
                    "Validation produced %d notes; attempting a single correction",
                    len(concept_set.validation_notes),
                )

                # Attempt to correct the concept set
                corrected_set = self.corrector.correct_concept_set(concept_set, parsed)

                # Re-validate the corrected set
                logger.info("Re-validating the corrected concept set")
                concept_set = self.validator.validate(corrected_set, parsed_description=parsed)

            logger.info("Validation complete")
            if concept_set.validation_notes:
                for note in concept_set.validation_notes[:5]:
                    logger.info("  • %s", note)
                if len(concept_set.validation_notes) > 5:
                    logger.info("  ... and %d more notes", len(concept_set.validation_notes) - 5)
            else:
                logger.info("  No validation issues found")

            logger.info("Coverage: %s", concept_set.coverage_summary)

        # Step 6: Export to ATLAS JSON
        logger.info("Step 6: Exporting to ATLAS JSON")

        atlas_json = export_to_atlas_json(concept_set)

        if export_path:
            import json

            with open(export_path, "w") as f:
                json.dump(atlas_json, f, indent=2)
            logger.info("Exported to: %s", export_path)
        else:
            logger.info("ATLAS JSON ready (%d items)", len(atlas_json["items"]))

        logger.info("Concept Set Creation Complete")

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
