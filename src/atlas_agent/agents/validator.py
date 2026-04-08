"""Validator Agent - Validates concept sets for clinical accuracy and ATLAS compliance."""
from typing import List
from agno.agent import Agent

from ..config import get_agent_config
from ..model_factory import create_model
from ..models import ConceptSet, ConceptSetItem


class ValidatorAgent:
    """
    Agent that validates concept sets for clinical correctness and ATLAS compliance.

    Performs thorough validation including:
    - Clinical accuracy (do concepts match intent?)
    - ATLAS rule compliance (correct use of descendants/exclusions)
    - Coverage assessment (are we missing key concepts?)
    - Quality checks (appropriate specificity, no duplicates)
    """

    def __init__(self):
        # Get agent configuration
        agent_config = get_agent_config("validator")

        self.agent = Agent(
            name=agent_config.get("name", "Concept Set Validator"),
            model=create_model(agent_config),
            description=agent_config.get("description", "Clinical informatics expert who validates OMOP concept sets for accuracy and completeness"),
            instructions=[
                "You are a clinical informatics expert validating OMOP concept sets for research use.",
                "Your task is to review concept sets and identify issues before they are used in cohort definitions.",
                "",
                "VALIDATION CRITERIA:",
                "",
                "1. CLINICAL ACCURACY",
                "   - Do the selected concepts semantically match the clinical intent?",
                "   - Are the concepts at the appropriate level of specificity?",
                "   - Are there any clinically inappropriate inclusions?",
                "",
                "2. ATLAS RULE COMPLIANCE",
                "   - Conditions: Should use includeDescendants=true (to capture all subtypes)",
                "   - Procedures: includeDescendants=true for classes, false for specific procedures",
                "   - Measurements: Usually includeDescendants=false (specific tests)",
                "   - Drugs: includeDescendants=true for drug classes, false for specific drugs",
                "   - Exclusions: Used appropriately to remove unwanted subtypes",
                "",
                "3. COVERAGE ASSESSMENT",
                "   - Are we missing key concepts that should be included?",
                "   - Are there common synonyms or related terms not covered?",
                "   - For conditions: Are we capturing all relevant clinical presentations?",
                "",
                "4. QUALITY CHECKS",
                "   - No duplicate concepts",
                "   - All concepts are STANDARD_CONCEPT='S' (preferred)",
                "   - Appropriate use of domain filters",
                "   - No contradictory inclusions/exclusions",
                "",
                "5. ATLAS EXPORT READINESS",
                "   - Concept set has clear, descriptive name",
                "   - Items have rationale explaining why they were included",
                "   - Set is ready for import into ATLAS without modification",
                "",
                "OUTPUT FORMAT:",
                "Provide structured validation feedback:",
                "- validation_status: 'PASS', 'PASS_WITH_WARNINGS', or 'FAIL'",
                "- issues: List of validation issues found (empty if PASS)",
                "- warnings: List of non-critical concerns",
                "- suggestions: Recommendations for improvement",
                "- coverage_summary: Assessment of clinical coverage",
                "",
                "Be thorough but practical. Focus on issues that would impact research validity.",
            ],
            markdown=False,
        )

    def validate(self, concept_set: ConceptSet, parsed_description=None) -> ConceptSet:
        """
        Validate a concept set and add validation notes.

        Args:
            concept_set: ConceptSet to validate
            parsed_description: Optional ParsedClinicalDescription with metadata (used internally, not output)

        Returns:
            ConceptSet with validation_notes and coverage_summary populated
        """
        # Format concept set for review
        items_text = self._format_items_for_review(concept_set.items)

        # Use metadata internally for better validation context (but don't output it)
        metadata_context = ""
        if parsed_description:
            # Provide context to validator for better assessment
            temporal_entities = [e for e in parsed_description.entities if e.temporal_constraint]
            if temporal_entities:
                constraints = [f"{e.text} ({e.temporal_constraint})" for e in temporal_entities]
                metadata_context += f"\nContext: Temporal constraints noted: {', '.join(constraints)}"

            relationship_entities = [e for e in parsed_description.entities if e.relationship_to_primary]
            if relationship_entities:
                relationships = [f"{e.text} ({e.relationship_to_primary})" for e in relationship_entities]
                metadata_context += f"\nContext: Entity relationships: {', '.join(relationships)}"

        prompt = f"""
Review this OMOP concept set for clinical accuracy and ATLAS compliance:

CONCEPT SET: "{concept_set.name}"
DESCRIPTION: {concept_set.description}

CONCEPTS ({len(concept_set.items)} items):
{items_text}{metadata_context}

Validate this concept set thoroughly. Check for:
1. Clinical accuracy - do these concepts match the description?
2. ATLAS rule compliance - correct use of includeDescendants, exclusions, etc.
3. Coverage - are we missing important concepts?
4. Quality - duplicates, non-standard concepts, contradictions
5. Exclusion completeness - are all required exclusions captured?

Focus only on the concept set quality. Provide validation_status, issues, warnings, suggestions, and coverage_summary.
"""

        # Get validation from LLM
        response = self.agent.run(prompt)
        validation_text = response.content if hasattr(response, 'content') else str(response)

        # Parse validation results
        validation_notes = self._parse_validation_results(validation_text)
        coverage_summary = self._extract_coverage_summary(validation_text)

        # Perform relationship coherence checks
        if parsed_description:
            relationship_issues = self._validate_relationship_coherence(
                concept_set, parsed_description
            )
            if relationship_issues:
                validation_notes.extend(relationship_issues)

        # Update concept set
        concept_set.validation_notes = validation_notes
        concept_set.coverage_summary = coverage_summary

        return concept_set

    def _format_items_for_review(self, items: List[ConceptSetItem]) -> str:
        """Format concept set items for LLM review."""
        lines = []
        for i, item in enumerate(items, 1):
            concept = item.concept
            flags = []
            if item.include_descendants:
                flags.append("DESCENDANTS")
            if item.is_excluded:
                flags.append("EXCLUDED")
            if item.include_mapped:
                flags.append("MAPPED")

            flag_str = f" [{', '.join(flags)}]" if flags else ""

            lines.append(
                f"{i}. [{concept.concept_id}] {concept.concept_name} "
                f"(Domain: {concept.domain_id}, Standard: {concept.standard_concept})"
                f"{flag_str}"
            )
            if item.rationale:
                lines.append(f"   Rationale: {item.rationale}")

        return "\n".join(lines)

    def _parse_validation_results(self, validation_text: str) -> List[str]:
        """Extract validation issues/warnings from LLM response."""
        notes = []

        # Simple parsing - look for structured sections
        for line in validation_text.split('\n'):
            line = line.strip()
            if not line:
                continue

            # Capture issues, warnings, suggestions
            if any(keyword in line.lower() for keyword in ['issue:', 'warning:', 'suggestion:', '- ']):
                # Clean up common prefixes
                line = line.lstrip('- ').lstrip('* ')
                if line:
                    notes.append(line)

        # If no structured notes found, include full response
        if not notes:
            notes.append(validation_text)

        return notes

    def _extract_coverage_summary(self, validation_text: str) -> str:
        """Extract coverage assessment from validation text."""
        # Look for coverage-related section
        for line in validation_text.split('\n'):
            if 'coverage' in line.lower():
                return line.strip()

        # Default summary
        return "Coverage assessment pending manual review"

    def _validate_relationship_coherence(
        self,
        concept_set: ConceptSet,
        parsed_description
    ) -> List[str]:
        """
        Validate concept set coherence using OMOP relationship data.

        Checks:
        1. Drug-condition coherence: drugs should treat conditions
        2. Symptom-condition coherence: symptoms should be manifestations
        3. Procedure-condition coherence: procedures should relate to conditions
        4. Missing exclusions: check for needed exclusions based on relationships

        Returns:
            List of relationship-based validation issues
        """
        issues = []

        # Extract entities by type
        condition_entities = [e for e in parsed_description.entities if e.entity_type == "condition"]
        drug_entities = [e for e in parsed_description.entities if e.entity_type == "drug"]
        symptom_entities = [e for e in parsed_description.entities if e.entity_type == "symptom"]
        procedure_entities = [e for e in parsed_description.entities if e.entity_type == "procedure"]

        # Extract concept set items by domain
        condition_items = [item for item in concept_set.items if item.concept.domain_id == "Condition"]
        drug_items = [item for item in concept_set.items if item.concept.domain_id == "Drug"]
        observation_items = [item for item in concept_set.items if item.concept.domain_id == "Observation"]
        procedure_items = [item for item in concept_set.items if item.concept.domain_id == "Procedure"]

        # Check 1: Drug-Condition Coherence
        if drug_items and condition_entities:
            # Check if drugs have treatment relationships
            for drug_item in drug_items:
                if drug_item.is_excluded:
                    continue

                relationships = drug_item.concept.relationship_types
                has_treatment_rel = any(
                    rel in relationships for rel in [
                        'May treat', 'May be treated by',
                        'Has FDA indication', 'FDA indication of',
                        'Has EMA indication', 'EMA indication of',
                        'Has PMDA indication', 'PMDA indication of',
                        'Has NMPA indication', 'NMPA indication of',
                        'Has HC indication', 'HC indication of',
                    ]
                )

                if not has_treatment_rel:
                    issues.append(
                        f"⚠️ RELATIONSHIP: Drug [{drug_item.concept.concept_id}] {drug_item.concept.concept_name} "
                        f"has no treatment/indication relationships in OMOP"
                    )

        # Check 2: Symptom-Condition Coherence
        if symptom_entities and condition_items:
            # Symptoms in Observation domain should have manifestation relationships
            for obs_item in observation_items:
                if obs_item.is_excluded:
                    continue

                # Check if this is a symptom/finding
                if any(keyword in obs_item.concept.concept_name.lower() for keyword in [
                    'pain', 'edema', 'swelling', 'finding', 'symptom', 'dyspnea', 'fatigue'
                ]):
                    relationships = obs_item.concept.relationship_types
                    has_manifestation_rel = any(
                        rel in relationships for rel in [
                            'Manifestation of', 'Has manifestation',
                            'Asso finding of', 'Finding asso with',
                            'Has asso finding', 'Asso with finding',
                        ]
                    )

                    if not has_manifestation_rel:
                        issues.append(
                            f"⚠️ RELATIONSHIP: Symptom/finding [{obs_item.concept.concept_id}] {obs_item.concept.concept_name} "
                            f"has no manifestation relationships to conditions"
                        )

        # Check 3: Procedure-Condition Coherence
        if procedure_items and condition_entities:
            # Check if procedures have diagnostic/therapeutic relationships
            for proc_item in procedure_items:
                if proc_item.is_excluded:
                    continue

                relationships = proc_item.concept.relationship_types
                has_clinical_rel = any(
                    rel in relationships for rel in [
                        'Has asso proc', 'Asso proc of',
                        'Has interprets', 'Interprets of',
                        'Using finding method', 'Finding method of',
                    ]
                )

                # Only flag if it's clearly a diagnostic/therapeutic procedure
                if any(keyword in proc_item.concept.concept_name.lower() for keyword in [
                    'ultrasound', 'imaging', 'test', 'biopsy', 'surgery', 'therapy'
                ]):
                    if not has_clinical_rel:
                        issues.append(
                            f"ℹ️ RELATIONSHIP: Procedure [{proc_item.concept.concept_id}] {proc_item.concept.concept_name} "
                            f"has limited relationship data (may be expected for diagnostic procedures)"
                        )

        # Check 4: Exclusion Completeness
        # If description mentions exclusions, verify they're captured
        exclusion_entities = [e for e in parsed_description.entities if e.is_exclusion]
        excluded_items = [item for item in concept_set.items if item.is_excluded]

        if exclusion_entities and not excluded_items:
            issues.append(
                f"⚠️ EXCLUSION: Description mentions exclusions ({len(exclusion_entities)}), "
                f"but no excluded concepts in concept set"
            )

        # Check 5: Parent-Child Hierarchy
        # Verify includeDescendants is used appropriately based on hierarchy relationships
        for item in concept_set.items:
            if item.is_excluded:
                continue

            has_hierarchy = any(
                rel in item.concept.relationship_types for rel in ['Is a', 'Subsumes']
            )

            # Conditions should typically include descendants if they have hierarchy
            if item.concept.domain_id == "Condition" and has_hierarchy and not item.include_descendants:
                issues.append(
                    f"ℹ️ HIERARCHY: Condition [{item.concept.concept_id}] {item.concept.concept_name} "
                    f"has 'Is a' hierarchy but includeDescendants=false (may miss subtypes)"
                )

        return issues
