"""Set Builder Agent - Applies ATLAS concept set rules."""
from typing import List
from ..models import ClinicalEntity, ConceptMatch, ConceptSet, ConceptSetItem


class SetBuilderAgent:
    """
    Agent that builds ATLAS-compliant concept sets from matched concepts.

    Uses hard-coded clinical rules rather than LLM reasoning for consistency.
    """

    ATLAS_RULES = {
        "Condition": {
            "include_descendants": True,  # Always include subtypes of conditions
            "include_mapped": False,
        },
        "Procedure": {
            "include_descendants": True,  # Include procedure subtypes
            "include_mapped": False,
        },
        "Drug": {
            "include_descendants": True,  # Include drug formulations/brands
            "include_mapped": False,
        },
        "Device": {
            "include_descendants": True,  # Include device variants/models
            "include_mapped": False,
        },
        "Measurement": {
            "include_descendants": False, # Measurements are usually specific
            "include_mapped": False,
        },
        "Observation": {
            "include_descendants": True,  # Include related symptoms/findings (changed from False)
            "include_mapped": False,
        },
        "Specimen": {
            "include_descendants": False, # Specimen types are specific
            "include_mapped": False,
        },
        "Visit": {
            "include_descendants": False, # Visit types are specific
            "include_mapped": False,
        },
        "Gender": {
            "include_descendants": False, # Gender is specific
            "include_mapped": False,
        },
        "Race": {
            "include_descendants": False, # Race is specific
            "include_mapped": False,
        },
        "Ethnicity": {
            "include_descendants": False, # Ethnicity is specific
            "include_mapped": False,
        },
    }

    # Concept class/domain validation rules
    DIAGNOSIS_SET_RULES = {
        "allowed_concept_classes": {
            "Disorder", "Clinical Finding", "Disease", "Condition"
        },
        "allowed_domains": {
            "Condition"
        },
        "forbidden_concept_classes": {
            "Procedure", "Substance", "Answer", "Context-dependent"
        }
    }

    def build_concept_set(
        self,
        concept_matches: List[tuple[ClinicalEntity, List[ConceptMatch]]],
        description: str,
        set_type: str = "diagnosis"  # "diagnosis", "measurement", "drug", etc.
    ) -> ConceptSet:
        """
        Build an ATLAS concept set from matched concepts.

        Args:
            concept_matches: List of (entity, [matched_concepts]) tuples
            description: Clinical description for the concept set
            set_type: Type of concept set (diagnosis, measurement, drug, etc.)

        Returns:
            ConceptSet ready for validation and export
        """
        items = []
        skipped_concepts = []

        for entity, matches in concept_matches:
            for match in matches:
                # Validate concept appropriateness for diagnosis sets
                if set_type == "diagnosis":
                    skip_reason = self._validate_diagnosis_concept(entity, match)
                    if skip_reason:
                        skipped_concepts.append((match, skip_reason))
                        print(f"⚠️  Skipping [{match.concept_id}] {match.concept_name}: {skip_reason}")
                        continue
                # Apply ATLAS rules based on domain
                domain_rules = self.ATLAS_RULES.get(
                    match.domain_id,
                    {"include_descendants": False, "include_mapped": False}
                )

                # Override with entity-specific requirements if specified
                include_descendants = (
                    entity.requires_descendants
                    if entity.requires_descendants is not None
                    else domain_rules["include_descendants"]
                )

                # Build rationale
                rationale = self._build_rationale(entity, match, include_descendants)

                # Create concept set item
                item = ConceptSetItem(
                    concept=match,
                    include_descendants=include_descendants,
                    is_excluded=entity.is_exclusion,
                    include_mapped=domain_rules["include_mapped"],
                    rationale=rationale,
                )
                items.append(item)

        # Generate concept set name from description
        name = self._generate_name(description)

        return ConceptSet(
            name=name,
            description=description,
            items=items,
            validation_notes=[],
            coverage_summary="",
        )

    def _validate_diagnosis_concept(
        self,
        entity: ClinicalEntity,
        match: ConceptMatch
    ) -> str:
        """
        Validate if concept is appropriate for a diagnosis concept set.

        Returns:
            Empty string if valid, otherwise reason for rejection
        """
        # Check for forbidden concept classes
        if match.concept_class_id in self.DIAGNOSIS_SET_RULES["forbidden_concept_classes"]:
            return f"Forbidden concept class '{match.concept_class_id}' for diagnosis set"

        # For non-exclusions (inclusions), enforce stricter rules
        if not entity.is_exclusion:
            # Check domain appropriateness
            if match.domain_id not in self.DIAGNOSIS_SET_RULES["allowed_domains"]:
                # Exception: demographic entities can be in Observation domain
                if entity.entity_type != "demographic":
                    return f"Domain '{match.domain_id}' not appropriate for diagnosis inclusion (use Condition domain)"

            # Check concept class appropriateness
            if match.concept_class_id not in self.DIAGNOSIS_SET_RULES["allowed_concept_classes"]:
                # Check if it's a substance or procedure masquerading as diagnosis
                if match.concept_class_id in {"Substance", "Procedure"}:
                    return f"Concept class '{match.concept_class_id}' is not a diagnosis"

                # Allow Clinical Finding for now, but warn if it's actually a measurement result
                if match.domain_id in {"Measurement", "Observation"} and entity.entity_type not in {"demographic", "observation"}:
                    return f"Measurement/lab concept in diagnosis set - should be in separate measurement phenotype"

        # Special validation for demographic concepts
        if entity.entity_type == "demographic":
            # Reject survey answer values from UK Biobank, etc.
            if match.concept_class_id == "Answer":
                return "Demographics should use cohort entry criteria, not answer value concepts"
            if "biobank" in match.vocabulary_id.lower():
                return "Survey answer values are not appropriate for demographic filtering"

        return ""  # Valid

    def _build_rationale(
        self,
        entity: ClinicalEntity,
        match: ConceptMatch,
        include_descendants: bool
    ) -> str:
        """Build human-readable rationale for including this concept."""
        parts = [
            f"Matched entity '{entity.text}' to OMOP concept '{match.concept_name}'",
            f"(similarity: {match.similarity_score:.2f})",
        ]

        if include_descendants:
            parts.append("- includes all descendant concepts")

        if entity.is_exclusion:
            parts.append("- EXCLUDED from set")

        return " ".join(parts)

    def _generate_name(self, description: str) -> str:
        """Generate a concise name from the description."""
        # Take first 50 chars or until first period/newline
        name = description.split('\n')[0].split('.')[0][:50].strip()
        return name if name else "OMOP Concept Set"
