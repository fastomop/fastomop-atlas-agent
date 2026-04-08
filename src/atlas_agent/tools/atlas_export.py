"""ATLAS JSON export utilities."""
from ..models import ConceptSet, AtlasConceptSet, AtlasConceptSetItem, AtlasConcept


def export_to_atlas_json(concept_set: ConceptSet) -> dict:
    """
    Convert internal ConceptSet to ATLAS-compatible JSON format.

    Args:
        concept_set: Internal concept set representation

    Returns:
        Dict ready for JSON export (ATLAS format)
    """
    atlas_items = []

    for item in concept_set.items:
        # Map standard_concept to ATLAS format
        standard_concept_caption_map = {
            "S": "Standard",
            "C": "Classification",
            "N": "Non-Standard",
            "": "Non-Standard",
        }

        invalid_reason_caption_map = {
            "V": "Valid",
            "D": "Invalid",
            "U": "Invalid",
            None: "Valid",
            "": "Valid",
        }

        # Create ATLAS concept
        atlas_concept = AtlasConcept(
            CONCEPT_ID=item.concept.concept_id,
            CONCEPT_NAME=item.concept.concept_name,
            DOMAIN_ID=item.concept.domain_id,
            VOCABULARY_ID=item.concept.vocabulary_id,
            CONCEPT_CLASS_ID=item.concept.concept_class_id,
            STANDARD_CONCEPT=item.concept.standard_concept or "",
            CONCEPT_CODE=item.concept.concept_code,
            VALID_START_DATE=item.concept.valid_start_date,
            VALID_END_DATE=item.concept.valid_end_date,
            INVALID_REASON=item.concept.invalid_reason or "V",
            INVALID_REASON_CAPTION=invalid_reason_caption_map.get(
                item.concept.invalid_reason, "Valid"
            ),
            STANDARD_CONCEPT_CAPTION=standard_concept_caption_map.get(
                item.concept.standard_concept, "Non-Standard"
            ),
        )

        # Create ATLAS item
        atlas_item = AtlasConceptSetItem(
            concept=atlas_concept,
            isExcluded=item.is_excluded,
            includeDescendants=item.include_descendants,
            includeMapped=item.include_mapped,
        )

        atlas_items.append(atlas_item)

    # Convert Pydantic model to dict for JSON serialization
    atlas_concept_set = AtlasConceptSet(items=atlas_items)
    return atlas_concept_set.model_dump()
