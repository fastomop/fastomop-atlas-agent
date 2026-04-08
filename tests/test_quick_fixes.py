"""Quick test to verify the fixes work without full pipeline."""
from src.atlas_agent.agents.clinical_parser import ClinicalParserAgent
from src.atlas_agent.agents.set_builder import SetBuilderAgent
from src.atlas_agent.models import ClinicalEntity, ConceptMatch

print("=" * 80)
print("TESTING: Quick fixes verification")
print("=" * 80)

# Test 1: Parser context filtering
print("\n1. Testing parser context filtering...")
print("-" * 80)

parser = ClinicalParserAgent()

# Simplified SLE description with explicit context sections
test_description = """
# Systemic Lupus Erythematosus

INCLUDE:
- Systemic lupus erythematosus
- SLE with organ involvement

EXCLUDE:
- Drug-induced lupus
- Cutaneous lupus erythematosus

CONTEXT (for context; not encoded in this concept set):
- Diagnostic Criteria: ANA positive, anti-dsDNA, anti-Sm antibodies, low C3/C4
- Common Treatments: Hydroxychloroquine, mycophenolate (not part of this diagnosis concept set)
- Population: Adults 18+
"""

try:
    parsed = parser.parse(test_description)
    print(f"✓ Parsed {len(parsed.entities)} entities:")
    for e in parsed.entities:
        status = "EXCLUDE" if e.is_exclusion else "INCLUDE"
        print(f"   [{status}] {e.text} (type: {e.entity_type}, domain: {e.domain})")

    # Check if context items were extracted (they shouldn't be)
    entity_texts = [e.text.lower() for e in parsed.entities]
    context_items = ['ana', 'anti-dsdn a', 'anti-sm', 'c3', 'c4', 'hydroxychloroquine', 'mycophenolate', 'adult']
    found_context = [item for item in context_items if any(item in text for text in entity_texts)]

    if found_context:
        print(f"\n⚠️  WARNING: Context items were extracted: {found_context}")
    else:
        print(f"\n✓ SUCCESS: No context items extracted")

except Exception as e:
    print(f"❌ Parser test failed: {e}")
    import traceback
    traceback.print_exc()

# Test 2: Set builder concept filtering
print("\n\n2. Testing set builder concept filtering...")
print("-" * 80)

builder = SetBuilderAgent()

# Create mock entity and concept matches
entity = ClinicalEntity(
    text="systemic lupus erythematosus",
    entity_type="condition",
    domain="Condition",
    is_exclusion=False,
    requires_descendants=True,
)

# Create mock concepts (some should be filtered out)
mock_concepts = [
    # Valid condition concept
    ConceptMatch(
        concept_id=80809,
        concept_name="Systemic lupus erythematosus",
        domain_id="Condition",
        vocabulary_id="SNOMED",
        concept_class_id="Clinical Finding",
        standard_concept="S",
        similarity_score=0.95,
        relationship_types=[],
    ),
    # Invalid: Procedure concept class
    ConceptMatch(
        concept_id=123456,
        concept_name="ANA measurement",
        domain_id="Measurement",
        vocabulary_id="LOINC",
        concept_class_id="Procedure",
        standard_concept="S",
        similarity_score=0.85,
        relationship_types=[],
    ),
    # Invalid: UK Biobank answer value
    ConceptMatch(
        concept_id=1234567,
        concept_name="Adult",
        domain_id="Observation",
        vocabulary_id="UKB",
        concept_class_id="Answer",
        standard_concept="S",
        similarity_score=0.80,
        relationship_types=[],
    ),
]

try:
    concept_set = builder.build_concept_set(
        concept_matches=[(entity, mock_concepts)],
        description="Test SLE concept set",
        set_type="diagnosis"
    )

    print(f"✓ Built concept set with {len(concept_set.items)} items:")
    for item in concept_set.items:
        print(f"   [{item.concept.concept_id}] {item.concept.concept_name}")
        print(f"      Domain: {item.concept.domain_id}, Class: {item.concept.concept_class_id}")

    # Check if invalid concepts were filtered
    concept_ids = [item.concept.concept_id for item in concept_set.items]

    if 123456 in concept_ids or 1234567 in concept_ids:
        print(f"\n⚠️  WARNING: Invalid concepts were not filtered")
    else:
        print(f"\n✓ SUCCESS: Invalid concepts were filtered out")

except Exception as e:
    print(f"❌ Set builder test failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("QUICK FIXES VERIFICATION COMPLETE")
print("=" * 80)
