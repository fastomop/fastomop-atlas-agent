"""Test the refined relationship reasoning with hierarchy analysis."""
from src.atlas_agent.agents.clinical_parser import ClinicalParserAgent
from src.atlas_agent.agents.concept_finder import ConceptFinderAgent
from src.atlas_agent.agents.relationship_reasoner import RelationshipReasonerAgent

# Initialize agents
parser = ClinicalParserAgent()
finder = ConceptFinderAgent()
reasoner = RelationshipReasonerAgent()

# Test case: "ANA positive" should prefer "Antinuclear antibody" over "True positive"
test_description = "Patient with ANA positive"

print("=" * 80)
print("Testing: ANA positive")
print("=" * 80)

# Parse
parsed = parser.parse(test_description)
entity = parsed.entities[0]
print(f"\n📋 Parsed entity: '{entity.text}' (type: {entity.entity_type}, domain: {entity.domain})")

# Find candidates
print(f"\n🔍 Searching for candidates...")
candidates = finder.find_concepts(entity, top_k=10, min_similarity=0.6)
print(f"   Found {len(candidates)} candidates:")
for i, c in enumerate(candidates[:5], 1):
    rel_count = len(c.relationship_types)
    print(f"   {i}. [{c.concept_id}] {c.concept_name} (sim: {c.similarity_score:.3f}, {rel_count} relationships)")

# Relationship reasoning
print(f"\n🧠 Applying relationship reasoning with hierarchy analysis...")
selected = reasoner.reason_about_concepts(
    entity=entity,
    candidate_concepts=candidates,
    all_entities=parsed.entities,
)

print(f"\n✅ Selected {len(selected)} concept(s):")
for c in selected:
    print(f"   → [{c.concept_id}] {c.concept_name}")

print("\n" + "=" * 80)
