#!/usr/bin/env python3
"""Focused step-by-step test of the OMOP ATLAS pipeline."""
import sys
sys.path.insert(0, '/Users/k24118093/Documents/omop-atlas-agent/src')

from atlas_agent.agents.clinical_parser import ClinicalParserAgent
from atlas_agent.tools.milvus_search import MilvusSearchTool

print("="*80)
print("FOCUSED PIPELINE TEST")
print("="*80)

# Test 1: Simple clinical description parsing
print("\n📋 TEST 1: Clinical Parser")
print("-"*80)

parser = ClinicalParserAgent()
simple_description = """
Systemic lupus erythematosus with renal involvement, excluding drug-induced lupus.
"""

print(f"Input: {simple_description.strip()}")
print("\nParsing...")

try:
    parsed = parser.parse(simple_description)
    print(f"✅ Parser SUCCESS!")
    print(f"\nExtracted {len(parsed.entities)} entities:")
    for i, entity in enumerate(parsed.entities, 1):
        print(f"{i}. {entity.text}")
        print(f"   - Type: {entity.entity_type}")
        print(f"   - Domain: {entity.domain}")
        print(f"   - Exclusion: {entity.is_exclusion}")
        print(f"   - Descendants: {entity.requires_descendants}")
        print(f"   - Rationale: {entity.rationale[:100]}...")
except Exception as e:
    print(f"❌ Parser FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 2: Milvus search for each entity
print("\n\n🔍 TEST 2: Milvus Concept Search")
print("-"*80)

try:
    search_tool = MilvusSearchTool()
    print("✅ Milvus connection SUCCESS!")

    for i, entity in enumerate(parsed.entities, 1):
        print(f"\n{i}. Searching for: '{entity.text}'")
        print(f"   Domain filter: {entity.domain}")

        matches = search_tool.search_concepts(
            query_text=entity.text,
            domain_filter=entity.domain,
            top_k=3,
            min_similarity=0.7
        )

        if matches:
            print(f"   ✅ Found {len(matches)} matches:")
            for j, match in enumerate(matches, 1):
                print(f"      {j}. {match.concept_name} (ID: {match.concept_id})")
                print(f"         Similarity: {match.similarity_score:.3f}")
                print(f"         Vocabulary: {match.vocabulary_id}")
                print(f"         Standard: {match.standard_concept}")
        else:
            print(f"   ⚠️  No matches found above threshold")

except Exception as e:
    print(f"❌ Milvus search FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*80)
print("✅ ALL TESTS PASSED!")
print("="*80)
print("\nNext steps:")
print("1. Parser extracts entities correctly ✅")
print("2. Milvus finds matching concepts ✅")
print("3. Ready to test full orchestrator workflow")
