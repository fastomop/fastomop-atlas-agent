#!/usr/bin/env python3
"""Simple test of clinical parser agent with short input."""
import sys
sys.path.insert(0, '/Users/k24118093/Documents/omop-atlas-agent/src')

from atlas_agent.agents.clinical_parser import ClinicalParserAgent

# Very simple test case
parser = ClinicalParserAgent()

simple_description = """
Acute myocardial infarction in adults, excluding silent MI.
"""

print(f"Testing parser with simple description:")
print(f"{simple_description}")
print(f"="*80)

try:
    result = parser.parse(simple_description)
    print(f"✅ SUCCESS!")
    print(f"Extracted {len(result.entities)} entities:")
    for entity in result.entities:
        print(f"  - {entity.entity_text} ({entity.entity_type}, exclusion={entity.is_exclusion})")
except Exception as e:
    print(f"❌ FAILED: {e}")
    import traceback
    traceback.print_exc()
