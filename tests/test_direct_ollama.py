#!/usr/bin/env python3
"""Test gpt-oss:20b directly with JSON instructions (no format parameter)."""
import json
from ollama import Client

client = Client(host="http://localhost:11434")

prompt = """
Analyze this clinical description and extract entities for an OMOP concept set:

CLINICAL DESCRIPTION:
Acute myocardial infarction in adults, excluding silent MI.

Respond with ONLY valid JSON matching this schema:
{
    "entities": [
        {
            "entity_text": "string",
            "entity_type": "condition|symptom|procedure|measurement|drug|device|visit|demographic",
            "domain": "string",
            "is_required": true/false,
            "requires_descendants": true/false,
            "is_exclusion": true/false,
            "temporal_constraint": "string or null",
            "relationship_to_primary": "string or null",
            "rationale": "string"
        }
    ]
}

Your response must be a single JSON object. Do not include markdown code blocks or explanations.
"""

print("Testing gpt-oss:20b with JSON instructions (NO format parameter)...")
print("="*80)

response = client.chat(
    model="gpt-oss:20b",
    messages=[{"role": "user", "content": prompt}]
    # NO format parameter - model returns text, we parse JSON from it
)

raw_response = response['message']['content']
print(f"Raw response ({len(raw_response)} chars):")
print(raw_response[:500])
print("\n")

try:
    parsed = json.loads(raw_response)
    print(f"✅ Successfully parsed JSON!")
    print(f"Entities: {len(parsed.get('entities', []))}")
    for entity in parsed.get('entities', []):
        print(f"  - {entity.get('entity_text')} ({entity.get('entity_type')}, exclusion={entity.get('is_exclusion')})")
except Exception as e:
    print(f"❌ JSON parsing failed: {e}")
