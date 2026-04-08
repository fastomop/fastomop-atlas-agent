#!/usr/bin/env python3
"""Test SLE parsing with streaming to see progress."""
import sys
import time
sys.path.insert(0, '/Users/k24118093/Documents/omop-atlas-agent/src')

from ollama import Client

# Shortened SLE description - focus on core criteria
sle_focused = """
Build a concept set for Systemic Lupus Erythematosus (SLE) - systemic form only.

INCLUDE:
- Systemic lupus erythematosus
- SLE with organ involvement (nephritis, carditis, neurologic)
- Disseminated lupus erythematosus

EXCLUDE:
- Cutaneous lupus erythematosus (without systemic involvement)
- Discoid lupus
- Drug-induced lupus
- Neonatal lupus
- Antiphospholipid syndrome (without SLE)
- Mixed connective tissue disease
- Undifferentiated connective tissue disease
- Scleroderma
- Dermatomyositis
- Rheumatoid arthritis
- Sjögren syndrome
- Fibromyalgia
"""

prompt = f"""
Analyze this clinical description and extract entities for an OMOP concept set.

CLINICAL DESCRIPTION:
{sle_focused}

Extract entities following these rules:
- For each INCLUDE item, create entity with is_exclusion=false
- For each EXCLUDE item, create entity with is_exclusion=true
- Set requires_descendants=true for conditions (include subtypes)
- Set domain="Condition" for all disease entities

Respond with ONLY valid JSON:
{{
    "entities": [
        {{
            "text": "entity name",
            "entity_type": "condition",
            "domain": "Condition",
            "is_required": true,
            "requires_descendants": true,
            "is_exclusion": false,
            "temporal_constraint": null,
            "relationship_to_primary": null,
            "rationale": "explanation"
        }}
    ]
}}
"""

print("Testing SLE parsing with gpt-oss:20b (streaming)...")
print("="*80)

client = Client(host="http://localhost:11434")

print("Starting generation (this may take 2-3 minutes)...")
start_time = time.time()
response_text = ""
token_count = 0

try:
    for chunk in client.chat(
        model="gpt-oss:20b",
        messages=[{"role": "user", "content": prompt}],
        stream=True
    ):
        content = chunk['message']['content']
        response_text += content
        token_count += len(content.split())

        # Print progress every 50 tokens
        if token_count % 50 == 0:
            elapsed = time.time() - start_time
            print(f"  {token_count} tokens generated in {elapsed:.1f}s ({token_count/elapsed:.1f} tok/s)")

    elapsed = time.time() - start_time
    print(f"\n✅ Generation complete!")
    print(f"Time: {elapsed:.1f}s")
    print(f"Tokens: {token_count}")
    print(f"Speed: {token_count/elapsed:.1f} tok/s")
    print(f"\nResponse length: {len(response_text)} chars")
    print(f"\nFirst 500 chars:\n{response_text[:500]}")

    # Try to parse JSON
    import json
    import re

    json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
    if json_match:
        parsed = json.loads(json_match.group(0))
        print(f"\n✅ JSON parsed successfully!")
        print(f"Entities extracted: {len(parsed.get('entities', []))}")

        print(f"\nIncluded entities:")
        for e in parsed['entities']:
            if not e.get('is_exclusion'):
                print(f"  ✓ {e['text']}")

        print(f"\nExcluded entities:")
        for e in parsed['entities']:
            if e.get('is_exclusion'):
                print(f"  ✗ {e['text']}")
    else:
        print(f"❌ No JSON found in response")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
