#!/usr/bin/env python3
"""Test if gpt-oss:20b supports JSON format with Ollama."""
import json
from ollama import Client

client = Client(host="http://localhost:11434")

# Test 1: Basic JSON generation
print("🧪 Test 1: Basic JSON generation")
print("="*80)
response = client.chat(
    model="gpt-oss:20b",
    messages=[
        {
            "role": "user",
            "content": "Generate a JSON object for a movie with fields: name, genre, year. Give me valid JSON only."
        }
    ],
    format="json"
)
print(f"Response: {response['message']['content']}")
try:
    parsed = json.loads(response['message']['content'])
    print(f"✅ Valid JSON: {parsed}")
except:
    print(f"❌ Invalid JSON")

print("\n")

# Test 2: Structured output with schema
print("🧪 Test 2: JSON with schema")
print("="*80)
schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "genre": {"type": "string"},
        "year": {"type": "integer"}
    },
    "required": ["name", "genre", "year"]
}

response = client.chat(
    model="gpt-oss:20b",
    messages=[
        {
            "role": "user",
            "content": f"Generate a movie JSON matching this schema: {json.dumps(schema)}"
        }
    ],
    format=schema
)
print(f"Response: {response['message']['content']}")
try:
    parsed = json.loads(response['message']['content'])
    print(f"✅ Valid JSON: {parsed}")
except:
    print(f"❌ Invalid JSON")
