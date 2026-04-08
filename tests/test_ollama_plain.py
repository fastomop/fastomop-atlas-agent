#!/usr/bin/env python3
"""Test if gpt-oss:20b works without JSON format."""
from ollama import Client

client = Client(host="http://localhost:11434")

print("🧪 Test: Plain text generation (no format constraint)")
print("="*80)
response = client.chat(
    model="gpt-oss:20b",
    messages=[
        {
            "role": "user",
            "content": "Write a one-sentence summary of Systemic Lupus Erythematosus."
        }
    ]
    # NO format parameter
)
print(f"Response: {response['message']['content']}")
print(f"Length: {len(response['message']['content'])} characters")
