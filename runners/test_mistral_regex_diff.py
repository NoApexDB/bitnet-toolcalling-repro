import json
import sys
from transformers import AutoTokenizer

sys.stdout.reconfigure(encoding="utf-8")

tok_default = AutoTokenizer.from_pretrained('C:/ai_models/BitNet-b1.58-2B-4T')
tok_fixed = AutoTokenizer.from_pretrained('C:/ai_models/BitNet-b1.58-2B-4T', fix_mistral_regex=True)

SYSTEM_PROMPT_BASE = """You are a helpful AI assistant with access to tools.
When you need to use a tool to assist the user, you MUST respond ONLY with a JSON object in the following format:
```json
{"tool": "tool_name", "parameters": {"param1": "value1"}}
```
Do not include any extra thoughts or text before or after the JSON when invoking a tool.
If no tool is required, answer the user directly and concisely in natural language.

Available tools:
1. get_current_weather(location: str, unit: str = "celsius") - Retrieve current weather data for a specific city.
2. web_search(query: str, max_results: int = 5) - Search the web for up-to-date information.
3. execute_python(code: str) - Execute Python script in a secure sandbox and return standard output.
4. file_read(path: str) - Read text content of a local file.
5. query_database(sql: str) - Execute a read-only SQL query against the database."""

def format_prompt(messages, system_prompt):
    prompt = f"System: {system_prompt}<|eot_id|>"
    for m in messages:
        role = "User" if m["role"] == "user" else "Assistant"
        prompt += f"{role}: {m['content']}<|eot_id|>"
    prompt += "Assistant: "
    return prompt

with open('C:/ai_models/test_suite_standardized.json', encoding='utf-8') as f:
    cases = json.load(f)

print("=== COMPARING DEFAULT TOKENIZER VS FIX_MISTRAL_REGEX=TRUE ===")
for c in cases:
    full_prompt = format_prompt(c['messages'], SYSTEM_PROMPT_BASE)
    t1 = tok_default.encode(full_prompt, add_special_tokens=False)
    t2 = tok_fixed.encode(full_prompt, add_special_tokens=False)
    if t1 != t2:
        print(f"\nCase {c['id']}: len(default)={len(t1)}, len(fixed)={len(t2)}")
        diffs = [(i, x, y, tok_default.decode([x]), tok_fixed.decode([y])) for i, (x, y) in enumerate(zip(t1, t2)) if x != y]
        print(f"  First 3 diffs: {diffs[:3]}")
