import requests

# Start a quick server check or test with llama-cli
# Let's check how llama.cpp tokenizes with and without --special
prompt = """<|begin_of_text|>System: You are a helpful AI assistant with access to tools.
When you need to use a tool to assist the user, you MUST respond ONLY with a JSON object in the following format:
```json
{"tool": "tool_name", "parameters": {"param1": "value1"}}
```
Do not include any extra thoughts or text before or after the JSON when invoking a tool.
If no tool is required, answer the user directly and concisely in natural language.

Available tools:
1. get_current_weather(location: str, unit: str = "celsius") - Retrieve current weather data for a specific city.<|eot_id|>User: What is the temperature in London right now?<|eot_id|>Assistant: """

with open(r"C:\ai_models\test_special_prompt.txt", "w", encoding="utf-8") as f:
    f.write(prompt)

print("Saved prompt. Running test with --repeat-penalty 1.15 --special...")
