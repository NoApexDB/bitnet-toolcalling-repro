from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained('C:/ai_models/BitNet-b1.58-2B-4T')
SYSTEM_PROMPT = """You are a helpful AI assistant with access to tools.
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
5. query_database(sql: str) - Execute a read-only SQL query against the database.
"""

messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": "Какая сейчас погода в Токио?"}
]

txt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print("=== EXACT PROMPT START ===")
print(txt)
print("=== EXACT PROMPT END ===")
