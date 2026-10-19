import json
import urllib.request

tools_desc = json.dumps([
    {"name": "get_weather", "description": "Get current weather condition and temperature for a city.", "parameters": {"city": "string"}},
    {"name": "web_search", "description": "Search the web for up-to-date facts, news, or general search queries.", "parameters": {"query": "string"}}
], indent=2)

system_prompt = f"""You are an AI assistant with access to tools.
Tools:
{tools_desc}

Rule: If a tool is needed, respond ONLY with a JSON object in format:
{{"tool": "<tool_name>", "parameters": {{"<param>": "<value>"}}}}
If no tool is needed, answer directly in plain text."""

test_queries = [
    "What is the weather in London right now?",
    "Search the web: What is the latest stock price of Apple?",
    "Найди в интернете: курс доллара к рублю сегодня",
    "What is 15 + 27?"
]

print("=== QWEN 1.5B VARIATIONS TEST ===")
for q in test_queries:
    msgs = [{"role": "system", "content": system_prompt}, {"role": "user", "content": q}]
    data = json.dumps({
        "model": "qwen2.5:1.5b",
        "messages": msgs,
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": 60}
    }).encode("utf-8")

    req = urllib.request.Request(
        "http://127.0.0.1:11434/api/chat",
        data=data,
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        res = json.loads(resp.read().decode("utf-8"))
        resp_text = res.get("message", {}).get("content", "").strip()
    print(f"QUERY: {q}")
    print(f"RESPONSE:\n{resp_text}")
    print("---------------------------------")
