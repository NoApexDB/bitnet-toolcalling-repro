import urllib.request
import json
import time

url = "http://127.0.0.1:8088/completion"
SYSTEM_PROMPT = """You are a helpful AI assistant with access to tools.
When you need to use a tool to assist the user, you MUST respond ONLY with a JSON object in the following format:
```json
{"tool": "tool_name", "parameters": {"param1": "value1"}}
```
Available tools:
1. get_current_weather(location: str)
2. web_search(query: str)
3. execute_python(code: str)
4. file_read(path: str)
5. query_database(sql: str)
"""

USER_QUERY = "Какая сейчас погода в Токио?"

full_prompt = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{USER_QUERY}<|im_end|>\n<|im_start|>assistant\n"

data = {
    "prompt": full_prompt,
    "n_predict": 64,
    "temperature": 0.0,
    "repeat_penalty": 1.15,
    "stop": ["<|im_end|>", "```\n", "<|endoftext|>"],
    "stream": False
}
req = urllib.request.Request(
    url,
    data=json.dumps(data).encode("utf-8"),
    headers={"Content-Type": "application/json"}
)

t0 = time.time()
with urllib.request.urlopen(req) as resp:
    res = json.loads(resp.read().decode("utf-8"))
dt = time.time() - t0

print(f"Status Code: {resp.status}")
print(f"Content: {res.get('content')}")
timings = res.get("timings", {})
print(f"Timings: {timings}")
print(f"Total Wall Time: {dt:.3f} s")
if timings:
    p_spd = timings.get("prompt_per_second", 0)
    g_spd = timings.get("predicted_per_second", 0)
    print(f"Prompt Speed: {p_spd:.2f} t/s")
    print(f"Generation Speed: {g_spd:.2f} t/s")
