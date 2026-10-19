import requests

session = requests.Session()
session.trust_env = False

system_prompt = """You are a helpful AI assistant with access to tools.
When you need to use a tool to assist the user, you MUST respond ONLY with a JSON object in the following format:
```json
{"tool": "tool_name", "parameters": {"param1": "value1"}}
```
Available tools:
1. query_database(sql: str) - Execute SQL query against database.
2. read_log_file(path: str) - Read server log file.

Example:
Human: Read config file /etc/nginx.conf
BITNETAssistant: ```json
{"tool": "read_log_file", "parameters": {"path": "/etc/nginx.conf"}}
```"""

user_prompt = "Query the database for all records in table audit_log where event is 'privilege_escalation'."

prompt = f"{system_prompt}\n\nHuman: {user_prompt}\n\nBITNETAssistant: ```json\n"

payload = {
    "prompt": prompt,
    "n_predict": 64,
    "temperature": 0.0,
    "repeat_penalty": 1.15,
    "stop": ["```", "<|end_of_text|>", "Human:"]
}

resp = session.post("http://127.0.0.1:8088/completion", json=payload, timeout=30.0)
data = resp.json()
print("OUTPUT:")
print(data.get("content", ""))
