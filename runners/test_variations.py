import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = r"C:\ai_models\BitNet-b1.58-2B-4T"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="auto")

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

print("=== BITNET VARIATIONS TEST ===")
for q in test_queries:
    msgs = [{"role": "system", "content": system_prompt}, {"role": "user", "content": q}]
    prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=60, temperature=0.1)
    gen = out[0][inputs.input_ids.shape[1]:]
    resp = tokenizer.decode(gen, skip_special_tokens=True).strip()
    print(f"QUERY: {q}")
    print(f"RESPONSE:\n{resp}")
    print("---------------------------------")
