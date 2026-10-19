import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

base_path = "C:/ai_models/BitNet-b1.58-2B-4T"
tok = AutoTokenizer.from_pretrained(base_path)
model = AutoModelForCausalLM.from_pretrained(base_path, torch_dtype=torch.float32, device_map="cuda:0")
model.eval()

SYSTEM_PROMPT = """You are a helpful AI assistant with access to tools.
When you need to use a tool to assist the user, you MUST respond ONLY with a JSON object in the following format:
```json
{"tool": "tool_name", "parameters": {"param1": "value1"}}
```
Do not include any extra thoughts or text before or after the JSON when invoking a tool.
If no tool is required, answer the user directly and concisely in natural language.

Available tools:
1. get_current_weather(location: str, unit: str = "celsius") - Retrieve current weather data for a specific city."""

dialog = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": "What is the temperature in London right now?"}
]

prompt_text = tok.apply_chat_template(dialog, tokenize=False, add_generation_prompt=True)
print("=== APPLIED CHAT TEMPLATE ===")
print(repr(prompt_text))

inputs = tok(prompt_text, return_tensors="pt").to("cuda:0")
print("=== INPUT TOKEN IDS (first 20) ===")
print(inputs.input_ids[0][:20])

with torch.no_grad():
    out = model.generate(
        **inputs,
        max_new_tokens=64,
        do_sample=False,
        repetition_penalty=1.15,
        pad_token_id=tok.eos_token_id
    )

gen_tokens = out[0][inputs.input_ids.shape[1]:]
print("=== PYTORCH GENERATED TOKENS ===")
print(tok.decode(gen_tokens))
