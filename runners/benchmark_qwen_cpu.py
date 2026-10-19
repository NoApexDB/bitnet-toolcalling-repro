import time
import torch
import psutil
import os
from transformers import AutoModelForCausalLM, AutoTokenizer

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

MODEL_PATH = r"C:\ai_models\Qwen2.5-1.5B-Instruct"

torch.set_num_threads(4)

print("[1/3] Загрузка токенизатора и модели Qwen 1.5B на CPU (torch.float32)...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
t_load_0 = time.time()
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float32,
    device_map="cpu"
)
load_time = time.time() - t_load_0

process = psutil.Process(os.getpid())
ram_mb = process.memory_info().rss / (1024 * 1024)

print(f"[2/3] Модель загружена за {load_time:.2f} с. Потребление RAM: {ram_mb:.1f} МБ", flush=True)

messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": USER_QUERY}
]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt")

print(f"[3/3] Генерация 42 токенов на CPU (4 потока)...", flush=True)
t_gen_0 = time.time()
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=42,
        do_sample=False,
        repetition_penalty=1.15
    )
gen_time = time.time() - t_gen_0

gen_tokens = outputs[0][inputs["input_ids"].shape[1]:]
response_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

ram_after_mb = process.memory_info().rss / (1024 * 1024)
tokens_generated = len(gen_tokens)
speed_tps = tokens_generated / gen_time

print(f"Ответ: {response_text}")
print(f"Токенов сгенерировано: {tokens_generated}")
print(f"Время генерации: {gen_time:.3f} с")
print(f"Скорость на CPU: {speed_tps:.2f} t/s")
print(f"Итоговое потребление RAM: {ram_after_mb:.1f} МБ ({ram_after_mb/1024:.2f} ГБ)")
