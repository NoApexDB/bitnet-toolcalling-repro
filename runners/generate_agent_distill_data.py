import json
import urllib.request
import os

OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
TEACHER_MODEL = "huihui_ai/qwen2.5-coder-abliterate:7b"
OUTPUT_FILE = r"C:\ai_models\agent_distill_dataset.jsonl"

AGENT_SYSTEM_PROMPT = """You are an autonomous AI agent capable of using tools and reasoning step-by-step.
When a query requires external data or action, output a JSON tool call:
```json
{"tool": "<tool_name>", "parameters": {<params>}}
```
When you receive the tool execution output, provide a clear, factual answer.
If no tool is required, answer directly without JSON."""

CATEGORIES = [
    {
        "category": "single_tool_weather_search",
        "instruction": "Сгенерируй 5 разнообразных реалистичных сценариев (запрос пользователя -> точный JSON вызов тулза -> результат тулза -> финальный ответ агента) для инструментов 'get_weather(city)' и 'web_search(query)'. Включи города мира, курсы валют, научные факты."
    },
    {
        "category": "system_and_code_ops",
        "instruction": "Сгенерируй 5 сценариев для инструментов 'execute_command(command)' (терминал PowerShell/bash) и 'read_file(path)'. Пользователь просит проверить статус процесса, найти файлы в папке, прочитать конфиг."
    },
    {
        "category": "math_and_database",
        "instruction": "Сгенерируй 5 сценариев для инструментов 'calculator(expression)' и 'database_query(query)'. Пользователь просит посчитать сложные проценты или найти клиента в базе."
    },
    {
        "category": "negative_no_tool",
        "instruction": "Сгенерируй 5 сценариев, где пользователь задает вопросы по теории программирования, логике, переводу или совету, где инструменты НЕ НУЖНЫ, и агент должен ответить прямо и уверенно БЕЗ вызова JSON."
    },
    {
        "category": "error_handling_recovery",
        "instruction": "Сгенерируй 5 сценариев, где инструмент вернул ошибку (например, город не найден, таймаут сети или файл отсутствует), и агент вежливо сообщает пользователю и предлагает корректное решение."
    }
]

def call_teacher(prompt: str) -> str:
    payload = {
        "model": TEACHER_MODEL,
        "messages": [
            {
                "role": "system",
                "content": "Ты экспертный генератор синтетических данных для дистилляции агентных LLM. Отвечай строго в формате JSON-списка объектов с полями: 'user_query', 'needs_tool' (bool), 'tool_call' (dict или null), 'tool_result' (dict или null), 'final_answer' (str). Без вступительного текста."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "stream": False,
        "options": {
            "temperature": 0.3,
            "num_predict": 1200
        }
    }
    req = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=90) as resp:
        res = json.loads(resp.read().decode("utf-8"))
        return res.get("message", {}).get("content", "").strip()

def extract_json_array(text: str):
    start = text.find('[')
    end = text.rfind(']')
    if start != -1 and end != -1 and end > start:
        candidate = text[start:end+1]
        try:
            return json.loads(candidate)
        except Exception:
            pass
    return None

def main():
    print(f"[Distillation DataGen] Запуск генерации обучающего корпуса через {TEACHER_MODEL}...")
    dataset = []

    for cat in CATEGORIES:
        print(f"\n[Категория: {cat['category']}] Запрос к учителю...")
        try:
            raw_response = call_teacher(cat["instruction"])
            items = extract_json_array(raw_response)
            if items and isinstance(items, list):
                print(f"-> Успешно получено {len(items)} примеров.")
                for item in items:
                    # Формируем ChatML структуру для обучения
                    messages = [{"role": "system", "content": AGENT_SYSTEM_PROMPT}]
                    messages.append({"role": "user", "content": item.get("user_query", "")})
                    
                    if item.get("needs_tool") and item.get("tool_call"):
                        # Шаг 1: Ассистент генерирует вызов тулза
                        tool_call_str = json.dumps(item["tool_call"], ensure_ascii=False)
                        messages.append({"role": "assistant", "content": f"```json\n{tool_call_str}\n```"})
                        # Шаг 2: Симуляция результата выполнения тулза
                        tool_res_str = json.dumps(item.get("tool_result", {}), ensure_ascii=False)
                        messages.append({"role": "user", "content": f"Tool execution result: {tool_res_str}"})
                        # Шаг 3: Финальный ответ ассистента
                        messages.append({"role": "assistant", "content": item.get("final_answer", "")})
                    else:
                        # Прямой ответ без инструментов
                        messages.append({"role": "assistant", "content": item.get("final_answer", "")})

                    dataset.append({
                        "category": cat["category"],
                        "messages": messages
                    })
            else:
                print("-> Не удалось распарсить JSON массив, сырой вывод:\n", raw_response[:200])
        except Exception as e:
            print(f"-> Ошибка при генерации категории {cat['category']}: {e}")

    # Сохраняем в JSONL
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for entry in dataset:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"\n[DONE] Всего сформировано {len(dataset)} валидных агентных диалогов в {OUTPUT_FILE}.")

if __name__ == "__main__":
    main()
