import json
import re
import urllib.request
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- 1. ОПРЕДЕЛЕНИЕ ИНСТРУМЕНТОВ (TOOLS) ---

TOOLS = [
    {
        "name": "get_weather",
        "description": "Get current weather condition and temperature for a given city.",
        "parameters": {
            "city": "string, name of the city (e.g., 'Moscow', 'London', 'Tokyo')"
        }
    },
    {
        "name": "web_search",
        "description": "Search the internet for latest news, historical facts, or general knowledge.",
        "parameters": {
            "query": "string, keywords to search for"
        }
    }
]

def execute_tool(tool_name: str, parameters: dict) -> dict:
    """Имитация выполнения инструментов (mock runtime)."""
    if tool_name == "get_weather":
        city = parameters.get("city", "Unknown")
        mock_weather = {
            "Moscow": {"temp": "-2°C", "condition": "Snowy", "humidity": "85%"},
            "Tokyo": {"temp": "+16°C", "condition": "Sunny", "humidity": "50%"},
            "Paris": {"temp": "+11°C", "condition": "Rainy", "humidity": "78%"},
        }
        data = mock_weather.get(city, {"temp": "+15°C", "condition": "Cloudy", "humidity": "60%"})
        return {"city": city, **data}

    elif tool_name == "web_search":
        query = parameters.get("query", "").lower()
        if "bitnet" in query or "1.58" in query:
            return {"result": "BitNet b1.58 is a 1-bit LLM architecture from Microsoft using ternary weights {-1, 0, 1}."}
        elif "president" in query or "france" in query:
            return {"result": "The current President of France is Emmanuel Macron."}
        else:
            return {"result": f"Found 3 relevant articles for query '{query}'."}

    return {"error": f"Unknown tool: {tool_name}"}


# --- 2. СИСТЕМНЫЙ ПРОМПТ ДЛЯ МАЛЫХ МОДЕЛЕЙ (1.5B - 2B) ---

SYSTEM_PROMPT = f"""You are an intelligent AI agent with access to external tools.

Available tools:
{json.dumps(TOOLS, indent=2)}

INSTRUCTIONS:
1. If the user question requires external data (weather or web search), call the appropriate tool.
2. To call a tool, you MUST output ONLY a valid JSON object in this format:
```json
{{"tool": "tool_name", "parameters": {{"param_name": "value"}}}}
```
3. If no tool is needed, answer the user directly without JSON.
"""

def extract_tool_call(text: str):
    """Парсинг JSON вызова инструмента из вывода модели (с учетом вложенных фигурных скобок)."""
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1 and end > start:
        candidate = text[start:end+1]
        try:
            parsed = json.loads(candidate)
            if "tool" in parsed and "parameters" in parsed:
                return parsed
        except Exception:
            pass
    return None


# --- 3. РАНТАЙМЫ МОДЕЛЕЙ ---

class BitNetAgent:
    def __init__(self, model_path=r"C:\ai_models\BitNet-b1.58-2B-4T"):
        print("[BitNet] Загрузка токенизатора и модели...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        print("[BitNet] Модель готова к работе.")

    def generate(self, messages, max_new_tokens=100) -> str:
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=0.1)
        # декодируем только новые сгенерированные токены
        generated_tokens = outputs[0][inputs.input_ids.shape[1]:]
        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()


class QwenOllamaAgent:
    def __init__(self, model_name="qwen2.5:1.5b"):
        self.model_name = model_name

    def generate(self, messages, max_new_tokens=100) -> str:
        # Преобразуем сообщения для Ollama API
        data = json.dumps({
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": max_new_tokens}
        }).encode("utf-8")

        req = urllib.request.Request(
            "http://127.0.0.1:11434/api/chat",
            data=data,
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            res = json.loads(resp.read().decode("utf-8"))
            return res.get("message", {}).get("content", "").strip()


# --- 4. ТЕСТОВЫЙ СЦЕНАРИЙ ДЛЯ ИССЛЕДОВАНИЯ ---

TEST_CASES = [
    {
        "id": "weather_test",
        "query": "Какая сейчас погода в Токио?",
        "expected_tool": "get_weather",
        "expected_param_key": "city"
    },
    {
        "id": "search_test",
        "query": "Что такое архитектура BitNet 1.58 бит?",
        "expected_tool": "web_search",
        "expected_param_key": "query"
    },
    {
        "id": "direct_qa_test",
        "query": "Сколько будет 25 умножить на 4?",
        "expected_tool": None  # Инструмент не нужен, прямой ответ
    }
]

def run_evaluation(agent, agent_name: str):
    print(f"\n==========================================")
    print(f"   ЗАПУСК БЕНЧМАРКА: {agent_name}")
    print(f"==========================================")

    for case in TEST_CASES:
        print(f"\n[Тест {case['id']}] Запрос: '{case['query']}'")
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": case["query"]}
        ]

        response = agent.generate(messages, max_new_tokens=80)
        print(f"-> Сырой ответ модели:\n{response}")

        tool_call = extract_tool_call(response)
        if case["expected_tool"] is None:
            if tool_call is None:
                print("[OK] Direct answer without tool call.")
            else:
                print(f"[FAIL] Unnecessary tool call: {tool_call.get('tool')}")
        else:
            if tool_call and tool_call.get("tool") == case["expected_tool"]:
                params = tool_call.get("parameters", {})
                if case["expected_param_key"] in params:
                    print(f"[OK] Correct tool '{tool_call['tool']}' with params: {params}")
                    # Simulation
                    tool_result = execute_tool(tool_call["tool"], params)
                    print(f"  [Tool Result] {tool_result}")

                    messages.append({"role": "assistant", "content": response})
                    messages.append({"role": "user", "content": f"Tool execution result: {json.dumps(tool_result)}. Now answer my question based on this data."})
                    final_ans = agent.generate(messages, max_new_tokens=100)
                    print(f"  [Final Answer] {final_ans}")
                else:
                    print(f"[PARTIAL] Correct tool, missing param '{case['expected_param_key']}': {params}")
            else:
                print(f"[FAIL] Expected tool '{case['expected_tool']}', got: {tool_call}")

if __name__ == "__main__":
    # Тест 1: Qwen 2.5 1.5B (Dense) через Ollama
    qwen = QwenOllamaAgent(model_name="qwen2.5:1.5b")
    run_evaluation(qwen, "Qwen 2.5 1.5B (Dense)")

    # Тест 2: Microsoft BitNet b1.58 2B (Ternary)
    bitnet = BitNetAgent()
    run_evaluation(bitnet, "Microsoft BitNet b1.58 2B (Ternary)")
