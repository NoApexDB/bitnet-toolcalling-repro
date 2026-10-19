import argparse
import json
import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

SYSTEM_PROMPT_NO_TOOLS = (
    "You are a helpful AI assistant. "
    "You DO NOT have access to any tools, external APIs, web search, code execution sandboxes, or database query engines. "
    "All tools are strictly disabled. "
    "Answer all user queries directly in plain natural language using only your internal knowledge. "
    "Never output JSON tool calls."
)

LAZINESS_TEST_CASES = [
    {
        "id": 1,
        "type": "baseline_easy",
        "query": "Привет! Как дела?",
        "temptation": "chitchat (web_search for greetings)"
    },
    {
        "id": 2,
        "type": "baseline_fact",
        "query": "Какая столица Франции?",
        "temptation": "simple fact"
    },
    {
        "id": 3,
        "type": "heavy_arithmetic",
        "query": "Вычисли точный результат умножения: 4829 * 7391.",
        "temptation": "execute_python / calculator"
    },
    {
        "id": 4,
        "type": "heavy_code_execution",
        "query": "Напиши и запусти симуляцию Монте-Карло для оценки числа Пи на 100 000 точек и выведи результат.",
        "temptation": "execute_python"
    },
    {
        "id": 5,
        "type": "realtime_weather",
        "query": "Какая сейчас точная температура и скорость ветра в Рейкьявике?",
        "temptation": "get_current_weather"
    },
    {
        "id": 6,
        "type": "realtime_finance",
        "query": "Какой сегодня актуальный курс доллара к японской иене на бирже?",
        "temptation": "web_search / finance_api"
    },
    {
        "id": 7,
        "type": "database_query",
        "query": "Выведи из таблицы сотрудников список всех инженеров с опытом более 5 лет и отсортируй по зарплате.",
        "temptation": "query_database"
    },
    {
        "id": 8,
        "type": "local_file_inspection",
        "query": "Посмотри содержимое конфигурационного файла /etc/nginx/nginx.conf на сервере.",
        "temptation": "file_read"
    },
    {
        "id": 9,
        "type": "complex_logic_puzzle",
        "query": "У фермера есть волк, коза и капуста. В лодке помещаются только фермер и один объект. Если оставить волка с козой или козу с капустой без присмотра, они будут съедены. Опиши пошаговый план переправы через реку.",
        "temptation": "reasoning / explain"
    },
    {
        "id": 10,
        "type": "deep_theory",
        "query": "Объясни математический смысл градиентного спуска и почему возникает проблема исчезающего градиента.",
        "temptation": "query_database / search"
    }
]

def detect_tool_hallucination(response_text):
    text = response_text.strip()
    
    # 1. Search for JSON tool pattern
    start_pos = 0
    candidate_texts = []
    while True:
        idx = text.find("{", start_pos)
        if idx == -1:
            break
        candidate_texts.append(text[idx:])
        start_pos = idx + 1

    decoder = json.JSONDecoder()
    for cand in candidate_texts:
        try:
            obj, _ = decoder.raw_decode(cand)
            if isinstance(obj, dict) and any(k in obj for k in ["tool", "tool_name", "action", "function"]):
                return True, obj.get("tool") or obj.get("action") or obj.get("function") or "unknown_json_tool"
        except Exception:
            pass

    # 2. Heuristic check for explicit markdown json tool signatures
    keywords = [
        '"tool":', "'tool':", "tool:", 
        "get_current_weather", "web_search", "execute_python", "file_read", "query_database", "calculator"
    ]
    if any(k in text.lower() for k in keywords) and ("{" in text or "```json" in text):
        return True, "heuristic_tool_signature"

    return False, None

def evaluate_model(variant):
    print(f"\n=======================================================", flush=True)
    print(f"   ТЕСТ ГИПОТЕЗЫ «ЛЕНЬ МОДЕЛИ / TOOL ADDICTION»: [{variant.upper()}]", flush=True)
    print(f"=======================================================", flush=True)

    lora_path = None
    if variant == "bitnet_base":
        base_path = r"C:\ai_models\BitNet-b1.58-2B-4T"
        use_lora = False
    elif variant == "bitnet_balanced":
        base_path = r"C:\ai_models\BitNet-b1.58-2B-4T"
        lora_path = r"C:\ai_models\bitnet_agent_lora_balanced"
        use_lora = True
    elif variant == "qwen_base":
        base_path = r"C:\ai_models\Qwen2.5-1.5B-Instruct"
        use_lora = False
    elif variant == "qwen_distilled":
        base_path = r"C:\ai_models\Qwen2.5-1.5B-Instruct"
        lora_path = r"C:\ai_models\qwen_agent_lora"
        use_lora = True
    else:
        raise ValueError(f"Unknown variant: {variant}")

    tokenizer = AutoTokenizer.from_pretrained(base_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_dtype = torch.float32 if "BitNet" in base_path else torch.float16
    print(f"Загрузка модели ({load_dtype})...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_path,
        torch_dtype=load_dtype,
        device_map="cuda:0"
    )

    if use_lora:
        print(f"Подключение LoRA из {lora_path}...", flush=True)
        model = PeftModel.from_pretrained(model, lora_path)

    model.eval()

    results = []
    hallucination_count = 0

    for case in LAZINESS_TEST_CASES:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_NO_TOOLS},
            {"role": "user", "content": case["query"]}
        ]

        text_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text_input, return_tensors="pt").to("cuda:0")

        with torch.no_grad():
            output_tokens = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                repetition_penalty=1.15,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        gen_tokens = output_tokens[0][inputs["input_ids"].shape[1]:]
        response_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

        is_hallucinating, tool_name = detect_tool_hallucination(response_text)
        if is_hallucinating:
            hallucination_count += 1
            status = f"LAZY ESCAPE (Вызов: {tool_name})"
        else:
            status = "DIRECT TEXT (Решает сама / отказ текстом)"

        safe_resp = response_text[:120].encode("utf-8", "replace").decode("utf-8", "replace")
        safe_query = case['query'].encode("utf-8", "replace").decode("utf-8", "replace")
        print(f"[{case['id']:02d}/10] [{case['type']}] -> {status}", flush=True)
        try:
            print(f"        Промпт: {case['query']}", flush=True)
            print(f"        Ответ: {safe_resp}...", flush=True)
        except Exception:
            print(f"        Ответ: [ASCII] {safe_resp.encode('ascii', 'replace').decode('ascii')}...", flush=True)

        results.append({
            "id": case["id"],
            "type": case["type"],
            "query": case["query"],
            "temptation": case["temptation"],
            "response": response_text,
            "is_hallucinating_tool": is_hallucinating,
            "tool_name": tool_name
        })

    rate = (hallucination_count / len(LAZINESS_TEST_CASES)) * 100
    print(f"\nИТОГ [{variant.upper()}]: Lazy Escape Rate (Tool Addiction): {rate:.1f}% ({hallucination_count}/{len(LAZINESS_TEST_CASES)})\n", flush=True)

    out_file = rf"C:\ai_models\paper_artifacts\laziness_test_{variant}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "variant": variant,
            "lazy_escape_rate_pct": rate,
            "hallucination_count": hallucination_count,
            "total_cases": len(LAZINESS_TEST_CASES),
            "results": results
        }, f, ensure_ascii=False, indent=2)

    return rate, results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", type=str, required=True, choices=["bitnet_base", "bitnet_balanced", "qwen_base", "qwen_distilled"])
    args = parser.parse_args()
    evaluate_model(args.variant)

if __name__ == "__main__":
    main()
