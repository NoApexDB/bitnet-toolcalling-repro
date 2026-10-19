import argparse
import json
import os
import re
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

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

TEST_CASES = [
    # --- POSITIVE SINGLE-TURN CASES (1-12) ---
    {
        "id": 1,
        "category": "single_turn",
        "needs_tool": True,
        "expected_tool": "get_current_weather",
        "messages": [
            {"role": "user", "content": "Какая сейчас погода в Токио?"}
        ],
        "param_validator": lambda p: "tokyo" in str(p.get("location", "")).lower() or "токио" in str(p.get("location", "")).lower()
    },
    {
        "id": 2,
        "category": "single_turn",
        "needs_tool": True,
        "expected_tool": "get_current_weather",
        "messages": [
            {"role": "user", "content": "What is the temperature in London right now?"}
        ],
        "param_validator": lambda p: "london" in str(p.get("location", "")).lower()
    },
    {
        "id": 3,
        "category": "single_turn",
        "needs_tool": True,
        "expected_tool": "get_current_weather",
        "messages": [
            {"role": "user", "content": "Посмотри погоду в Берлине в градусах Фаренгейта"}
        ],
        "param_validator": lambda p: ("berlin" in str(p.get("location", "")).lower() or "берлин" in str(p.get("location", "")).lower()) and ("fahr" in str(p.get("unit", "")).lower())
    },
    {
        "id": 4,
        "category": "single_turn",
        "needs_tool": True,
        "expected_tool": "web_search",
        "messages": [
            {"role": "user", "content": "Найди последние новости про искусственный интеллект за сегодня"}
        ],
        "param_validator": lambda p: len(str(p.get("query", "")).strip()) > 5
    },
    {
        "id": 5,
        "category": "single_turn",
        "needs_tool": True,
        "expected_tool": "web_search",
        "messages": [
            {"role": "user", "content": "Какой сейчас курс биткоина к доллару?"}
        ],
        "param_validator": lambda p: any(w in str(p.get("query", "")).lower() for w in ["биткоин", "bitcoin", "btc"])
    },
    {
        "id": 6,
        "category": "single_turn",
        "needs_tool": True,
        "expected_tool": "web_search",
        "messages": [
            {"role": "user", "content": "Who won the recent UEFA Champions League match yesterday?"}
        ],
        "param_validator": lambda p: any(w in str(p.get("query", "")).lower() for w in ["champions", "uefa", "league", "match"])
    },
    {
        "id": 7,
        "category": "single_turn",
        "needs_tool": True,
        "expected_tool": "execute_python",
        "messages": [
            {"role": "user", "content": "Вычисли 50-е число Фибоначчи с помощью Python скрипта"}
        ],
        "param_validator": lambda p: "fib" in str(p.get("code", "")).lower() or "def " in str(p.get("code", "")) or "50" in str(p.get("code", ""))
    },
    {
        "id": 8,
        "category": "single_turn",
        "needs_tool": True,
        "expected_tool": "execute_python",
        "messages": [
            {"role": "user", "content": "Посчитай сумму всех простых чисел от 1 до 500 через Python"}
        ],
        "param_validator": lambda p: "500" in str(p.get("code", "")) or "prime" in str(p.get("code", "")).lower() or "range" in str(p.get("code", ""))
    },
    {
        "id": 9,
        "category": "single_turn",
        "needs_tool": True,
        "expected_tool": "file_read",
        "messages": [
            {"role": "user", "content": "Прочитай системный файл /etc/hosts"}
        ],
        "param_validator": lambda p: "/etc/hosts" in str(p.get("path", "")) or "hosts" in str(p.get("path", ""))
    },
    {
        "id": 10,
        "category": "single_turn",
        "needs_tool": True,
        "expected_tool": "file_read",
        "messages": [
            {"role": "user", "content": "Пожалуйста, открой и прочитай файл C:\\ai_models\\settings.json"}
        ],
        "param_validator": lambda p: "settings.json" in str(p.get("path", ""))
    },
    {
        "id": 11,
        "category": "single_turn",
        "needs_tool": True,
        "expected_tool": "query_database",
        "messages": [
            {"role": "user", "content": "Сделай SQL запрос для подсчета всех активных пользователей в таблице users"}
        ],
        "param_validator": lambda p: "select" in str(p.get("sql", "")).lower() and "users" in str(p.get("sql", "")).lower()
    },
    {
        "id": 12,
        "category": "single_turn",
        "needs_tool": True,
        "expected_tool": "query_database",
        "messages": [
            {"role": "user", "content": "Выведи топ 10 товаров по сумме продаж из таблицы orders"}
        ],
        "param_validator": lambda p: "select" in str(p.get("sql", "")).lower() and "orders" in str(p.get("sql", "")).lower()
    },

    # --- POSITIVE MULTI-TURN CONTEXT RETENTION CASES (13-15) ---
    {
        "id": 13,
        "category": "context_retention",
        "needs_tool": True,
        "expected_tool": "get_current_weather",
        "messages": [
            {"role": "user", "content": "На следующей неделе я лечу в командировку в Рим."},
            {"role": "assistant", "content": "Отличный город для командировки! Могу помочь с организацией или полезной информацией."},
            {"role": "user", "content": "Какая там сейчас погода?"}
        ],
        "param_validator": lambda p: "rome" in str(p.get("location", "")).lower() or "рим" in str(p.get("location", "")).lower()
    },
    {
        "id": 14,
        "category": "context_retention",
        "needs_tool": True,
        "expected_tool": "file_read",
        "messages": [
            {"role": "user", "content": "Я сохранил логи сервера в файл C:\\logs\\production_error.log."},
            {"role": "assistant", "content": "Понял, файл сохранен в директории логов. Чем могу помочь с анализом?"},
            {"role": "user", "content": "Пожалуйста, прочти его."}
        ],
        "param_validator": lambda p: "production_error.log" in str(p.get("path", "")) or "logs" in str(p.get("path", ""))
    },
    {
        "id": 15,
        "category": "context_retention",
        "needs_tool": True,
        "expected_tool": "web_search",
        "messages": [
            {"role": "user", "content": "Мы обсуждаем выход новой версии фреймворка PyTorch 2.5."},
            {"role": "assistant", "content": "PyTorch 2.5 включает множество улучшений компилятора TorchDynamo и оптимизаций памяти."},
            {"role": "user", "content": "Поищи в интернете дату официального релиза этой версии."}
        ],
        "param_validator": lambda p: "pytorch" in str(p.get("query", "")).lower() and ("2.5" in str(p.get("query", "")) or "release" in str(p.get("query", "")).lower() or "релиз" in str(p.get("query", "")).lower())
    },

    # --- NEGATIVE CONTROL CASES (NO TOOL NEEDED) (16-20) ---
    {
        "id": 16,
        "category": "negative_no_tool",
        "needs_tool": False,
        "expected_tool": None,
        "messages": [
            {"role": "user", "content": "Что такое градиентный спуск в машинном обучении?"}
        ],
        "param_validator": None
    },
    {
        "id": 17,
        "category": "negative_no_tool",
        "needs_tool": False,
        "expected_tool": None,
        "messages": [
            {"role": "user", "content": "Переведи фразу 'Artificial intelligence revolution' на русский язык."}
        ],
        "param_validator": None
    },
    {
        "id": 18,
        "category": "negative_no_tool",
        "needs_tool": False,
        "expected_tool": None,
        "messages": [
            {"role": "user", "content": "Объясни простыми словами разницу между структурой стек (Stack) и очередь (Queue)."}
        ],
        "param_validator": None
    },
    {
        "id": 19,
        "category": "negative_no_tool",
        "needs_tool": False,
        "expected_tool": None,
        "messages": [
            {"role": "user", "content": "Привет! Как поживаешь?"}
        ],
        "param_validator": None
    },
    {
        "id": 20,
        "category": "negative_no_tool",
        "needs_tool": False,
        "expected_tool": None,
        "messages": [
            {"role": "user", "content": "В чем главное концептуальное отличие синхронного выполнения от асинхронного?"}
        ],
        "param_validator": None
    }
]

def extract_tool_call(text):
    text = text.strip()
    candidate_texts = []

    # 1. Search for markdown code block ```json ... ``` or ``` ... ```
    match = re.search(r"```(?:json)?\s*(\{.*)", text, re.DOTALL)
    if match:
        candidate_texts.append(match.group(1))

    # 2. Search for any opening brace in the raw text
    start_pos = 0
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
            if isinstance(obj, dict) and "tool" in obj:
                return obj, True
        except Exception:
            pass

    # 3. Check if model clearly attempted a tool call but generated broken JSON
    tool_keywords = [
        '"tool"', "'tool'", "tool:", 
        "get_current_weather", "web_search", "execute_python", "file_read", "query_database"
    ]
    if any(k in text for k in tool_keywords):
        return None, True

    return None, False

def run_evaluation(variant, repetition_penalty=1.15):
    print(f"\n=======================================================", flush=True)
    print(f"   ЗАПУСК ОЦЕНКИ DUAL RECALL: [{variant.upper()}] (rep_penalty={repetition_penalty})", flush=True)
    print(f"=======================================================", flush=True)

    lora_path = None
    if variant == "bitnet_base":
        base_path = r"C:\ai_models\BitNet-b1.58-2B-4T"
        use_lora = False
    elif variant == "bitnet_distilled":
        base_path = r"C:\ai_models\BitNet-b1.58-2B-4T"
        lora_path = r"C:\ai_models\bitnet_agent_lora"
        use_lora = True
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

    print(f"[1/3] Загрузка токенизатора из {base_path}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(base_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[2/3] Загрузка базовой модели на GPU...", flush=True)
    load_dtype = torch.float32 if "BitNet" in base_path else torch.float16
    print(f"      Используемая точность (dtype): {load_dtype}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_path,
        torch_dtype=load_dtype,
        device_map="cuda:0"
    )

    if use_lora:
        print(f"[3/3] Подключение LoRA адаптера из {lora_path}...", flush=True)
        model = PeftModel.from_pretrained(model, lora_path)
    else:
        print(f"[3/3] Использование базовой модели БЕЗ адаптера...", flush=True)

    model.eval()

    results = []
    pos_cases = 0
    trigger_attempts = 0
    correct_executions = 0

    ctx_pos_cases = 0
    ctx_trigger_attempts = 0
    ctx_correct_executions = 0

    neg_cases = 0
    neg_direct_success = 0

    print("\n--- ТЕСТИРОВАНИЕ НА 20 СЦЕНАРИЯХ ---", flush=True)

    for case in TEST_CASES:
        case_id = case["id"]
        category = case["category"]
        needs_tool = case["needs_tool"]
        expected_tool = case["expected_tool"]
        messages = case["messages"]
        param_validator = case["param_validator"]

        full_dialog = [{"role": "system", "content": SYSTEM_PROMPT}] + messages
        prompt_text = tokenizer.apply_chat_template(full_dialog, tokenize=False, add_generation_prompt=True)

        inputs = tokenizer(prompt_text, return_tensors="pt").to("cuda:0")

        with torch.no_grad():
            output_tokens = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                repetition_penalty=repetition_penalty,
                pad_token_id=tokenizer.eos_token_id
            )

        gen_tokens = output_tokens[0][inputs["input_ids"].shape[1]:]
        response_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

        parsed_tool, attempted = extract_tool_call(response_text)

        test_record = {
            "id": case_id,
            "category": category,
            "needs_tool": needs_tool,
            "expected_tool": expected_tool,
            "user_query": messages[-1]["content"],
            "model_response": response_text,
            "attempted_tool": attempted,
            "parsed_call": parsed_tool,
            "trigger_success": False,
            "execution_success": False,
            "negative_success": False
        }

        if needs_tool:
            pos_cases += 1
            if category == "context_retention":
                ctx_pos_cases += 1

            if attempted:
                trigger_attempts += 1
                test_record["trigger_success"] = True
                if category == "context_retention":
                    ctx_trigger_attempts += 1

                if parsed_tool is not None:
                    actual_tool = parsed_tool.get("tool")
                    params = parsed_tool.get("parameters", {})
                    if actual_tool == expected_tool:
                        if param_validator is None or param_validator(params):
                            correct_executions += 1
                            test_record["execution_success"] = True
                            if category == "context_retention":
                                ctx_correct_executions += 1
        else:
            neg_cases += 1
            if not attempted:
                neg_direct_success += 1
                test_record["negative_success"] = True

        status_str = "CORRECT" if (test_record["execution_success"] or test_record["negative_success"]) else "FAILED"
        print(f"[{case_id:02d}/20] ({category.upper()}) | Expected: {expected_tool or 'Direct Text'} | Status: {status_str}", flush=True)

        results.append(test_record)

    trigger_recall = (trigger_attempts / pos_cases * 100) if pos_cases > 0 else 0.0
    execution_recall = (correct_executions / trigger_attempts * 100) if trigger_attempts > 0 else 0.0
    end_to_end_accuracy = (correct_executions / pos_cases * 100) if pos_cases > 0 else 0.0

    ctx_trigger_recall = (ctx_trigger_attempts / ctx_pos_cases * 100) if ctx_pos_cases > 0 else 0.0
    ctx_execution_recall = (ctx_correct_executions / ctx_trigger_attempts * 100) if ctx_trigger_attempts > 0 else 0.0

    neg_precision = (neg_direct_success / neg_cases * 100) if neg_cases > 0 else 0.0

    summary = {
        "variant": variant,
        "repetition_penalty": repetition_penalty,
        "total_test_cases": len(TEST_CASES),
        "positive_cases": pos_cases,
        "trigger_attempts": trigger_attempts,
        "trigger_recall_pct": round(trigger_recall, 2),
        "correct_executions": correct_executions,
        "execution_recall_pct": round(execution_recall, 2),
        "end_to_end_accuracy_pct": round(end_to_end_accuracy, 2),
        "context_retention": {
            "cases": ctx_pos_cases,
            "trigger_recall_pct": round(ctx_trigger_recall, 2),
            "execution_recall_pct": round(ctx_execution_recall, 2)
        },
        "negative_cases": neg_cases,
        "negative_direct_success": neg_direct_success,
        "negative_accuracy_pct": round(neg_precision, 2),
        "detailed_results": results
    }

    out_file = f"C:\\ai_models\\benchmark_results_{variant}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    os.makedirs(r"C:\ai_models\paper_artifacts", exist_ok=True)
    with open(f"C:\\ai_models\\paper_artifacts\\benchmark_results_{variant}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n[{variant.upper()}] Trigger Recall: {trigger_recall:.2f}% | Exec Recall: {execution_recall:.2f}% | Ctx Exec: {ctx_execution_recall:.2f}% | Neg Acc: {neg_precision:.2f}%", flush=True)

    # Free memory
    del model
    del tokenizer
    torch.cuda.empty_cache()

    return summary

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", type=str, required=True, choices=["bitnet_base", "bitnet_distilled", "bitnet_balanced", "qwen_base", "qwen_distilled", "all"])
    parser.add_argument("--repetition_penalty", type=float, default=1.15, help="Repetition penalty for generation (default: 1.15)")
    args = parser.parse_args()

    if args.variant == "all":
        variants = ["bitnet_base", "bitnet_distilled", "qwen_base", "qwen_distilled"]
    else:
        variants = [args.variant]

    all_summaries = {}
    for var in variants:
        summary = run_evaluation(var, repetition_penalty=args.repetition_penalty)
        all_summaries[var] = summary

    if len(variants) > 1:
        print("\n" + "="*80)
        print("                 ИТОГОВАЯ СВОДНАЯ ТАБЛИЦА СРАВНЕНИЯ")
        print("="*80)
        header = f"{'Модель / Вариант':<20} | {'Trigger Rec':<12} | {'Exec Rec':<10} | {'End-to-End':<10} | {'Ctx Exec':<10} | {'Neg Acc':<10}"
        print(header)
        print("-" * len(header))
        for var, s in all_summaries.items():
            print(f"{var:<20} | {s['trigger_recall_pct']:>10.2f}% | {s['execution_recall_pct']:>8.2f}% | {s['end_to_end_accuracy_pct']:>8.2f}% | {s['context_retention']['execution_recall_pct']:>8.2f}% | {s['negative_accuracy_pct']:>8.2f}%")
        print("="*80)

        # Save unified report
        with open(r"C:\ai_models\benchmark_comparison_matrix.json", "w", encoding="utf-8") as f:
            json.dump(all_summaries, f, ensure_ascii=False, indent=2)
        with open(r"C:\ai_models\paper_artifacts\benchmark_comparison_matrix.json", "w", encoding="utf-8") as f:
            json.dump(all_summaries, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
