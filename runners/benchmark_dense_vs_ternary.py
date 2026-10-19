import json
import os
import re
import shutil
import subprocess
import time
import psutil
import requests

SERVER_PORT = 8088
SERVER_URL = f"http://127.0.0.1:{SERVER_PORT}/completion"
LLAMA_SERVER_EXE = r"C:\ai_models\BitNet\build\bin\llama-server.exe"

SYSTEM_PROMPT_BASE = """You are a helpful AI assistant with access to tools.
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
5. query_database(sql: str) - Execute a read-only SQL query against the database."""

SYSTEM_PROMPT_CONTRASTIVE = """You are a helpful AI assistant with access to tools.
When you need to use a tool, you MUST respond ONLY with a JSON object in the following format:
```json
{"tool": "tool_name", "parameters": {"param1": "value1"}}
```
If no tool is required, answer the user directly and concisely in natural language without using JSON.

Examples:
User: Какая погода в Риме?
Assistant: ```json
{"tool": "get_current_weather", "parameters": {"location": "Rome"}}
```

User: Что такое градиентный спуск?
Assistant: Градиентный спуск — это метод оптимизации для поиска локального минимума функции потерь.

Available tools:
1. get_current_weather(location: str, unit: str = "celsius") - Retrieve current weather data for a specific city.
2. web_search(query: str, max_results: int = 5) - Search the web for up-to-date information.
3. execute_python(code: str) - Execute Python script in a secure sandbox and return standard output.
4. file_read(path: str) - Read text content of a local file.
5. query_database(sql: str) - Execute a read-only SQL query against the database."""

SYSTEM_PROMPT_BLOATED = """You are an enterprise-grade multi-tier cognitive decision-making and semantic reasoning engine deployed to evaluate complex user intentions across diverse domains including computational sciences, software engineering, linguistics, distributed databases, and general knowledge. When receiving a prompt, you must engage in thorough internal contextual analysis to determine whether the request pertains to factual conceptual explanations that should be answered natively, or whether it explicitly requires external operational intervention. If and only if an external side-effect or live retrieval is strictly mandatory, you must invoke one of the registered tools strictly adhering to the schema below. In all other scenarios, including conversational queries, code syntax questions, linguistic translations, and conceptual explanations, you must refuse tool invocation and formulate an eloquent, natural language textual response without injecting any JSON formatting, delimiters, or schema artifacts.

Available tools:
1. get_current_weather(location: str, unit: str = "celsius") - Retrieve current weather data for a specific city.
2. web_search(query: str, max_results: int = 5) - Search the web for up-to-date information.
3. execute_python(code: str) - Execute Python script in a secure sandbox and return standard output.
4. file_read(path: str) - Read text content of a local file.
5. query_database(sql: str) - Execute a read-only SQL query against the database.

JSON format:
```json
{"tool": "tool_name", "parameters": {"param1": "value1"}}
```"""

SYSTEM_PROMPT_NO_TOOL = """You are a helpful AI assistant with access to tools.
When you need to use a tool to assist the user, you MUST respond ONLY with a JSON object:
```json
{"tool": "tool_name", "parameters": {"param1": "value1"}}
```
CRITICAL RULE: If no tool is required, you MUST begin your response with "NO_TOOL: " followed by your answer in plain text.

Available tools:
1. get_current_weather(location: str, unit: str = "celsius") - Retrieve current weather data for a specific city.
2. web_search(query: str, max_results: int = 5) - Search the web for up-to-date information.
3. execute_python(code: str) - Execute Python script in a secure sandbox and return standard output.
4. file_read(path: str) - Read text content of a local file.
5. query_database(sql: str) - Execute a read-only SQL query against the database."""

TEST_CASES = [
    {"id": 1, "category": "single_turn", "needs_tool": True, "expected_tool": "get_current_weather",
     "messages": [{"role": "user", "content": "Какая сейчас погода в Токио?"}],
     "param_validator": lambda p: "tokyo" in str(p.get("location", "")).lower() or "токио" in str(p.get("location", "")).lower()},
    {"id": 2, "category": "single_turn", "needs_tool": True, "expected_tool": "get_current_weather",
     "messages": [{"role": "user", "content": "What is the temperature in London right now?"}],
     "param_validator": lambda p: "london" in str(p.get("location", "")).lower()},
    {"id": 3, "category": "single_turn", "needs_tool": True, "expected_tool": "get_current_weather",
     "messages": [{"role": "user", "content": "Посмотри погоду в Берлине в градусах Фаренгейта"}],
     "param_validator": lambda p: ("berlin" in str(p.get("location", "")).lower() or "берлин" in str(p.get("location", "")).lower()) and ("fahr" in str(p.get("unit", "")).lower())},
    {"id": 4, "category": "single_turn", "needs_tool": True, "expected_tool": "web_search",
     "messages": [{"role": "user", "content": "Кто стал чемпионом мира по футболу в 2022 году?"}],
     "param_validator": lambda p: "2022" in str(p.get("query", "")) or "футбол" in str(p.get("query", "")).lower() or "world cup" in str(p.get("query", "")).lower()},
    {"id": 5, "category": "single_turn", "needs_tool": True, "expected_tool": "web_search",
     "messages": [{"role": "user", "content": "Найди последние новости об искусственном интеллекте за эту неделю"}],
     "param_validator": lambda p: "искусствен" in str(p.get("query", "")).lower() or "ai" in str(p.get("query", "")).lower() or "intelligence" in str(p.get("query", "")).lower() or "новости" in str(p.get("query", "")).lower()},
    {"id": 6, "category": "single_turn", "needs_tool": True, "expected_tool": "web_search",
     "messages": [{"role": "user", "content": "Поищи в интернете информацию о BitNet b1.58 от Microsoft"}],
     "param_validator": lambda p: "bitnet" in str(p.get("query", "")).lower()},
    {"id": 7, "category": "single_turn", "needs_tool": True, "expected_tool": "execute_python",
     "messages": [{"role": "user", "content": "Вычисли 50-е число Фибоначчи с помощью Python скрипта"}],
     "param_validator": lambda p: "fib" in str(p.get("code", "")).lower() or "def " in str(p.get("code", "")) or "50" in str(p.get("code", ""))},
    {"id": 8, "category": "single_turn", "needs_tool": True, "expected_tool": "execute_python",
     "messages": [{"role": "user", "content": "Посчитай сумму всех простых чисел от 1 до 500 через Python"}],
     "param_validator": lambda p: "500" in str(p.get("code", "")) or "prime" in str(p.get("code", "")).lower() or "range" in str(p.get("code", ""))},
    {"id": 9, "category": "single_turn", "needs_tool": True, "expected_tool": "file_read",
     "messages": [{"role": "user", "content": "Прочитай системный файл /etc/hosts"}],
     "param_validator": lambda p: "/etc/hosts" in str(p.get("path", "")) or "hosts" in str(p.get("path", ""))},
    {"id": 10, "category": "single_turn", "needs_tool": True, "expected_tool": "file_read",
     "messages": [{"role": "user", "content": "Пожалуйста, открой и прочитай файл C:\\ai_models\\settings.json"}],
     "param_validator": lambda p: "settings.json" in str(p.get("path", ""))},
    {"id": 11, "category": "single_turn", "needs_tool": True, "expected_tool": "query_database",
     "messages": [{"role": "user", "content": "Сделай SQL запрос для подсчета всех активных пользователей в таблице users"}],
     "param_validator": lambda p: "select" in str(p.get("sql", "")).lower() and "users" in str(p.get("sql", "")).lower()},
    {"id": 12, "category": "single_turn", "needs_tool": True, "expected_tool": "query_database",
     "messages": [{"role": "user", "content": "Выполни SQL: найди топ 5 заказов с наибольшей суммой из таблицы orders"}],
     "param_validator": lambda p: "orders" in str(p.get("sql", "")).lower() and ("order by" in str(p.get("sql", "")).lower() or "desc" in str(p.get("sql", "")).lower() or "5" in str(p.get("sql", "")))},
    
    # Context Retention
    {"id": 13, "category": "context_retention", "needs_tool": True, "expected_tool": "get_current_weather",
     "messages": [
         {"role": "user", "content": "Я планирую поездку в Париж на следующей неделе."},
         {"role": "assistant", "content": "Париж прекрасен в любое время года! Чем могу помочь по подготовке к поездке?"},
         {"role": "user", "content": "Какая там сейчас погода?"}
     ],
     "param_validator": lambda p: "paris" in str(p.get("location", "")).lower() or "париж" in str(p.get("location", "")).lower()},
    {"id": 14, "category": "context_retention", "needs_tool": True, "expected_tool": "file_read",
     "messages": [
         {"role": "user", "content": "У нас произошел сбой на сервере. Лог ошибки сохранен по пути /var/log/production_error.log."},
         {"role": "assistant", "content": "Понял, зафиксировал путь к логу сбоя."},
         {"role": "user", "content": "Пожалуйста, прочти его."}
     ],
     "param_validator": lambda p: "production_error.log" in str(p.get("path", "")) or "logs" in str(p.get("path", ""))},
    {"id": 15, "category": "context_retention", "needs_tool": True, "expected_tool": "web_search",
     "messages": [
         {"role": "user", "content": "Мы обсуждаем выход новой версии фреймворка PyTorch 2.5."},
         {"role": "assistant", "content": "PyTorch 2.5 включает множество улучшений компилятора TorchDynamo и оптимизаций памяти."},
         {"role": "user", "content": "Поищи в интернете дату официального релиза этой версии."}
     ],
     "param_validator": lambda p: "pytorch" in str(p.get("query", "")).lower() and ("2.5" in str(p.get("query", "")) or "release" in str(p.get("query", "")).lower() or "релиз" in str(p.get("query", "")).lower())},

    # Negative Control
    {"id": 16, "category": "negative_no_tool", "needs_tool": False, "expected_tool": None,
     "messages": [{"role": "user", "content": "Что такое градиентный спуск в машинном обучении?"}],
     "param_validator": None},
    {"id": 17, "category": "negative_no_tool", "needs_tool": False, "expected_tool": None,
     "messages": [{"role": "user", "content": "Переведи фразу 'Artificial intelligence revolution' на русский язык."}],
     "param_validator": None},
    {"id": 18, "category": "negative_no_tool", "needs_tool": False, "expected_tool": None,
     "messages": [{"role": "user", "content": "Объясни простыми словами разницу между структурой стек (Stack) и очередь (Queue)."}],
     "param_validator": None},
    {"id": 19, "category": "negative_no_tool", "needs_tool": False, "expected_tool": None,
     "messages": [{"role": "user", "content": "Привет! Как поживаешь?"}],
     "param_validator": None},
    {"id": 20, "category": "negative_no_tool", "needs_tool": False, "expected_tool": None,
     "messages": [{"role": "user", "content": "В чем главное концептуальное отличие синхронного выполнения от асинхронного?"}],
     "param_validator": None}
]

def extract_tool_call(text):
    text = text.strip()
    candidate_texts = []
    match = re.search(r"```(?:json)?\s*(\{.*)", text, re.DOTALL)
    if match:
        candidate_texts.append(match.group(1))

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
            if isinstance(obj, dict) and ("tool" in obj or "name" in obj):
                return obj, True
        except Exception:
            pass

    tool_keywords = ['"tool"', "'tool'", "tool:", "get_current_weather", "web_search", "execute_python", "file_read", "query_database"]
    if any(k in text for k in tool_keywords):
        return None, True

    return None, False

class EnergySampler:
    def __init__(self, log_path="C:\\ai_models\\rapl_eval_log.csv"):
        self.log_path = log_path
        self.proc = None

    def start(self):
        if os.path.exists(self.log_path):
            os.remove(self.log_path)
        cmd = ["typeperf", r"\Energy Meter(RAPL_Package0_PKG)\Power", "-si", "1", "-o", self.log_path, "-y"]
        self.proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(1.5)

    def stop(self):
        if self.proc:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except Exception:
                self.proc.kill()
            self.proc = None
        time.sleep(0.5)

    def get_readings_w(self):
        if not os.path.exists(self.log_path):
            return []
        readings = []
        with open(self.log_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
        for line in lines[1:]:
            parts = line.strip().split(",")
            if len(parts) >= 2:
                raw_val = parts[1].replace('"', '').strip()
                try:
                    readings.append(float(raw_val.replace(',', '.')) / 1000.0)
                except Exception:
                    pass
        return readings

def format_prompt(messages, system_prompt, model_type="qwen"):
    if model_type == "qwen":
        prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        for m in messages:
            role = m["role"]
            content = m["content"]
            prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"
        prompt += "<|im_start|>assistant\n"
        stop_tokens = ["<|im_end|>", "<|endoftext|>"]
    else:
        prompt = f"System: {system_prompt}<|eot_id|>"
        for m in messages:
            role = "User" if m["role"] == "user" else "Assistant"
            prompt += f"{role}: {m['content']}<|eot_id|>"
        prompt += "Assistant: "
        stop_tokens = ["<|eot_id|>", "<|end_of_text|>", "User:"]
    return prompt, stop_tokens

def run_server(model_path, port=SERVER_PORT, threads=4, ctx=2048):
    # Kill any existing server first
    for p in psutil.process_iter(['pid', 'name']):
        if 'llama-server' in p.info['name'].lower():
            try:
                p.kill()
            except Exception:
                pass
    time.sleep(1.0)

    cmd = [
        LLAMA_SERVER_EXE,
        "-m", model_path,
        "--host", "127.0.0.1",
        "--port", str(port),
        "-t", str(threads),
        "-c", str(ctx)
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    ready = False
    for _ in range(40):
        try:
            r = requests.get(f"http://127.0.0.1:{port}/health", timeout=1)
            if r.status_code == 200:
                ready = True
                break
        except Exception:
            pass
        time.sleep(0.5)
    if not ready:
        raise RuntimeError(f"Failed to start llama-server with model {model_path}")
    return proc

def kill_server(proc):
    if proc:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()
    time.sleep(1.0)

def evaluate_model_on_server(model_name, model_type, system_prompt):
    print(f"\n=======================================================")
    print(f"BENCHMARK: {model_name} (type={model_type})")
    print(f"=======================================================")

    session = requests.Session()
    session.trust_env = False

    sampler = EnergySampler()
    sampler.start()
    print("Measuring baseline power for 3 seconds...")
    time.sleep(3.0)

    server_pid = None
    for p in psutil.process_iter(['pid', 'name']):
        if 'llama-server' in p.info['name'].lower():
            server_pid = p.info['pid']
            break
    server_proc = psutil.Process(server_pid) if server_pid else None
    mem_init_mb = server_proc.memory_info().rss / (1024 * 1024) if server_proc else 0.0

    pos_cases = 0
    trigger_attempts = 0
    correct_executions = 0
    ctx_pos_cases = 0
    ctx_trigger_attempts = 0
    ctx_correct_executions = 0
    neg_cases = 0
    neg_direct_success = 0

    results = []
    total_tokens_gen = 0
    t_start_all = time.perf_counter()

    for case in TEST_CASES:
        case_id = case["id"]
        category = case["category"]
        needs_tool = case["needs_tool"]
        expected_tool = case["expected_tool"]
        messages = case["messages"]
        param_validator = case["param_validator"]

        prompt_str, stop_tokens = format_prompt(messages, system_prompt, model_type=model_type)
        payload = {
            "prompt": prompt_str,
            "n_predict": 128,
            "temperature": 0.0,
            "repeat_penalty": 1.15,
            "stop": stop_tokens
        }

        t0 = time.perf_counter()
        resp = session.post(SERVER_URL, json=payload, timeout=60.0)
        t_elapsed = time.perf_counter() - t0
        data = resp.json()

        resp_text = data.get("content", "").strip()
        timings = data.get("timings", {})
        tokens_n = timings.get("predicted_n", 0)
        tps_gen = timings.get("predicted_per_second", 0.0)
        total_tokens_gen += tokens_n

        parsed_call, attempted = extract_tool_call(resp_text)

        trigger_success = False
        exec_success = False
        neg_success = False

        if needs_tool:
            pos_cases += 1
            if category == "context_retention":
                ctx_pos_cases += 1

            if attempted:
                trigger_attempts += 1
                trigger_success = True
                if category == "context_retention":
                    ctx_trigger_attempts += 1

                if parsed_call is not None:
                    called_tool = parsed_call.get("tool") or parsed_call.get("name")
                    params = parsed_call.get("parameters") or parsed_call.get("arguments") or {}
                    if called_tool == expected_tool:
                        if param_validator is None or param_validator(params):
                            correct_executions += 1
                            exec_success = True
                            if category == "context_retention":
                                ctx_correct_executions += 1
        else:
            neg_cases += 1
            if not attempted:
                neg_direct_success += 1
                neg_success = True

        status = "OK" if (exec_success if needs_tool else neg_success) else "FAIL"
        print(f"Case {case_id:2d} [{category:18s}]: {status} ({t_elapsed:.2f}s, {tokens_n} tok, {tps_gen:.1f} t/s)")

        results.append({
            "id": case_id,
            "category": category,
            "needs_tool": needs_tool,
            "expected_tool": expected_tool,
            "response": resp_text,
            "parsed_call": parsed_call,
            "trigger_success": trigger_success,
            "execution_success": exec_success,
            "negative_success": neg_success,
            "elapsed": t_elapsed,
            "tokens": tokens_n,
            "tps": tps_gen
        })

    t_total_all = time.perf_counter() - t_start_all
    mem_peak_mb = server_proc.memory_info().rss / (1024 * 1024) if server_proc else 0.0

    print("Cooling down for 2 seconds...")
    time.sleep(2.0)
    sampler.stop()
    readings = sampler.get_readings_w()

    idle_samples = readings[:3] if len(readings) >= 3 else readings
    p_idle = sum(idle_samples) / len(idle_samples) if idle_samples else 21.0
    p_mean = sum(readings) / len(readings) if readings else 35.0
    p_peak = max(readings) if readings else 55.0

    total_joules = sum(readings) * 1.0
    net_joules = max(0.0, total_joules - (p_idle * len(readings)))
    gross_mj_tok = (total_joules / total_tokens_gen * 1000.0) if total_tokens_gen > 0 else 0
    net_mj_tok = (net_joules / total_tokens_gen * 1000.0) if total_tokens_gen > 0 else 0

    tr_pct = (trigger_attempts / pos_cases * 100.0) if pos_cases > 0 else 0.0
    er_pct = (correct_executions / trigger_attempts * 100.0) if trigger_attempts > 0 else 0.0
    e2e_pct = (correct_executions / pos_cases * 100.0) if pos_cases > 0 else 0.0
    ctx_er_pct = (ctx_correct_executions / ctx_pos_cases * 100.0) if ctx_pos_cases > 0 else 0.0
    neg_acc = (neg_direct_success / neg_cases * 100.0) if neg_cases > 0 else 0.0

    energy_to_success = (total_joules / correct_executions) if correct_executions > 0 else 9999.0

    summary = {
        "model_name": model_name,
        "model_type": model_type,
        "memory": {
            "ram_init_mb": mem_init_mb,
            "ram_peak_mb": mem_peak_mb
        },
        "power_and_energy": {
            "idle_power_w": p_idle,
            "mean_power_w": p_mean,
            "peak_power_w": p_peak,
            "total_joules": total_joules,
            "net_joules": net_joules,
            "gross_mj_per_token": gross_mj_tok,
            "net_mj_per_token": net_mj_tok,
            "energy_to_success_joules": energy_to_success
        },
        "accuracy": {
            "trigger_recall_pct": tr_pct,
            "execution_recall_pct": er_pct,
            "end_to_end_accuracy_pct": e2e_pct,
            "context_retention_pct": ctx_er_pct,
            "negative_control_accuracy_pct": neg_acc
        },
        "performance": {
            "total_tokens": total_tokens_gen,
            "wall_clock_sec": t_total_all,
            "mean_tps": total_tokens_gen / t_total_all if t_total_all > 0 else 0
        },
        "detailed_results": results
    }

    print("\n" + "="*50)
    print(f"RESULTS FOR {model_name}:")
    print(f"RAM Peak:           {mem_peak_mb:.1f} MB ({mem_peak_mb/1024:.2f} GB)")
    print(f"Mean Generation:    {summary['performance']['mean_tps']:.1f} t/s")
    print(f"Net Energy/Token:   {net_mj_tok:.1f} mJ/tok (Gross: {gross_mj_tok:.1f} mJ/tok)")
    print(f"Trigger Recall:     {tr_pct:.1f}%")
    print(f"Execution Recall:   {er_pct:.1f}%")
    print(f"End-to-End Acc:     {e2e_pct:.1f}%")
    print(f"Context Retention:  {ctx_er_pct:.1f}%")
    print(f"Negative Control:   {neg_acc:.1f}%")
    print(f"Energy to Success:  {energy_to_success:.1f} J / successful task")
    print("="*50 + "\n")

    return summary

def main():
    experiments = [
        # 1. Qwen 1.5B Instruct Q4_K_M (Symmetric Dense Baseline)
        {
            "name": "qwen2.5_1.5b_q4_k_m",
            "type": "qwen",
            "path": r"C:\ai_models\qwen2.5-1.5b-instruct-q4_k_m.gguf",
            "prompt": SYSTEM_PROMPT_BASE
        },
        # 2. BitNet 2B 4T i2_s Base Prompt
        {
            "name": "bitnet_2b_i2_s_base",
            "type": "bitnet",
            "path": r"C:\ai_models\BitNet\models\BitNet-b1.58-2B-4T\ggml-model-i2_s.gguf",
            "prompt": SYSTEM_PROMPT_BASE
        },
        # 3. BitNet 2B 4T i2_s with Contrastive Few-Shot (Negative Control Test A)
        {
            "name": "bitnet_2b_i2_s_contrastive",
            "type": "bitnet",
            "path": r"C:\ai_models\BitNet\models\BitNet-b1.58-2B-4T\ggml-model-i2_s.gguf",
            "prompt": SYSTEM_PROMPT_CONTRASTIVE
        },
        # 4. BitNet 2B 4T i2_s with Bloated Prompt (Negative Control Test B - User Idea)
        {
            "name": "bitnet_2b_i2_s_bloated",
            "type": "bitnet",
            "path": r"C:\ai_models\BitNet\models\BitNet-b1.58-2B-4T\ggml-model-i2_s.gguf",
            "prompt": SYSTEM_PROMPT_BLOATED
        },
        # 5. BitNet 2B 4T i2_s with NO_TOOL Trigger (Negative Control Test C)
        {
            "name": "bitnet_2b_i2_s_notool",
            "type": "bitnet",
            "path": r"C:\ai_models\BitNet\models\BitNet-b1.58-2B-4T\ggml-model-i2_s.gguf",
            "prompt": SYSTEM_PROMPT_NO_TOOL
        },
        # 6. Qwen 2.5 3B Instruct Q4_K_M (Scaled Dense Competitor)
        {
            "name": "qwen2.5_3b_q4_k_m",
            "type": "qwen",
            "path": r"C:\ai_models\qwen2.5-3b-instruct-q4_k_m.gguf",
            "prompt": SYSTEM_PROMPT_BASE
        }
    ]

    all_results = {}
    for exp in experiments:
        name = exp["name"]
        m_type = exp["type"]
        path = exp["path"]
        prompt = exp["prompt"]

        if not os.path.exists(path):
            print(f"Skipping {name}: model file not found at {path}")
            continue

        print(f"\n>>> Starting server for {name} ({path})...")
        proc = run_server(path)
        try:
            res = evaluate_model_on_server(name, m_type, prompt)
            all_results[name] = res
        finally:
            kill_server(proc)

    out_file = r"C:\ai_models\paper_artifacts\symmetric_dense_vs_ternary.json"
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nAll symmetric results saved to {out_file}")

    desktop_copy = r"C:\Users\Рам\Desktop\BitNet_Research_Reports\symmetric_dense_vs_ternary.json"
    shutil.copyfile(out_file, desktop_copy)
    print(f"Copied to Desktop: {desktop_copy}")

if __name__ == "__main__":
    main()
