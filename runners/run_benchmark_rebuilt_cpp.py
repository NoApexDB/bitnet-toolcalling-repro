import json
import os
import re
import subprocess
import sys
import time
import psutil
import requests

sys.stdout.reconfigure(encoding="utf-8")

SERVER_EXE = r"C:\ai_models\BitNet\build\bin\llama-server.exe"
MODEL_PATH = r"C:\ai_models\BitNet\models\BitNet-b1.58-2B-4T\ggml-model-i2_s.gguf"
TEST_SUITE_PATH = r"C:\ai_models\test_suite_standardized.json"
OUTPUT_JSON = r"C:\ai_models\paper_artifacts\rebuilt_cpp_benchmark_20cases.json"
PORT = 8088
SERVER_URL = f"http://127.0.0.1:{PORT}/completion"

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

def format_prompt(messages, system_prompt):
    prompt = f"System: {system_prompt}<|eot_id|>"
    for m in messages:
        role = "User" if m["role"] == "user" else "Assistant"
        prompt += f"{role}: {m['content']}<|eot_id|>"
    prompt += "Assistant: "
    stop_tokens = ["<|eot_id|>", "<|end_of_text|>", "User:"]
    return prompt, stop_tokens

def kill_existing_server():
    for p in psutil.process_iter(['pid', 'name']):
        if 'llama-server' in p.info['name'].lower():
            try:
                p.kill()
            except Exception:
                pass
    time.sleep(1.0)

def start_server():
    kill_existing_server()
    cmd = [
        SERVER_EXE,
        "-m", MODEL_PATH,
        "--host", "127.0.0.1",
        "--port", str(PORT),
        "-t", "4",
        "-c", "2048"
    ]
    print(f"Starting server: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    session = requests.Session()
    session.trust_env = False

    ready = False
    for _ in range(40):
        try:
            r = session.get(f"http://127.0.0.1:{PORT}/health", timeout=1.0)
            if r.status_code == 200:
                ready = True
                break
        except Exception:
            pass
        time.sleep(0.5)

    if not ready:
        proc.kill()
        raise RuntimeError("Failed to start llama-server")
    print("Server is ready and healthy!")
    return proc

def main():
    print("=" * 65)
    print("RUNNING 20 BENCHMARK CASES ON REBUILT C++ BITNET (WITH RELU^2)")
    print("=" * 65)

    with open(TEST_SUITE_PATH, "r", encoding="utf-8") as f:
        test_cases = json.load(f)
    print(f"Loaded {len(test_cases)} test cases from {TEST_SUITE_PATH}")

    proc = start_server()
    session = requests.Session()
    session.trust_env = False

    results = []
    total_tokens = 0
    t_start_all = time.perf_counter()

    try:
        for idx, case in enumerate(test_cases):
            cid = case["id"]
            cat = case["category"]
            needs_tool = case["needs_tool"]
            exp_tool = case["expected_tool"]
            messages = case["messages"]

            prompt_str, stop_tokens = format_prompt(messages, SYSTEM_PROMPT_BASE)

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
            total_tokens += tokens_n

            print(f"Case {cid:02d} [{cat:18s}]: {tokens_n:3d} tok in {t_elapsed:5.2f}s ({tps_gen:5.1f} t/s) | Resp: {repr(resp_text[:60])}")

            results.append({
                "id": cid,
                "category": cat,
                "needs_tool": needs_tool,
                "expected_tool": exp_tool,
                "response": resp_text,
                "tokens_predicted": tokens_n,
                "tps": tps_gen,
                "elapsed_sec": t_elapsed
            })

    finally:
        print("\nStopping server...")
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()
        kill_existing_server()
        print("Server stopped.")

    total_time = time.perf_counter() - t_start_all
    avg_tps = total_tokens / total_time if total_time > 0 else 0.0

    print(f"\nCompleted {len(results)} cases in {total_time:.2f}s ({total_tokens} total tokens, {avg_tps:.1f} t/s)")

    out_data = {
        "model": "BitNet-b1.58-2B-4T (i2_s GGUF, rebuilt with LLM_FFN_RELU_SQR)",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "total_cases": len(results),
        "total_tokens": total_tokens,
        "total_time_sec": total_time,
        "avg_tps": avg_tps,
        "detailed_results": results
    }

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2, ensure_ascii=False)
    print(f"Saved results to: {OUTPUT_JSON}")

if __name__ == "__main__":
    main()
