import json
import os
import subprocess
import sys
import time
import requests

# Unbuffered print
import functools
print = functools.partial(print, flush=True)

LLAMA_SERVER = r"C:\ai_models\BitNet\build\bin\llama-server.exe"
MODEL_F32_GGUF = r"C:\ai_models\BitNet\models\BitNet-b1.58-2B-4T\ggml-model-f32.gguf"
TEST_SUITE_PATH = r"C:\ai_models\test_suite_standardized.json"
OUTPUT_PATH = r"C:\ai_models\benchmark_results_bitnet_f32_cpp.json"
PORT = 8088

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
    return prompt

def main():
    print("=" * 60)
    print("BITNET b1.58 GGUF F32 C++ BENCHMARK (DIAGNOSTIC RUN)")
    print("=" * 60)
    print(f"Model: {MODEL_F32_GGUF}")
    print(f"Server binary: {LLAMA_SERVER}")
    print(f"Threads: 4, Ctx: 2048, Temp: 0.0, Repeat Penalty: 1.15 (window 64)")
    print("=" * 60)

    # Kill any existing server first
    try:
        subprocess.run(["taskkill", "/F", "/IM", "llama-server.exe"], capture_output=True)
    except Exception:
        pass
    time.sleep(1.0)

    # Launch server
    cmd = [
        LLAMA_SERVER,
        "-m", MODEL_F32_GGUF,
        "--host", "127.0.0.1",
        "--port", str(PORT),
        "-t", "4",
        "-c", "2048"
    ]
    print("Starting llama-server with GGUF F32...")
    t0_srv = time.perf_counter()
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    try:
        ready = False
        for _ in range(60):
            try:
                r = requests.get(f"http://127.0.0.1:{PORT}/health", timeout=1)
                if r.status_code == 200:
                    ready = True
                    break
            except Exception:
                pass
            time.sleep(0.5)

        if not ready:
            print("ERROR: Failed to initialize llama-server with F32 GGUF!")
            return

        srv_time = time.perf_counter() - t0_srv
        print(f"Server ready in {srv_time:.2f} s")

        with open(TEST_SUITE_PATH, "r", encoding="utf-8") as f:
            test_cases = json.load(f)

        detailed_results = []
        total_tokens_all = 0
        t_start_all = time.perf_counter()

        for case in test_cases:
            cid = case["id"]
            category = case["category"]
            needs_tool = case["needs_tool"]
            exp_tool = case["expected_tool"]
            messages = case["messages"]
            user_query = messages[-1]["content"]

            prompt_str = format_prompt(messages, SYSTEM_PROMPT_BASE)

            payload = {
                "prompt": prompt_str,
                "n_predict": 128,
                "temperature": 0.0,
                "repeat_penalty": 1.15,
                "repeat_last_n": 64,
                "stop": ["<|eot_id|>", "<|end_of_text|>", "User:"]
            }

            t0_case = time.perf_counter()
            resp = requests.post(f"http://127.0.0.1:{PORT}/completion", json=payload, timeout=120.0)
            elapsed = time.perf_counter() - t0_case

            data = resp.json()
            resp_text = data.get("content", "").strip()
            timings = data.get("timings", {})
            tokens_n = timings.get("predicted_n", 0)
            tps_gen = timings.get("predicted_per_second", 0.0)
            total_tokens_all += tokens_n

            print(f"Case {cid:02d} [{category:18s}]: {tokens_n:3d} tokens in {elapsed:5.2f}s ({tps_gen:4.1f} t/s)")
            print(f"    Raw preview: {repr(resp_text[:70])}")

            detailed_results.append({
                "id": cid,
                "category": category,
                "needs_tool": needs_tool,
                "expected_tool": exp_tool,
                "user_query": user_query,
                "generated_tokens": tokens_n,
                "elapsed_sec": round(elapsed, 4),
                "tokens_per_sec": round(tps_gen, 2),
                "response": resp_text
            })

        t_total_all = time.perf_counter() - t_start_all
        mean_tps = total_tokens_all / t_total_all if t_total_all > 0 else 0.0

        print("=" * 60)
        print("GGUF F32 C++ BENCHMARK COMPLETED")
        print(f"Total time:       {t_total_all:.2f} s")
        print(f"Total tokens:     {total_tokens_all}")
        print(f"Mean rate:        {mean_tps:.2f} t/s")
        print("=" * 60)

        output_payload = {
            "model_name": "BitNet b1.58 Base (bitnet.cpp CPU F32)",
            "model_path": MODEL_F32_GGUF,
            "runtime": "bitnet.cpp / llama-server (CPU F32)",
            "environment": {
                "server_binary": LLAMA_SERVER,
                "cpu_threads": 4,
                "device": "cpu",
                "compute_dtype": "F32",
                "repetition_penalty": 1.15,
                "repeat_last_n": 64,
                "max_context_length": 2048,
                "max_new_tokens": 128,
                "temperature": 0.0,
                "stop_sequences": ["<|eot_id|>", "<|end_of_text|>", "User:"]
            },
            "performance": {
                "total_tokens": total_tokens_all,
                "wall_clock_sec": round(t_total_all, 2),
                "mean_tps": round(mean_tps, 2)
            },
            "detailed_results": detailed_results
        }

        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(output_payload, f, ensure_ascii=False, indent=2)

        print(f"Results saved to: {OUTPUT_PATH}")

    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()

if __name__ == "__main__":
    main()
