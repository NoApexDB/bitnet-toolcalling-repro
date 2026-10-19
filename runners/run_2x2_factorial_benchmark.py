import hashlib
import json
import os
import sys
import time
import socket
import subprocess
import requests
import psutil
from transformers import AutoTokenizer

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"C:\ai_models")

from evaluator_rigorous import (
    parse_tool_envelope,
    evaluate_static_heuristic_arguments,
    evaluate_execution_and_result_sandboxed
)

SERVER_SILU = r"C:\ai_models\BitNet\build\bin\llama-server-silu.exe"
SERVER_RELU2 = r"C:\ai_models\BitNet\build\bin\llama-server-relu2.exe"
MODEL_GGUF = r"C:\ai_models\BitNet\models\BitNet-b1.58-2B-4T\ggml-model-i2_s.gguf"
MODEL_HF = r"C:\ai_models\BitNet-b1.58-2B-4T"
TEST_SUITE_PATH = r"C:\ai_models\test_suite_standardized.json"
OUTPUT_MATRIX_JSON = r"C:\ai_models\paper_artifacts\factorial_2x2_benchmark_matrix.json"
MANIFEST_JSON = r"C:\ai_models\paper_artifacts\BENCHMARK_ARTIFACTS_MANIFEST.json"
PORT = 8095

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

def compute_sha256(path):
    if not os.path.exists(path):
        return "NOT_FOUND"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()

def stop_process(proc):
    if proc is not None and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()

def get_listening_pid(port):
    try:
        for c in psutil.net_connections(kind='inet'):
            if c.laddr and c.laddr.port == port and c.status == psutil.CONN_LISTEN:
                return c.pid
    except Exception:
        pass
    return None

def is_port_in_use(port, host="127.0.0.1"):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        if s.connect_ex((host, port)) == 0:
            return True
    return get_listening_pid(port) is not None

def format_prompt(messages, system_prompt):
    prompt = f"System: {system_prompt}<|eot_id|>"
    for m in messages:
        role = "User" if m["role"] == "user" else "Assistant"
        prompt += f"{role}: {m['content']}<|eot_id|>"
    prompt += "Assistant: "
    return prompt

def run_single_cell(cell_id, cell_name, server_exe, input_mode, cases, tok):
    print("\n" + "=" * 70)
    print(f"RUNNING CELL {cell_id}: {cell_name} (exe={os.path.basename(server_exe)}, mode={input_mode})")
    print("=" * 70)

    if is_port_in_use(PORT):
        alien_pid = get_listening_pid(PORT)
        raise RuntimeError(
            f"Port {PORT} is already occupied (PID {alien_pid}) prior to launching {server_exe}. "
            f"Aborting to avoid directing benchmark requests to an alien server."
        )

    cmd = [
        server_exe,
        "-m", MODEL_GGUF,
        "--host", "127.0.0.1",
        "--port", str(PORT),
        "-t", "4",
        "-c", "2048"
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    session = requests.Session()
    session.trust_env = False

    ready = False
    deadline = time.time() + 25.0
    while time.time() < deadline:
        exit_code = proc.poll()
        if exit_code is not None:
            raise RuntimeError(
                f"Server process {server_exe} (PID {proc.pid}) died prematurely with returncode {exit_code} during startup."
            )

        listening_pid = get_listening_pid(PORT)
        if listening_pid is not None and listening_pid != proc.pid:
            stop_process(proc)
            raise RuntimeError(
                f"Port {PORT} was captured by alien PID {listening_pid} (expected child server PID {proc.pid}). Aborting."
            )

        resp_status = None
        try:
            r = session.get(f"http://127.0.0.1:{PORT}/health", timeout=1)
            resp_status = r.status_code
        except (requests.RequestException, socket.error):
            pass

        if resp_status == 200:
            if proc.poll() is not None:
                raise RuntimeError(
                    f"Server process {server_exe} (PID {proc.pid}) died immediately after /health responded."
                )

            confirmed_pid = get_listening_pid(PORT)
            if confirmed_pid == proc.pid:
                ready = True
                break
            elif confirmed_pid is not None:
                stop_process(proc)
                raise RuntimeError(
                    f"/health responded 200, but port {PORT} belongs to alien PID {confirmed_pid} (expected {proc.pid})!"
                )
        time.sleep(0.5)

    if not ready:
        stop_process(proc)
        raise RuntimeError(f"Server {server_exe} (PID {proc.pid}) failed to start on port {PORT} within timeout")
    print("Server ready!")

    stop_tokens = ["<|eot_id|>", "<|end_of_text|>", "User:"]
    cell_results = []
    t_start = time.perf_counter()

    try:
        for idx, case in enumerate(cases):
            cid = case["id"]
            cat = case["category"]
            exp_tool = case["expected_tool"]
            needs_tool = case["needs_tool"]

            prompt_str = format_prompt(case["messages"], SYSTEM_PROMPT_BASE)
            exact_ids = tok(prompt_str).input_ids

            if input_mode == "tokens":
                prompt_payload = exact_ids
            else:
                prompt_payload = prompt_str

            payload = {
                "prompt": prompt_payload,
                "n_predict": 128,
                "temperature": 0.0,
                "repeat_penalty": 1.15,
                "repeat_last_n": 64,
                "stop": stop_tokens
            }

            t0 = time.perf_counter()
            resp = session.post(f"http://127.0.0.1:{PORT}/completion", json=payload, timeout=60).json()
            elapsed = time.perf_counter() - t0

            content = resp.get("content", "").strip()
            timings = resp.get("timings", {})
            tokens_n = timings.get("predicted_n", 0)
            tps = timings.get("predicted_per_second", 0.0)

            print(f"Case {cid:02d} [{cat:18s}]: {tokens_n:3d} tok in {elapsed:5.2f}s ({tps:5.1f} t/s) | {repr(content[:45])}")

            cell_results.append({
                "id": cid,
                "category": cat,
                "needs_tool": needs_tool,
                "expected_tool": exp_tool,
                "prompt_tokens_len": len(exact_ids),
                "exact_input_ids": exact_ids,
                "generated_tokens": tokens_n,
                "elapsed_s": elapsed,
                "tps": tps,
                "response": content
            })

    finally:
        print("Stopping server...")
        stop_process(proc)
        print("Server stopped.")

    total_time = time.perf_counter() - t_start

    # Evaluate strictly under 5-stage criteria
    s1_cnt = 0
    s2_cnt = 0
    s3_cnt = 0
    s4_local_cnt = 0
    s5_local_cnt = 0
    local_total = 5
    non_tool_clean = 0

    evaluated_cases = []

    for idx, case in enumerate(cases):
        cid = case["id"]
        exp_tool = case["expected_tool"]
        raw_text = cell_results[idx]["response"]

        s1, s2, envelope, env_err = parse_tool_envelope(raw_text)

        s3 = False
        s3_detail = ""
        s4_status = "N/A"
        s5_status = "N/A"
        rule_str = "—"
        exec_detail = ""

        if case["needs_tool"]:
            if s1: s1_cnt += 1
            if s2: s2_cnt += 1

            if s2:
                s3, s3_detail = evaluate_static_heuristic_arguments(case, envelope)
            if s3: s3_cnt += 1

            if s3 and envelope:
                s4_res, s5_res, exec_detail = evaluate_execution_and_result_sandboxed(case, envelope)
                s4_status = "[+]" if s4_res is True else ("[-]" if s4_res is False else str(s4_res))
                s5_status = "[+]" if s5_res is True else ("[-]" if s5_res is False else str(s5_res))
                if s4_res is True: s4_local_cnt += 1
                if s5_res is True: s5_local_cnt += 1
            elif not s2:
                exec_detail = env_err or "Невалидная схема"
            elif not s3:
                exec_detail = s3_detail
        else:
            if not s1:
                non_tool_clean += 1
                rule_str = "[OK] Нет вызова"
            else:
                rule_str = "[FAIL] Вызов инструмента"
            exec_detail = "Текст: " + repr(raw_text[:50])

        evaluated_cases.append({
            "id": cid,
            "category": case["category"],
            "needs_tool": case["needs_tool"],
            "expected_tool": exp_tool,
            "response": raw_text,
            "exact_input_ids": cell_results[idx]["exact_input_ids"],
            "s1_tool_attempted": s1,
            "s2_schema_valid": s2,
            "s3_heuristic_valid": s3,
            "s4_executed": s4_status,
            "s5_result_correct": s5_status,
            "non_tool_rule": rule_str,
            "detail": exec_detail
        })

    summary = {
        "s1_attempts": f"{s1_cnt}/15 ({s1_cnt/15*100:.1f}%)",
        "s2_valid_schema": f"{s2_cnt}/15 ({s2_cnt/15*100:.1f}%)",
        "s3_heuristic": f"{s3_cnt}/15 ({s3_cnt/15*100:.1f}%)",
        "s4_local_exec": f"{s4_local_cnt}/{local_total} ({s4_local_cnt/local_total*100:.1f}%)",
        "s5_local_result": f"{s5_local_cnt}/{local_total} ({s5_local_cnt/local_total*100:.1f}%)",
        "non_tool_refusals": f"{non_tool_clean}/5 ({non_tool_clean/5*100:.1f}%)",
        "total_elapsed_s": total_time,
        "s1_n": s1_cnt,
        "s2_n": s2_cnt,
        "s3_n": s3_cnt,
        "s4_n": s4_local_cnt,
        "s5_n": s5_local_cnt,
        "non_tool_n": non_tool_clean
    }

    return {
        "cell_id": cell_id,
        "cell_name": cell_name,
        "server_exe": server_exe,
        "input_mode": input_mode,
        "summary": summary,
        "cases": evaluated_cases
    }

def main():
    print("=" * 80)
    print("CONTROLLED 2x2 FACTORIAL BENCHMARK EXPERIMENT")
    print("Factor 1: Graph Implementation (Old SwiGLU vs New ReLU^2)")
    print("Factor 2: Tokenization Protocol (String Prompt vs Exact HF Token IDs)")
    print("=" * 80)

    # 1. Load tokenizer and test suite
    print("Loading HF tokenizer and test suite...")
    tok = AutoTokenizer.from_pretrained(MODEL_HF)
    with open(TEST_SUITE_PATH, "r", encoding="utf-8") as f:
        cases = json.load(f)

    # 2. Define the 4 cells
    cells_def = [
        ("cell_1", "Old Graph (SwiGLU) + String Input", SERVER_SILU, "string"),
        ("cell_2", "Old Graph (SwiGLU) + Exact Token IDs", SERVER_SILU, "tokens"),
        ("cell_3", "New Graph (ReLU^2) + String Input", SERVER_RELU2, "string"),
        ("cell_4", "New Graph (ReLU^2) + Exact Token IDs", SERVER_RELU2, "tokens")
    ]

    all_cells = {}
    for cid, cname, se_exe, imode in cells_def:
        res = run_single_cell(cid, cname, se_exe, imode, cases, tok)
        all_cells[cid] = res

    # 3. Build Manifest including run scripts and evaluators
    print("\n" + "=" * 80)
    print("COMPUTING COMPLETE SHA-256 MANIFEST OF EXPERIMENT ARTIFACTS")
    print("=" * 80)
    artifacts = {
        "model_safetensors": r"C:\ai_models\BitNet-b1.58-2B-4T\model.safetensors",
        "model_config": r"C:\ai_models\BitNet-b1.58-2B-4T\config.json",
        "gguf_model": MODEL_GGUF,
        "llama_server_silu_exe": SERVER_SILU,
        "llama_server_relu2_exe": SERVER_RELU2,
        "dump_cpp_exe": r"C:\ai_models\dump_cpp_activations.exe",
        "evaluator_rigorous_py": r"C:\ai_models\evaluator_rigorous.py",
        "run_2x2_factorial_benchmark_py": os.path.abspath(__file__),
        "test_suite_json": TEST_SUITE_PATH
    }
    manifest = {}
    for name, path in artifacts.items():
        sha = compute_sha256(path)
        size = os.path.getsize(path) if os.path.exists(path) else 0
        manifest[name] = {
            "path": path,
            "sha256": sha,
            "size_bytes": size
        }
        print(f"  {name:<28}: {sha} ({size} bytes)")

    with open(MANIFEST_JSON, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Updated manifest saved to: {MANIFEST_JSON}")

    # 4. Save entire 2x2 matrix data
    matrix_output = {
        "title": "2x2 Factorial Benchmark Matrix: FFN Graph (SwiGLU vs ReLU^2) x Tokenization (String vs Exact HF IDs)",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "fixed_generation_params": {
            "threads": 4,
            "context_limit": 2048,
            "temperature": 0.0,
            "repeat_penalty": 1.15,
            "repeat_last_n": 64,
            "stop_tokens": ["<|eot_id|>", "<|end_of_text|>", "User:"]
        },
        "manifest": manifest,
        "cells": all_cells
    }

    with open(OUTPUT_MATRIX_JSON, "w", encoding="utf-8") as f:
        json.dump(matrix_output, f, indent=2, ensure_ascii=False)
    print(f"Matrix experiment results saved to: {OUTPUT_MATRIX_JSON}")

    # 5. Print summary table
    print("\n" + "=" * 115)
    print("FINAL 2x2 FACTORIAL EXPERIMENT MATRIX")
    print("=" * 115)
    print(f"{'Cell':<8} | {'Graph':<10} | {'Input Protocol':<22} | {'S1 (JSON Attempt)':<18} | {'S2 (Valid Schema)':<18} | {'S3 (Heuristic)':<16} | {'S4/S5 (Local)':<14} | {'Non-Tool':<10}")
    print("-" * 115)
    for cid, cname, se_exe, imode in cells_def:
        c = all_cells[cid]["summary"]
        graph_str = "SwiGLU" if "SwiGLU" in cname else "ReLU^2"
        mode_str = "String (GGUF BPE)" if imode == "string" else "Exact HF Token IDs"
        s45_str = f"{c['s4_local_exec']}"
        print(f"{cid:<8} | {graph_str:<10} | {mode_str:<22} | {c['s1_attempts']:<18} | {c['s2_valid_schema']:<18} | {c['s3_heuristic']:<16} | {s45_str:<14} | {c['non_tool_refusals']:<10}")
    print("=" * 115)

if __name__ == "__main__":
    main()
