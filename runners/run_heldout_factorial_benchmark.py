import os
import sys
import json
import time
import socket
import requests
import subprocess
import psutil

sys.path.insert(0, r"C:\ai_models")
from evaluator_rigorous import (
    parse_tool_envelope,
    evaluate_static_heuristic_arguments,
    evaluate_execution_and_result_sandboxed
)
from transformers import AutoTokenizer

MODEL_DIR = r"C:\ai_models\BitNet-b1.58-2B-4T"
MODEL_GGUF = r"C:\ai_models\BitNet\models\BitNet-b1.58-2B-4T\ggml-model-i2_s.gguf"
SERVER_SILU = r"C:\ai_models\BitNet\build\bin\llama-server-silu.exe"
SERVER_RELU2 = r"C:\ai_models\BitNet\build\bin\llama-server-relu2.exe"
TEST_SUITE = r"C:\ai_models\heldout_test_suite_30.json"
OUT_MATRIX = r"C:\ai_models\paper_artifacts\heldout_30_cases_paired_transitions.json"
PORT = 8080

SYSTEM_PROMPT_BASE = """You are a helpful assistant with access to tools.
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

def run_cell(cell_id, cell_name, server_exe, input_mode, cases, tok):
    print("\n" + "=" * 70)
    print(f"RUNNING HELDOUT CELL {cell_id}: {cell_name} ({input_mode})")
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
        raise RuntimeError(f"Server {server_exe} (PID {proc.pid}) failed to become ready on port {PORT} within timeout")

    print("Server ready! Executing 30 held-out cases...")
    cell_results = []
    stop_tokens = ["<|eot_id|>", "<|end_of_text|>", "User:"]

    try:
        for case in cases:
            cid = case["id"]
            cat = case["category"]
            needs_tool = case["needs_tool"]
            exp_tool = case.get("expected_tool")

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
        stop_process(proc)

    # Evaluate results using evaluator_rigorous
    evaluated = []
    for r, case in zip(cell_results, cases):
        raw_text = r["response"]
        s1, s2, envelope, env_err = parse_tool_envelope(raw_text)

        s3 = False
        s3_detail = ""
        s4_status = "N/A"
        s5_status = "N/A"
        rule_str = "—"
        exec_detail = ""

        if case["needs_tool"]:
            if s2:
                s3, s3_detail = evaluate_static_heuristic_arguments(case, envelope)
            if s3 and envelope:
                s4_res, s5_res, exec_detail = evaluate_execution_and_result_sandboxed(case, envelope)
                s4_status = "[+]" if s4_res is True else ("[-]" if s4_res is False else str(s4_res))
                s5_status = "[+]" if s5_res is True else ("[-]" if s5_res is False else str(s5_res))
            elif not s2:
                exec_detail = env_err or "Невалидная схема"
            elif not s3:
                exec_detail = s3_detail
        else:
            if not s1:
                rule_str = "[OK] Нет вызова"
            else:
                rule_str = "[FAIL] Вызов инструмента"
            exec_detail = "Текст: " + repr(raw_text[:50])

        # Precise categorical state
        if not case["needs_tool"]:
            if s1:
                state = "FALSE_POSITIVE_TOOL_CALL"
            else:
                if "1." in raw_text or "Available tools" in raw_text or len(raw_text) < 5:
                    state = "DEGENERATED_ECHO_NO_TOOL"
                else:
                    state = "VALID_CONVERSATIONAL_REFUSAL"
        else:
            if not s1:
                state = "DEGENERATED_LOOP_NO_TOOL"
            elif not s2:
                state = "HALLUCINATED_OR_INVALID_SCHEMA"
            elif not s3:
                state = "VALID_SCHEMA_BAD_ARGS"
            else:
                if s4_status == "[+]" and s5_status == "[+]":
                    state = "FULL_EXECUTION_SUCCESS"
                else:
                    state = "VALID_TOOL_CALL"

        evaluated.append({
            "id": case["id"],
            "category": case["category"],
            "needs_tool": case["needs_tool"],
            "expected_tool": case.get("expected_tool"),
            "response": raw_text,
            "generated_tokens": r["generated_tokens"],
            "elapsed_s": r["elapsed_s"],
            "s1_tool_attempted": s1,
            "s2_schema_valid": s2,
            "s3_heuristic_valid": s3,
            "s4_executed": s4_status,
            "s5_result_correct": s5_status,
            "detail": exec_detail,
            "case_state": state
        })

    tool_cases = [x for x in evaluated if x["needs_tool"]]
    non_tool_cases = [x for x in evaluated if not x["needs_tool"]]

    s1_n = sum(1 for x in tool_cases if x.get("s1_tool_attempted"))
    s2_n = sum(1 for x in tool_cases if x.get("s2_schema_valid"))
    s3_n = sum(1 for x in tool_cases if x.get("s3_heuristic_valid"))
    s4_all = sum(1 for x in tool_cases if x.get("s4_executed") == "[+]")
    s5_all = sum(1 for x in tool_cases if x.get("s5_result_correct") == "[+]")

    local_ids = {10, 11, 12, 13, 14, 15, 17, 19}
    local_cases = [x for x in tool_cases if x["id"] in local_ids]
    local_reached = [x for x in local_cases if x.get("s4_executed") != "N/A"]
    local_s4_success = [x for x in local_cases if x.get("s4_executed") == "[+]"]
    local_s5_success = [x for x in local_cases if x.get("s5_result_correct") == "[+]"]

    valid_refusals = sum(1 for x in non_tool_cases if x.get("case_state") == "VALID_CONVERSATIONAL_REFUSAL")
    degenerated_refusals = sum(1 for x in non_tool_cases if x.get("case_state") == "DEGENERATED_ECHO_NO_TOOL")

    summary = {
        "s1_attempts": f"{s1_n}/{len(tool_cases)} ({s1_n/len(tool_cases)*100:.1f}%)" if tool_cases else "N/A",
        "s2_valid_schema": f"{s2_n}/{len(tool_cases)} ({s2_n/len(tool_cases)*100:.1f}%)" if tool_cases else "N/A",
        "s3_heuristic": f"{s3_n}/{len(tool_cases)} ({s3_n/len(tool_cases)*100:.1f}%)" if tool_cases else "N/A",
        "s4_s5_local_applicable_tasks": len(local_cases),
        "s4_s5_local_reached_executor": len(local_reached),
        "s4_s5_success_among_actual_runs": f"{len(local_s5_success)}/{len(local_reached)} ({len(local_s5_success)/len(local_reached)*100:.1f}%)" if local_reached else "0/0 (N/A)",
        "s4_s5_overall_local_applicable": f"{len(local_s5_success)}/{len(local_cases)} ({len(local_s5_success)/len(local_cases)*100:.1f}%)",
        "s4_s5_end_to_end_all_tool_queries": f"{s5_all}/{len(tool_cases)} ({s5_all/len(tool_cases)*100:.1f}%)",
        "external_api_and_python_unmocked": f"{len(tool_cases) - len(local_cases)} (N/A)",
        "valid_conversational_refusals": f"{valid_refusals}/{len(non_tool_cases)} ({valid_refusals/len(non_tool_cases)*100:.1f}%)",
        "degenerated_no_tool": f"{degenerated_refusals}/{len(non_tool_cases)} ({degenerated_refusals/len(non_tool_cases)*100:.1f}%)",
        "total_elapsed_s": sum(x.get("elapsed_s", 0) for x in evaluated),
        "s1_n": s1_n,
        "s2_n": s2_n,
        "s3_n": s3_n,
        "s4_local_n": len(local_s4_success),
        "s5_local_n": len(local_s5_success),
        "s4_all_n": s4_all,
        "s5_all_n": s5_all,
        "valid_refusals_n": valid_refusals,
        "degenerated_no_tool_n": degenerated_refusals
    }

    return {
        "cell_id": cell_id,
        "cell_name": cell_name,
        "server_exe": server_exe,
        "input_mode": input_mode,
        "summary": summary,
        "cases": evaluated
    }

def main():
    print("Loading HF tokenizer...")
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=False)

    with open(TEST_SUITE, "r", encoding="utf-8") as f:
        cases = json.load(f)

    print(f"Loaded {len(cases)} held-out cases.")

    cell_1 = run_cell("cell_1", "Old Graph (SwiGLU) + String Input", SERVER_SILU, "string", cases, tok)
    cell_2 = run_cell("cell_2", "Old Graph (SwiGLU) + Exact HF Token IDs", SERVER_SILU, "tokens", cases, tok)
    cell_3 = run_cell("cell_3", "New Graph (ReLU^2) + String Input", SERVER_RELU2, "string", cases, tok)
    cell_4 = run_cell("cell_4", "New Graph (ReLU^2) + Exact HF Token IDs", SERVER_RELU2, "tokens", cases, tok)

    paired_transitions = []
    for i in range(len(cases)):
        c = cases[i]
        c1 = cell_1["cases"][i]
        c2 = cell_2["cases"][i]
        c3 = cell_3["cases"][i]
        c4 = cell_4["cases"][i]

        paired_transitions.append({
            "case_id": c["id"],
            "category": c["category"],
            "needs_tool": c["needs_tool"],
            "expected_tool": c.get("expected_tool"),
            "prompt_user": c["messages"][-1]["content"],
            "states": {
                "cell_1_swiglu_string": c1["case_state"],
                "cell_2_swiglu_tokens": c2["case_state"],
                "cell_3_relu2_string": c3["case_state"],
                "cell_4_relu2_tokens": c4["case_state"]
            },
            "responses": {
                "cell_1": c1["response"][:70],
                "cell_2": c2["response"][:70],
                "cell_3": c3["response"][:70],
                "cell_4": c4["response"][:70]
            },
            "transition_graph_effect": f"{c1['case_state']} -> {c3['case_state']}",
            "transition_token_effect": f"{c3['case_state']} -> {c4['case_state']}"
        })

    output_data = {
        "title": "Held-Out 30-Case Benchmark with Paired Case-by-Case Transitions",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "note": "30-case runner used SYSTEM_PROMPT_BASE without 'AI' (You are a helpful assistant with access to tools.)",
        "metrics_provenance": {
            "calculation_method": "embedded_in_runner_and_verified_by_postprocessor",
            "runner_script": "C:\\ai_models\\run_heldout_factorial_benchmark.py",
            "reference_postprocessor_script": "C:\\ai_models\\update_heldout_json_metrics.py",
            "reference_postprocessor_sha256": "b144fb29276d5bcac14e217cd6ae728c337c4db7fefb344df9d39d1e9be8587b",
            "formula_rules": {
                "local_applicable_case_ids": [10, 11, 12, 13, 14, 15, 17, 19],
                "denominator_local_applicable": 8,
                "denominator_all_tool_queries": 20,
                "denominator_negative_dialog": 10,
                "external_unmocked_tools": ["get_current_weather", "web_search", "execute_python"]
            }
        },
        "cells": {
            "cell_1": cell_1,
            "cell_2": cell_2,
            "cell_3": cell_3,
            "cell_4": cell_4
        },
        "paired_transitions": paired_transitions
    }

    os.makedirs(os.path.dirname(OUT_MATRIX), exist_ok=True)
    with open(OUT_MATRIX, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 135)
    print("HELDOUT 30-CASE FACTORIAL BENCHMARK SUMMARY")
    print("=" * 135)
    print(f"{'Cell':<8} | {'Graph':<10} | {'Mode':<8} | {'S1 (JSON)':<15} | {'S2 (Schema)':<15} | {'S3 (Heur)':<15} | {'S4/S5 Reached':<15} | {'S4/S5 Local (8)':<18} | {'S4/S5 All (20)':<16} | {'Refusals (10)'}")
    print("-" * 135)
    for cid, cdata in output_data["cells"].items():
        s = cdata["summary"]
        print(f"{cid:<8} | {cdata['cell_name'].split('+')[0].strip():<10} | {cdata['input_mode']:<8} | {s['s1_attempts']:<15} | {s['s2_valid_schema']:<15} | {s['s3_heuristic']:<15} | {s['s4_s5_local_reached_executor']}/8 reached     | {s['s4_s5_overall_local_applicable']:<18} | {s['s4_s5_end_to_end_all_tool_queries']:<16} | {s['valid_conversational_refusals']}")
    print("=" * 135)
    print(f"Saved complete results to: {OUT_MATRIX}")

if __name__ == "__main__":
    main()
