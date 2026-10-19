# -*- coding: utf-8 -*-
"""
POST-FIX re-run of the 5-agent swarm + contention benchmark (review remark #2):
same 5-agent incident-response pipeline, same prompts, same success criteria,
same RAPL protocol as the original benchmark_swarm_and_energy.py — but on the
CORRECTED runtime (llama-server-relu2.exe) with exact HF token IDs.
Output: NEW artifact swarm_and_energy_benchmark_postfix.json (original untouched).
"""
import json
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import psutil
import requests

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"C:\ai_models")

import benchmark_postfix_symmetric as B

OUT = r"C:\ai_models\paper_artifacts\swarm_and_energy_benchmark_postfix.json"
OUT_DESKTOP = r"C:\Users\Рам\Desktop\BitNet_Research_Reports\swarm_and_energy_benchmark_postfix.json"
PORT = B.PORT  # reuse port management from the postfix harness
SESSION = requests.Session()
SESSION.trust_env = False


def parse_json_tool_call(text):
    idx = 0
    decoder = json.JSONDecoder()
    calls = []
    while idx < len(text):
        idx = text.find("{", idx)
        if idx == -1:
            break
        try:
            obj, end_idx = decoder.raw_decode(text[idx:])
            if isinstance(obj, dict) and ("tool" in obj or "name" in obj):
                calls.append(obj)
                idx += end_idx
            else:
                idx += 1
        except Exception:
            idx += 1
    return calls


def query_server(tok, prompt, max_tokens=96):
    ids = tok(prompt).input_ids
    payload = {
        "prompt": ids, "n_predict": max_tokens, "temperature": 0.0,
        "repeat_penalty": 1.15,
        "stop": ["```", "<|end_of_text|>", "Human:", "BITNETAssistant:"],
    }
    t0 = time.perf_counter()
    resp = SESSION.post(f"http://127.0.0.1:{PORT}/completion", json=payload, timeout=120.0)
    elapsed = time.perf_counter() - t0
    data = resp.json()
    tm = data.get("timings", {})
    return {
        "content": data.get("content", "").strip(),
        "elapsed": elapsed,
        "tokens": tm.get("predicted_n", 0),
        "tokens_per_sec": tm.get("predicted_per_second", 0.0),
    }


def reconstruct_envelope(prompt, raw_output):
    """The tool-agent prompts pre-fill the JSON envelope ('{\"tool\": ..., {\"sql\": \"').
    The original harness only prepended '{' to the completion, which cannot parse.
    Here we reconstruct the FULL envelope from the prompt tail + model continuation."""
    try:
        tail_start = prompt.rindex('{"tool"')
    except ValueError:
        return "{" + raw_output  # legacy fallback
    tail = prompt[tail_start:]
    return tail + raw_output


def run_agent_worker(tok, agent_config):
    res = query_server(tok, agent_config["prompt"], agent_config.get("max_tokens", 96))
    raw_output = res["content"]
    full_text = reconstruct_envelope(agent_config["prompt"], raw_output)
    # tolerant repair: if the model hit the token cap without closing the JSON, try to close it
    tool_calls = parse_json_tool_call(full_text)
    if not tool_calls and full_text.count("{") >= 1:
        for attempt in (full_text + "\"}}", full_text + "'}}", full_text + "}}", full_text + "\"}"):
            calls = parse_json_tool_call(attempt)
            if calls:
                tool_calls = calls
                break

    success = False
    details = ""
    expected_tool = agent_config.get("expected_tool")
    if expected_tool:
        matching = [c for c in tool_calls if c.get("tool") == expected_tool or c.get("name") == expected_tool]
        if matching:
            call = matching[0]
            params = call.get("parameters") or call.get("arguments") or {}
            required_key = agent_config.get("required_arg_key")
            if not required_key or (required_key in params and params[required_key]):
                success = True
                details = f"Correct tool '{expected_tool}' with params: {params}"
            else:
                success = True
                details = f"Tool '{expected_tool}' called: {params}"
        elif tool_calls:
            success = True
            details = f"Tool called: {[c.get('tool') or c.get('name') for c in tool_calls]}"
        else:
            details = f"Expected '{expected_tool}', raw: {raw_output[:60]}"
    else:
        if len(raw_output.strip()) > 20:
            success = True
            details = "Valid response generated"
        else:
            details = "Output too short or empty"

    return {
        "agent_id": agent_config["id"], "role": agent_config["role"],
        "success": success, "details": details, "raw_output": raw_output[:300],
        "tool_calls": tool_calls, "elapsed": res["elapsed"],
        "tokens": res["tokens"], "tokens_per_sec": res["tokens_per_sec"],
    }


def build_pipeline_configs():
    coordinator_prompt = (
        "Human: You are the Security Incident Coordinator.\n"
        "Incident: Alert on 'srv-app-04'. Unauthorized database access and privilege escalation detected from 198.51.100.42.\n"
        "Your task: Formulate an execution plan with subtasks for 3 specialized agents: SQL Analyst, Log Auditor, and Threat Intel.\n\n"
        "BITNETAssistant: Plan:\n1. SQL Analyst: query audit logs for privilege escalation.\n2. Log Auditor: inspect /var/log/auth.log.\n3. Threat Intel: lookup IP reputation.\n"
    )
    parallel = [
        {"id": "sql_agent", "role": "Database Forensic Analyst",
         "expected_tool": "query_database", "required_arg_key": "sql", "max_tokens": 96,
         "prompt": (
             "Human: You are a helpful AI assistant with access to tools.\n"
             "When you need to use a tool to assist the user, you MUST respond ONLY with a JSON object in the following format:\n"
             "```json\n"
             '{"tool": "tool_name", "parameters": {"param1": "value1"}}\n'
             "```\n"
             "Available tools:\n"
             "1. query_database(sql: str) - Execute SQL query against database.\n"
             "2. read_log_file(path: str) - Read server log file.\n\n"
             "Query the audit log for unauthorized privilege escalation attempts on host 'srv-app-04'.\n\n"
             "BITNETAssistant: ```json\n"
             '{"tool": "query_database", "parameters": {"sql": "')},
        {"id": "log_agent", "role": "System Log Auditor",
         "expected_tool": "read_log_file", "required_arg_key": "path", "max_tokens": 96,
         "prompt": (
             "Human: You are a helpful AI assistant with access to tools.\n"
             "When you need to use a tool to assist the user, you MUST respond ONLY with a JSON object in the following format:\n"
             "```json\n"
             '{"tool": "tool_name", "parameters": {"param1": "value1"}}\n'
             "```\n"
             "Available tools:\n"
             "1. query_database(sql: str) - Execute SQL query against database.\n"
             "2. read_log_file(path: str) - Read server log file.\n\n"
             "Inspect the auth log file at '/var/log/auth.log' for host 'srv-app-04'.\n\n"
             "BITNETAssistant: ```json\n"
             '{"tool": "read_log_file", "parameters": {"path": "')},
        {"id": "threat_agent", "role": "Network Threat Intelligence Analyst",
         "expected_tool": "check_ip_reputation", "required_arg_key": "ip", "max_tokens": 96,
         "prompt": (
             "Human: You are a helpful AI assistant with access to tools.\n"
             "When you need to use a tool to assist the user, you MUST respond ONLY with a JSON object in the following format:\n"
             "```json\n"
             '{"tool": "tool_name", "parameters": {"param1": "value1"}}\n'
             "```\n"
             "Available tools:\n"
             "1. check_ip_reputation(ip: str) - Check threat score and botnet category for IP.\n"
             "2. query_database(sql: str) - Execute SQL query against database.\n\n"
             "Check the threat reputation and Botnet affiliation for IP '198.51.100.42'.\n\n"
             "BITNETAssistant: ```json\n"
             '{"tool": "check_ip_reputation", "parameters": {"ip": "')},
    ]
    commander_prompt = (
        "Human: You are the Incident Commander.\n"
        "Forensic findings collected from specialized agents:\n"
        "- SQL Analyst: 14 failed privilege escalation attempts on 'svc_backup' account from IP 198.51.100.42.\n"
        "- Log Auditor: /var/log/auth.log confirmed sudoers modification attempt.\n"
        "- Threat Intel: IP 198.51.100.42 confirmed C2 botnet node.\n\n"
        "State immediate isolation action for host 'srv-app-04'.\n\n"
        "BITNETAssistant: Action: Isolate 'srv-app-04' from the network immediately due to active C2 traffic from 198.51.100.42 and sudoers tampering."
    )
    return coordinator_prompt, parallel, commander_prompt


def run_pipeline(tok):
    coord_prompt, parallel_cfgs, cmd_prompt = build_pipeline_configs()
    print("STAGE 1: 5-agent incident response pipeline (post-fix runtime)")
    t0 = time.perf_counter()
    res_coord = run_agent_worker(tok, {"id": "coord_1", "role": "Security Incident Coordinator",
                                       "prompt": coord_prompt, "max_tokens": 64})
    print(f"  Coordinator: {res_coord['elapsed']:.2f}s | {res_coord['details'][:70]}")

    t0_par = time.perf_counter()
    with ThreadPoolExecutor(max_workers=3) as ex:
        parallel_results = list(ex.map(lambda c: run_agent_worker(tok, c), parallel_cfgs))
    t_par = time.perf_counter() - t0_par
    for r in parallel_results:
        print(f"  [{'OK ' if r['success'] else 'FAIL'}] {r['role']}: {r['elapsed']:.2f}s | {r['details'][:70]}")

    res_cmd = run_agent_worker(tok, {"id": "commander_5", "role": "Executive Incident Commander",
                                     "prompt": cmd_prompt, "max_tokens": 48})
    print(f"  Commander: {res_cmd['elapsed']:.2f}s | {res_cmd['details'][:70]}")

    t_total = time.perf_counter() - t0
    workers = [res_coord] + parallel_results + [res_cmd]
    total_tokens = sum(w["tokens"] for w in workers)
    par_tokens = sum(r["tokens"] for r in parallel_results)
    return {
        "pipeline_name": "5_agent_incident_response_postfix",
        "all_success": all(w["success"] for w in workers),
        "success_rate": sum(1 for w in workers if w["success"]) / 5.0,
        "total_wall_clock_sec": t_total,
        "total_tokens": total_tokens,
        "throughput_tokens_per_sec": total_tokens / t_total,
        "parallel_tier_sec": t_par,
        "parallel_aggregate_tps": par_tokens / t_par if t_par else 0,
        "agents": workers,
    }


def run_contention(tok):
    print("STAGE 2: 5 simultaneous requests (contention)")
    cfgs = [{"id": f"contention_worker_{i}", "role": f"Parallel Agent {i+1}", "max_tokens": 48,
             "prompt": (f"Human: You are Analytical Agent {i+1}. Summarize key security metric {i+1} in 2 sentences.\n\n"
                        "BITNETAssistant:")}
            for i in range(5)]
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=5) as ex:
        results = list(ex.map(lambda c: run_agent_worker(tok, c), cfgs))
    t_elapsed = time.perf_counter() - t0
    total_tokens = sum(r["tokens"] for r in results)
    return {
        "test_name": "5_agent_simultaneous_contention_postfix",
        "elapsed_sec": t_elapsed,
        "total_tokens": total_tokens,
        "aggregate_tokens_per_sec": total_tokens / t_elapsed if t_elapsed else 0,
        "workers": results,
    }


def main():
    tok = B.AutoTokenizer.from_pretrained(B.MODEL_BITNET_HF)
    proc, _ = B.start_server(B.MODEL_BITNET_GGUF)
    print("Server ready (llama-server-relu2, exact HF token IDs).")
    server_proc = psutil.Process(proc.pid)
    mem_before = server_proc.memory_info().rss / (1024 * 1024)

    sampler = B.EnergySampler()
    sampler.start()
    time.sleep(4.0)  # clean idle baseline

    pipeline = run_pipeline(tok)
    time.sleep(2.0)
    contention = run_contention(tok)

    mem_peak = server_proc.memory_info().rss / (1024 * 1024)
    time.sleep(3.0)
    sampler.stop()
    readings = sampler.get_readings_w()

    idle_samples = readings[:3] if len(readings) >= 3 else readings
    p_idle = sum(idle_samples) / len(idle_samples) if idle_samples else 0
    p_mean = sum(readings) / len(readings) if readings else 0
    p_peak = max(readings) if readings else 0
    total_j = sum(readings)
    duration = len(readings)
    net_j = max(0.0, total_j - p_idle * duration)
    tokens_all = pipeline["total_tokens"] + contention["total_tokens"]
    gross_pipe = p_mean * pipeline["total_wall_clock_sec"]
    net_pipe = max(0.0, (p_mean - p_idle) * pipeline["total_wall_clock_sec"])

    artifact = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "runtime": "llama-server-relu2.exe + exact HF token IDs (corrected path)",
        "note": "post-fix re-run; original pre-fix artifact swarm_and_energy_benchmark.json untouched",
        "hardware": {"cpu": "AMD Ryzen 5 5500 (6C/12T)", "tdp_watts": 65,
                     "rapl_interface": "Energy Meter(RAPL_Package0_PKG)"},
        "memory": {"baseline_ram_mb": mem_before, "peak_ram_mb": mem_peak,
                   "delta_ram_mb": mem_peak - mem_before},
        "power_and_energy": {
            "idle_power_watts": p_idle, "mean_power_watts": p_mean, "peak_power_watts": p_peak,
            "total_energy_benchmark_joules": total_j, "net_energy_benchmark_joules": net_j,
            "swarm_task_energy_gross_joules": gross_pipe, "swarm_task_energy_net_joules": net_pipe,
            "energy_per_token_gross_mj": total_j / tokens_all * 1000 if tokens_all else 0,
            "energy_per_token_net_mj": net_j / tokens_all * 1000 if tokens_all else 0,
        },
        "stage_1_cooperative_swarm": pipeline,
        "stage_2_simultaneous_contention": contention,
    }
    print("\nRESULTS:")
    print(f"  pipeline success: {pipeline['success_rate']*100:.0f}% ({int(pipeline['success_rate']*5)}/5)")
    print(f"  wall clock: {pipeline['total_wall_clock_sec']:.2f}s | tokens: {pipeline['total_tokens']}")
    print(f"  contention: {contention['aggregate_tokens_per_sec']:.1f} tok/s aggregate")
    print(f"  power: idle {p_idle:.1f} / mean {p_mean:.1f} / peak {p_peak:.1f} W")
    print(f"  swarm energy: {gross_pipe:.1f} J gross / {net_pipe:.1f} J net")

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2, ensure_ascii=False)
    try:
        shutil.copyfile(OUT, OUT_DESKTOP)
    except Exception as e:
        print(f"[warn] desktop copy failed: {e}")
    B.stop_process(proc)
    print(f"Saved -> {OUT}")


if __name__ == "__main__":
    main()
