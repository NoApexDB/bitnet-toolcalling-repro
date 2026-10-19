# -*- coding: utf-8 -*-
"""
POST-FIX symmetric benchmark: BitNet 2B (I2_S) on corrected ReLU^2 graph + exact HF token IDs
vs dense Qwen 2.5 1.5B / 3B (Q4_K_M), identical engine (llama-server-relu2.exe), 4 CPU threads.

Differences from the original benchmark_dense_vs_ternary.py (2026-10-14, pre-fix):
  * BitNet is fed EXACT HF token IDs (the "fixed" runtime path established by the 2x2 factorial);
    a string-input ablation row is included.
  * S1-S5 rigorous ladder (evaluator_rigorous.py) is reported alongside the legacy Dual Recall
    metrics so pre-fix and post-fix rows stay comparable.
  * A fixed-budget speed probe (3 reps x exactly 128 generated tokens, stop tokens disabled)
    separates decode/prefill throughput from output-length confounds.
  * PID-scoped server management (no global llama-server kills), port-ownership verification.
  * Results go to a NEW file: paper_artifacts/symmetric_dense_vs_ternary_postfix.json
    (the pre-fix artifact is never overwritten). Provenance hashes are embedded in the output.

Usage:
  python benchmark_postfix_symmetric.py list
  python benchmark_postfix_symmetric.py run <config_name>
  python benchmark_postfix_symmetric.py run_all          (skips configs already present)
  python benchmark_postfix_symmetric.py run_all --force
"""
import hashlib
import json
import os
import re
import socket
import statistics
import subprocess
import sys
import time

import psutil
import requests
from transformers import AutoTokenizer

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"C:\ai_models")

from evaluator_rigorous import (
    parse_tool_envelope,
    evaluate_static_heuristic_arguments,
    evaluate_execution_and_result_sandboxed,
)

SERVER_RELU2 = r"C:\ai_models\BitNet\build\bin\llama-server-relu2.exe"
MODEL_BITNET_GGUF = r"C:\ai_models\BitNet\models\BitNet-b1.58-2B-4T\ggml-model-i2_s.gguf"
MODEL_BITNET_HF = r"C:\ai_models\BitNet-b1.58-2B-4T"
MODEL_QWEN15 = r"C:\ai_models\qwen2.5-1.5b-instruct-q4_k_m.gguf"
MODEL_QWEN3 = r"C:\ai_models\qwen2.5-3b-instruct-q4_k_m.gguf"
TEST_SUITE_PATH = r"C:\ai_models\test_suite_standardized.json"
OUTPUT_JSON = r"C:\ai_models\paper_artifacts\symmetric_dense_vs_ternary_postfix.json"
OUTPUT_JSON_DESKTOP = r"C:\Users\Рам\Desktop\BitNet_Research_Reports\symmetric_dense_vs_ternary_postfix.json"
RAPL_LOG = r"C:\ai_models\rapl_postfix_log.csv"
PORT = 8096

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

CONFIGS = [
    {
        "name": "bitnet_2b_i2_s_relu2_hf_ids",
        "label": "BitNet 2B I2_S, corrected ReLU^2 graph, exact HF token IDs (FIXED runtime)",
        "model_type": "bitnet",
        "model_path": MODEL_BITNET_GGUF,
        "input_mode": "tokens",
        "prompt": SYSTEM_PROMPT_BASE,
    },
    {
        "name": "bitnet_2b_i2_s_relu2_string",
        "label": "BitNet 2B I2_S, corrected ReLU^2 graph, string input (tokenizer ablation)",
        "model_type": "bitnet",
        "model_path": MODEL_BITNET_GGUF,
        "input_mode": "string",
        "prompt": SYSTEM_PROMPT_BASE,
    },
    {
        "name": "qwen2.5_1.5b_q4_k_m",
        "label": "Qwen2.5-1.5B-Instruct Q4_K_M (dense baseline)",
        "model_type": "qwen",
        "model_path": MODEL_QWEN15,
        "input_mode": "string",
        "prompt": SYSTEM_PROMPT_BASE,
    },
    {
        "name": "qwen2.5_3b_q4_k_m",
        "label": "Qwen2.5-3B-Instruct Q4_K_M (dense scaled competitor)",
        "model_type": "qwen",
        "model_path": MODEL_QWEN3,
        "input_mode": "string",
        "prompt": SYSTEM_PROMPT_BASE,
    },
]


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
        for c in psutil.net_connections(kind="inet"):
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


def start_server(model_path):
    if is_port_in_use(PORT):
        alien = get_listening_pid(PORT)
        raise RuntimeError(f"Port {PORT} already occupied by alien PID {alien}; aborting.")

    cmd = [
        SERVER_RELU2,
        "-m", model_path,
        "--host", "127.0.0.1",
        "--port", str(PORT),
        "-t", "4",
        "-c", "2048",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    session = requests.Session()
    session.trust_env = False

    deadline = time.time() + 60.0
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"Server died during startup (rc={proc.returncode}).")
        listening = get_listening_pid(PORT)
        if listening is not None and listening != proc.pid:
            stop_process(proc)
            raise RuntimeError(f"Port {PORT} captured by alien PID {listening} != child {proc.pid}.")
        try:
            r = session.get(f"http://127.0.0.1:{PORT}/health", timeout=1)
            if r.status_code == 200:
                if proc.poll() is not None:
                    raise RuntimeError("Server died right after /health 200.")
                confirmed = get_listening_pid(PORT)
                if confirmed == proc.pid:
                    return proc, session
                elif confirmed is not None:
                    stop_process(proc)
                    raise RuntimeError(f"/health 200 but port owned by alien PID {confirmed}.")
        except (requests.RequestException, socket.error):
            pass
        time.sleep(0.5)

    stop_process(proc)
    raise RuntimeError("Server failed to become ready within timeout.")


class EnergySampler:
    """Identical protocol to the original benchmark_dense_vs_ternary.py (typeperf, 1 Hz)."""

    def __init__(self, log_path=RAPL_LOG):
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
                raw_val = parts[1].replace('"', "").strip()
                try:
                    readings.append(float(raw_val.replace(",", ".")) / 1000.0)
                except Exception:
                    pass
        return readings


def format_prompt(messages, system_prompt, model_type):
    if model_type == "qwen":
        prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        for m in messages:
            prompt += f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
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


def extract_tool_call(text):
    """Legacy Dual-Recall parser, copied verbatim from benchmark_dense_vs_ternary.py."""
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
    tool_keywords = ['"tool"', "'tool'", "tool:", "get_current_weather", "web_search",
                     "execute_python", "file_read", "query_database"]
    if any(k in text for k in tool_keywords):
        return None, True
    return None, False


def legacy_param_validator(case, parsed_call):
    """Reconstructs the original param_validator semantics from validation_rules."""
    if parsed_call is None:
        return False
    params = parsed_call.get("parameters") or parsed_call.get("arguments") or {}
    rules = case.get("validation_rules", {})
    if not rules:
        return True
    for key, expected_substrings in rules.items():
        value = str(params.get(key, "")).lower()
        if not any(es in value for es in expected_substrings):
            return False
    return True


def run_config(cfg, cases, tok):
    name = cfg["name"]
    print("\n" + "=" * 72)
    print(f"POST-FIX BENCHMARK: {name}")
    print(f"  {cfg['label']}")
    print("=" * 72)

    proc, session = start_server(cfg["model_path"])
    print("Server ready.")

    server_proc = psutil.Process(proc.pid)

    sampler = EnergySampler()
    sampler.start()
    print("Measuring baseline power for 3 seconds...")
    time.sleep(3.0)

    mem_init_mb = server_proc.memory_info().rss / (1024 * 1024)

    # ---------- 20-case evaluation ----------
    results = []
    dual = {"pos": 0, "trigger": 0, "exec": 0, "ctx_pos": 0, "ctx_exec": 0,
            "neg": 0, "neg_direct": 0}
    s1_cnt = s2_cnt = s3_cnt = s4_cnt = s5_cnt = non_tool_cnt = 0
    total_tokens_gen = 0
    prefill_tps_samples = []
    decode_tps_samples = []
    t_start_all = time.perf_counter()

    for case in cases:
        cid = case["id"]
        cat = case["category"]
        needs_tool = case["needs_tool"]
        prompt_str, stop_tokens = format_prompt(case["messages"], cfg["prompt"], cfg["model_type"])

        if cfg["input_mode"] == "tokens":
            exact_ids = tok(prompt_str).input_ids
            prompt_payload = exact_ids
        else:
            exact_ids = tok(prompt_str).input_ids  # reference length only
            prompt_payload = prompt_str

        payload = {
            "prompt": prompt_payload,
            "n_predict": 128,
            "temperature": 0.0,
            "repeat_penalty": 1.15,
            "repeat_last_n": 64,
            "stop": stop_tokens,
        }

        t0 = time.perf_counter()
        resp = session.post(f"http://127.0.0.1:{PORT}/completion", json=payload, timeout=120.0).json()
        t_elapsed = time.perf_counter() - t0

        content = resp.get("content", "").strip()
        timings = resp.get("timings", {})
        tokens_n = timings.get("predicted_n", 0)
        tps = timings.get("predicted_per_second", 0.0)
        total_tokens_gen += tokens_n
        if timings.get("prompt_per_second"):
            prefill_tps_samples.append(timings["prompt_per_second"])
        if tps:
            decode_tps_samples.append(tps)

        # ---- legacy Dual Recall ----
        parsed_call, attempted = extract_tool_call(content)
        trig_ok = exec_ok = neg_ok = False
        if needs_tool:
            dual["pos"] += 1
            if cat == "context_retention":
                dual["ctx_pos"] += 1
            if attempted:
                dual["trigger"] += 1
                trig_ok = True
                if cat == "context_retention":
                    pass
                if parsed_call is not None:
                    called = parsed_call.get("tool") or parsed_call.get("name")
                    if called == case["expected_tool"] and legacy_param_validator(case, parsed_call):
                        dual["exec"] += 1
                        exec_ok = True
                        if cat == "context_retention":
                            dual["ctx_exec"] += 1
        else:
            dual["neg"] += 1
            if not attempted:
                dual["neg_direct"] += 1
                neg_ok = True

        # ---- rigorous S1..S5 ----
        s1, s2, envelope, env_err = parse_tool_envelope(content)
        s3 = False
        s4_status = "N/A"
        s5_status = "N/A"
        detail = ""
        rule_str = "—"
        if needs_tool:
            if s1: s1_cnt += 1
            if s2: s2_cnt += 1
            if s2:
                s3, s3_detail = evaluate_static_heuristic_arguments(case, envelope)
            if s3:
                s3_cnt += 1
                s4_res, s5_res, detail = evaluate_execution_and_result_sandboxed(case, envelope)
                s4_status = "[+]" if s4_res is True else ("[-]" if s4_res is False else str(s4_res))
                s5_status = "[+]" if s5_res is True else ("[-]" if s5_res is False else str(s5_res))
                if s4_res is True: s4_cnt += 1
                if s5_res is True: s5_cnt += 1
            elif not s2:
                detail = env_err or "Невалидная схема"
            else:
                detail = s3_detail
        else:
            if not s1:
                non_tool_cnt += 1
                rule_str = "[OK] Нет вызова"
            else:
                rule_str = "[FAIL] Вызов инструмента"

        status = "OK" if (exec_ok if needs_tool else neg_ok) else "FAIL"
        print(f"Case {cid:2d} [{cat:18s}]: {status} ({t_elapsed:5.2f}s, {tokens_n:3d} tok, "
              f"{tps:5.1f} t/s) S1={int(s1)} S2={int(s2)} S3={int(s3)} | {repr(content[:40])}")

        results.append({
            "id": cid,
            "category": cat,
            "needs_tool": needs_tool,
            "expected_tool": case["expected_tool"],
            "prompt_tokens_len": len(exact_ids),
            "generated_tokens": tokens_n,
            "elapsed_s": t_elapsed,
            "tps": tps,
            "response": content,
            "legacy": {
                "parsed_call": parsed_call,
                "trigger_success": trig_ok,
                "execution_success": exec_ok,
                "negative_success": neg_ok,
            },
            "s1_tool_attempted": s1,
            "s2_schema_valid": s2,
            "s3_heuristic_valid": s3,
            "s4_executed": s4_status,
            "s5_result_correct": s5_status,
            "non_tool_rule": rule_str,
            "detail": detail,
        })

    t_total_all = time.perf_counter() - t_start_all
    mem_peak_mb = server_proc.memory_info().rss / (1024 * 1024)

    # ---------- fixed-budget speed probe ----------
    print("Speed probe: 3 reps x fixed 128 tokens (stop disabled)...")
    probe_prompt_str, _ = format_prompt(cases[0]["messages"], cfg["prompt"], cfg["model_type"])
    if cfg["input_mode"] == "tokens":
        probe_payload_prompt = tok(probe_prompt_str).input_ids
    else:
        probe_payload_prompt = probe_prompt_str
    probe = []
    for rep in range(3):
        pr = session.post(f"http://127.0.0.1:{PORT}/completion", json={
            "prompt": probe_payload_prompt,
            "n_predict": 128,
            "temperature": 0.0,
            "repeat_penalty": 1.15,
            "repeat_last_n": 64,
        }, timeout=180.0).json()
        tm = pr.get("timings", {})
        probe.append({
            "rep": rep + 1,
            "prompt_n": tm.get("prompt_n", 0),
            "prompt_ms": tm.get("prompt_ms", 0.0),
            "prompt_per_second": tm.get("prompt_per_second", 0.0),
            "predicted_n": tm.get("predicted_n", 0),
            "predicted_ms": tm.get("predicted_ms", 0.0),
            "predicted_per_second": tm.get("predicted_per_second", 0.0),
        })
        time.sleep(1.0)
    dec_tps = [p["predicted_per_second"] for p in probe if p["predicted_per_second"] > 0]
    pre_tps = [p["prompt_per_second"] for p in probe if p["prompt_per_second"] > 0]
    speed_probe_summary = {
        "reps": probe,
        "decode_tps_mean": statistics.mean(dec_tps) if dec_tps else 0.0,
        "decode_tps_stdev": statistics.stdev(dec_tps) if len(dec_tps) > 1 else 0.0,
        "prefill_tps_mean": statistics.mean(pre_tps) if pre_tps else 0.0,
        "prefill_tps_stdev": statistics.stdev(pre_tps) if len(pre_tps) > 1 else 0.0,
        "prompt_tokens_fixed": probe[0]["prompt_n"] if probe else 0,
        "generated_tokens_fixed": probe[0]["predicted_n"] if probe else 0,
    }

    # ---------- energy ----------
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

    tr_pct = dual["trigger"] / dual["pos"] * 100 if dual["pos"] else 0
    er_pct = dual["exec"] / dual["trigger"] * 100 if dual["trigger"] else 0
    e2e_pct = dual["exec"] / dual["pos"] * 100 if dual["pos"] else 0
    ctx_pct = dual["ctx_exec"] / dual["ctx_pos"] * 100 if dual["ctx_pos"] else 0
    neg_pct = dual["neg_direct"] / dual["neg"] * 100 if dual["neg"] else 0
    energy_to_success = (total_joules / dual["exec"]) if dual["exec"] > 0 else 9999.0

    summary = {
        "model_name": name,
        "label": cfg["label"],
        "model_type": cfg["model_type"],
        "server_exe": SERVER_RELU2,
        "input_mode": cfg["input_mode"],
        "memory": {"ram_init_mb": mem_init_mb, "ram_peak_mb": mem_peak_mb},
        "power_and_energy": {
            "idle_power_w": p_idle, "mean_power_w": p_mean, "peak_power_w": p_peak,
            "total_joules": total_joules, "net_joules": net_joules,
            "gross_mj_per_token": gross_mj_tok, "net_mj_per_token": net_mj_tok,
            "energy_to_success_joules": energy_to_success,
        },
        "accuracy_legacy_dual_recall": {
            "trigger_recall_pct": tr_pct, "execution_recall_pct": er_pct,
            "end_to_end_accuracy_pct": e2e_pct, "context_retention_pct": ctx_pct,
            "negative_control_accuracy_pct": neg_pct,
        },
        "accuracy_rigorous_s1_s5": {
            "s1_attempts": f"{s1_cnt}/15", "s2_valid_schema": f"{s2_cnt}/15",
            "s3_heuristic": f"{s3_cnt}/15",
            "s4_local_exec": f"{s4_cnt}/5", "s5_local_result": f"{s5_cnt}/5",
            "non_tool_refusals": f"{non_tool_cnt}/5",
            "s1_n": s1_cnt, "s2_n": s2_cnt, "s3_n": s3_cnt,
            "s4_n": s4_cnt, "s5_n": s5_cnt, "non_tool_n": non_tool_cnt,
        },
        "performance": {
            "total_tokens": total_tokens_gen,
            "wall_clock_sec": t_total_all,
            "mean_tps": total_tokens_gen / t_total_all if t_total_all > 0 else 0,
            "decode_tps_mean_eval": statistics.mean(decode_tps_samples) if decode_tps_samples else 0,
            "prefill_tps_mean_eval": statistics.mean(prefill_tps_samples) if prefill_tps_samples else 0,
        },
        "speed_probe_fixed_budget": speed_probe_summary,
        "detailed_results": results,
    }

    print("\n" + "=" * 60)
    print(f"RESULTS FOR {name}:")
    print(f"RAM Peak:        {mem_peak_mb:.1f} MB")
    print(f"Net Energy/Tok:  {net_mj_tok:.1f} mJ/tok (Gross: {gross_mj_tok:.1f})")
    print(f"DualRecall:      Trig {tr_pct:.1f}% | Exec {er_pct:.1f}% | E2E {e2e_pct:.1f}% | "
          f"Ctx {ctx_pct:.1f}% | Neg {neg_pct:.1f}%")
    print(f"S1-S5:           S1 {s1_cnt}/15 | S2 {s2_cnt}/15 | S3 {s3_cnt}/15 | "
          f"S4 {s4_cnt}/5 | S5 {s5_cnt}/5 | NonTool {non_tool_cnt}/5")
    print(f"Fixed-budget:    decode {speed_probe_summary['decode_tps_mean']:.1f}"
          f"±{speed_probe_summary['decode_tps_stdev']:.1f} t/s | "
          f"prefill {speed_probe_summary['prefill_tps_mean']:.0f} t/s")
    print(f"Energy/Success:  {energy_to_success:.1f} J")
    print("=" * 60)

    stop_process(proc)
    print("Server stopped.")
    return summary


def load_output():
    if os.path.exists(OUTPUT_JSON):
        with open(OUTPUT_JSON, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"title": "POST-FIX Symmetric Dense vs Ternary benchmark (corrected ReLU^2 graph + exact HF token IDs)",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "fixed_generation_params": {
                "threads": 4, "context_limit": 2048, "temperature": 0.0,
                "repeat_penalty": 1.15, "repeat_last_n": 64,
                "bitnet_stop_tokens": ["<|eot_id|>", "<|end_of_text|>", "User:"],
                "qwen_stop_tokens": ["<|im_end|>", "<|endoftext|>"],
                "energy_protocol": "typeperf \\Energy Meter(RAPL_Package0_PKG)\\Power -si 1 (identical to pre-fix run)",
            },
            "configs": {}}


def save_output(data):
    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    try:
        import shutil
        shutil.copyfile(OUTPUT_JSON, OUTPUT_JSON_DESKTOP)
    except Exception as e:
        print(f"[warn] desktop copy failed: {e}")


def main():
    args = [a for a in sys.argv[1:]]
    force = "--force" in args
    args = [a for a in args if not a.startswith("--")]

    print("Loading HF tokenizer and test suite...")
    tok = AutoTokenizer.from_pretrained(MODEL_BITNET_HF)
    with open(TEST_SUITE_PATH, "r", encoding="utf-8") as f:
        cases = json.load(f)
    print(f"Tokenizer ok; {len(cases)} cases loaded.")

    data = load_output()

    if args and args[0] == "list":
        for c in CONFIGS:
            state = "DONE" if c["name"] in data.get("configs", {}) else "pending"
            print(f"  [{state:>7}] {c['name']}: {c['label']}")
        return

    if args and args[0] == "run":
        target = args[1]
        todo = [c for c in CONFIGS if c["name"] == target]
        if not todo:
            print(f"Unknown config: {target}")
            return
    elif args and args[0] == "run_all":
        todo = [c for c in CONFIGS if force or c["name"] not in data.get("configs", {})]
    else:
        print(__doc__)
        return

    for cfg in todo:
        summary = run_config(cfg, cases, tok)
        data.setdefault("configs", {})[cfg["name"]] = summary
        # self-describing provenance
        data["provenance"] = {
            "script_sha256": compute_sha256(os.path.abspath(__file__)),
            "server_relu2_sha256": compute_sha256(SERVER_RELU2),
            "bitnet_gguf_sha256": compute_sha256(MODEL_BITNET_GGUF),
            "qwen15_gguf_sha256": compute_sha256(MODEL_QWEN15),
            "qwen3_gguf_sha256": compute_sha256(MODEL_QWEN3),
            "test_suite_sha256": compute_sha256(TEST_SUITE_PATH),
            "evaluator_sha256": compute_sha256(r"C:\ai_models\evaluator_rigorous.py"),
            "note": "New artifacts; NOT yet added to BENCHMARK_ARTIFACTS_MANIFEST.json",
        }
        save_output(data)
        print(f"\nSaved -> {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
