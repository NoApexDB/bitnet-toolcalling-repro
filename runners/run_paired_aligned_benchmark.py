import hashlib
import json
import os
import sys
import time
import uuid
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
TOKENIZER_JSON = r"C:\ai_models\BitNet-b1.58-2B-4T\tokenizer.json"
TOKENIZER_CONFIG = r"C:\ai_models\BitNet-b1.58-2B-4T\tokenizer_config.json"
TEST_SUITE_PATH = r"C:\ai_models\test_suite_explicit_schemas.json"
OUTPUT_JSON = r"C:\ai_models\paper_artifacts\paired_ab_benchmark_results.json"
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

def start_server_and_wait(server_exe, port, model_gguf):
    if is_port_in_use(port):
        alien_pid = get_listening_pid(port)
        raise RuntimeError(
            f"Port {port} is already occupied (PID {alien_pid}) prior to launching {server_exe}. "
            f"Aborting to avoid directing benchmark requests to an alien server."
        )

    cmd = [
        server_exe,
        "-m", model_gguf,
        "--host", "127.0.0.1",
        "--port", str(port),
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

        listening_pid = get_listening_pid(port)
        if listening_pid is not None and listening_pid != proc.pid:
            stop_process(proc)
            raise RuntimeError(
                f"Port {port} was captured by alien PID {listening_pid} (expected child server PID {proc.pid}). Aborting."
            )

        resp_status = None
        try:
            r = session.get(f"http://127.0.0.1:{port}/health", timeout=1)
            resp_status = r.status_code
        except (requests.RequestException, socket.error):
            pass

        if resp_status == 200:
            if proc.poll() is not None:
                raise RuntimeError(
                    f"Server process {server_exe} (PID {proc.pid}) died immediately after /health responded."
                )

            confirmed_pid = get_listening_pid(port)
            if confirmed_pid == proc.pid:
                ready = True
                break
            elif confirmed_pid is not None:
                stop_process(proc)
                raise RuntimeError(
                    f"/health responded 200, but port {port} belongs to alien PID {confirmed_pid} (expected {proc.pid})!"
                )
        time.sleep(0.5)

    if not ready:
        stop_process(proc)
        raise RuntimeError(f"Server {server_exe} (PID {proc.pid}) failed to become ready on port {port} within timeout")

    return proc, session

def run_configuration_inference(server_name, server_exe, cases, tok, port):
    print("\n" + "=" * 75)
    print(f"RUNNING CONFIGURATION: {server_name}")
    print(f"Binary: {server_exe}")
    print("=" * 75)

    proc, session = start_server_and_wait(server_exe, port, MODEL_GGUF)
    print(f"Server {server_name} ready on port {port} (PID {proc.pid})")

    stop_tokens = ["<|eot_id|>", "<|end_of_text|>", "User:"]
    raw_results = []
    t_start = time.perf_counter()

    try:
        for idx, case in enumerate(cases):
            cid = case["id"]
            cat = case["category"]
            exp_tool = case["expected_tool"]
            needs_tool = case["needs_tool"]

            prompt_str = format_prompt(case["messages"], SYSTEM_PROMPT_BASE)
            exact_ids = tok(prompt_str).input_ids
            ids_hash = hashlib.sha256(json.dumps(exact_ids).encode("utf-8")).hexdigest()

            payload = {
                "prompt": exact_ids,
                "n_predict": 128,
                "temperature": 0.0,
                "repeat_penalty": 1.15,
                "repeat_last_n": 64,
                "stop": stop_tokens
            }

            t0 = time.perf_counter()
            resp = session.post(f"http://127.0.0.1:{port}/completion", json=payload, timeout=60).json()
            elapsed = time.perf_counter() - t0

            content = resp.get("content", "").strip()
            timings = resp.get("timings", {})
            tokens_n = timings.get("predicted_n", 0)
            tps = timings.get("predicted_per_second", 0.0)

            print(f"[{server_name}] Case {cid:02d} [{cat:18s}]: {len(exact_ids):3d} in_tok -> {tokens_n:3d} out_tok in {elapsed:5.2f}s ({tps:5.1f} t/s) | {repr(content[:50])}")

            raw_results.append({
                "id": cid,
                "category": cat,
                "needs_tool": needs_tool,
                "expected_tool": exp_tool,
                "exact_prompt": prompt_str,
                "input_token_ids": exact_ids,
                "input_tokens_sha256": ids_hash,
                "prompt_tokens_len": len(exact_ids),
                "generated_tokens": tokens_n,
                "elapsed_s": elapsed,
                "tps": tps,
                "response": content
            })
    finally:
        print(f"Stopping server {server_name} (PID {proc.pid})...")
        stop_process(proc)
        print("Server stopped.")

    total_time = time.perf_counter() - t_start

    # Evaluate responses under strict 5-stage criteria
    print(f"\nEvaluating responses for {server_name}...")
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
        raw_rec = raw_results[idx]
        raw_text = raw_rec["response"]

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
            "exact_prompt": raw_rec["exact_prompt"],
            "input_token_ids": raw_rec["input_token_ids"],
            "input_tokens_sha256": raw_rec["input_tokens_sha256"],
            "prompt_tokens_len": raw_rec["prompt_tokens_len"],
            "generated_tokens": raw_rec["generated_tokens"],
            "elapsed_s": raw_rec["elapsed_s"],
            "tps": raw_rec["tps"],
            "response": raw_text,
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
        "s1_n": s1_cnt,
        "s2_n": s2_cnt,
        "s3_n": s3_cnt,
        "s4_n": s4_local_cnt,
        "s5_n": s5_local_cnt,
        "non_tool_n": non_tool_clean,
        "total_elapsed_s": total_time
    }

    return {
        "server_name": server_name,
        "server_exe": server_exe,
        "summary": summary,
        "cases": evaluated_cases
    }

def compute_paired_transitions(control_cases, rebuilt_cases):
    """
    Computes rigorous paired transitions between Control (SiLU) and Rebuilt (ReLU^2).
    """
    metrics = ["s1_tool_attempted", "s2_schema_valid", "s3_heuristic_valid"]
    transitions = {}

    for m in metrics:
        both_pass = []
        both_fail = []
        control_only = []
        rebuilt_only = []

        for c_case, r_case in zip(control_cases, rebuilt_cases):
            if not c_case["needs_tool"]:
                continue
            cid = c_case["id"]
            c_val = c_case[m]
            r_val = r_case[m]

            if c_val and r_val:
                both_pass.append(cid)
            elif not c_val and not r_val:
                both_fail.append(cid)
            elif c_val and not r_val:
                control_only.append(cid)
            elif not c_val and r_val:
                rebuilt_only.append(cid)

        transitions[m] = {
            "both_pass_count": len(both_pass),
            "both_pass_ids": both_pass,
            "both_fail_count": len(both_fail),
            "both_fail_ids": both_fail,
            "control_only_count": len(control_only),
            "control_only_ids": control_only,
            "rebuilt_only_count": len(rebuilt_only),
            "rebuilt_only_ids": rebuilt_only
        }

    # S4 and S5 local transitions (cases 9, 10, 11, 12, 14)
    for m in ["s4_executed", "s5_result_correct"]:
        both_pass = []
        both_fail = []
        control_only = []
        rebuilt_only = []

        for c_case, r_case in zip(control_cases, rebuilt_cases):
            cid = c_case["id"]
            if cid not in [9, 10, 11, 12, 14]:
                continue
            c_val = (c_case[m] == "[+]")
            r_val = (r_case[m] == "[+]")

            if c_val and r_val:
                both_pass.append(cid)
            elif not c_val and not r_val:
                both_fail.append(cid)
            elif c_val and not r_val:
                control_only.append(cid)
            elif not c_val and r_val:
                rebuilt_only.append(cid)

        transitions[m] = {
            "both_pass_count": len(both_pass),
            "both_pass_ids": both_pass,
            "both_fail_count": len(both_fail),
            "both_fail_ids": both_fail,
            "control_only_count": len(control_only),
            "control_only_ids": control_only,
            "rebuilt_only_count": len(rebuilt_only),
            "rebuilt_only_ids": rebuilt_only
        }

    # Negative control transitions (cases 16..20)
    both_clean = []
    both_triggered = []
    control_clean_only = []
    rebuilt_clean_only = []

    for c_case, r_case in zip(control_cases, rebuilt_cases):
        if c_case["needs_tool"]:
            continue
        cid = c_case["id"]
        c_clean = ("[OK]" in c_case["non_tool_rule"])
        r_clean = ("[OK]" in r_case["non_tool_rule"])

        if c_clean and r_clean:
            both_clean.append(cid)
        elif not c_clean and not r_clean:
            both_triggered.append(cid)
        elif c_clean and not r_clean:
            control_clean_only.append(cid)
        elif not c_clean and r_clean:
            rebuilt_clean_only.append(cid)

    transitions["negative_control_refusals"] = {
        "both_refused_count": len(both_clean),
        "both_refused_ids": both_clean,
        "both_triggered_count": len(both_triggered),
        "both_triggered_ids": both_triggered,
        "control_only_refused_count": len(control_clean_only),
        "control_only_refused_ids": control_clean_only,
        "rebuilt_only_refused_count": len(rebuilt_clean_only),
        "rebuilt_only_refused_ids": rebuilt_clean_only
    }

    return transitions

def commit_results_and_sync_manifest(
    final_payload: dict,
    output_json: str,
    manifest_json: str,
    manifest_key: str = "paired_ab_benchmark_results_json"
) -> dict:
    run_timestamp = time.strftime("%Y%m%d_%H%M%S")
    unique_suffix = f"{os.getpid()}_{uuid.uuid4().hex[:8]}"
    versioned_filename = f"paired_ab_benchmark_results_{run_timestamp}_{unique_suffix}.json"
    out_dir = os.path.dirname(os.path.abspath(output_json))
    versioned_output = os.path.join(out_dir, versioned_filename)

    if os.path.exists(versioned_output):
        raise FileExistsError(f"Target versioned file already exists: {versioned_output}")

    temp_versioned = versioned_output + ".tmp"
    temp_canonical = output_json + ".tmp"
    temp_manifest = manifest_json + ".tmp"

    tracked_temp_files = [temp_versioned, temp_canonical, temp_manifest]
    manifest_committed = False
    versioned_finalized = False

    try:
        # Step A: write payload to temporary versioned file
        with open(temp_versioned, "w", encoding="utf-8") as f:
            json.dump(final_payload, f, indent=2, ensure_ascii=False)

        # Step B: compute exact hash and size from temporary versioned file
        with open(temp_versioned, "rb") as bf:
            out_hash = hashlib.sha256(bf.read()).hexdigest()
        out_size = os.path.getsize(temp_versioned)

        # Step C: finalize authoritative versioned file (immutable new artifact)
        os.replace(temp_versioned, versioned_output)
        versioned_finalized = True
        if temp_versioned in tracked_temp_files:
            tracked_temp_files.remove(temp_versioned)

        # Step D: prepare temporary canonical copy file
        with open(temp_canonical, "w", encoding="utf-8") as f:
            json.dump(final_payload, f, indent=2, ensure_ascii=False)

        # Step E: load existing manifest and build target record
        with open(manifest_json, "r", encoding="utf-8") as mf:
            m_data = json.load(mf)

        record = {
            "path": os.path.abspath(versioned_output),
            "sha256": out_hash,
            "size_bytes": out_size,
            "canonical_copy_path": os.path.abspath(output_json),
            "canonical_copy_status": "pending",
            "committed_at": time.strftime("%Y-%m-%dT%H:%M:%S")
        }
        m_data[manifest_key] = record

        # Also register the explicit schema test suite and the paired benchmark runner
        m_data["test_suite_explicit_schemas_json"] = {
            "path": os.path.abspath(TEST_SUITE_PATH),
            "sha256": compute_sha256(TEST_SUITE_PATH),
            "size_bytes": os.path.getsize(TEST_SUITE_PATH)
        }
        runner_path = os.path.abspath(__file__)
        m_data["run_paired_aligned_benchmark_py"] = {
            "path": runner_path,
            "sha256": compute_sha256(runner_path),
            "size_bytes": os.path.getsize(runner_path)
        }

        # Step F: write updated manifest to temporary file
        with open(temp_manifest, "w", encoding="utf-8") as mf:
            json.dump(m_data, mf, indent=2, ensure_ascii=False)

        # Step G: SOLE ATOMIC COMMIT POINT (Single atomic filesystem operation)
        os.replace(temp_manifest, manifest_json)
        manifest_committed = True
        if temp_manifest in tracked_temp_files:
            tracked_temp_files.remove(temp_manifest)

        # Step H: update canonical copy and finalize status
        canonical_ok = False
        try:
            os.replace(temp_canonical, output_json)
            canonical_ok = True
            if temp_canonical in tracked_temp_files:
                tracked_temp_files.remove(temp_canonical)
        except Exception as canon_err:
            try:
                if temp_manifest not in tracked_temp_files:
                    tracked_temp_files.append(temp_manifest)
                m_data[manifest_key]["canonical_copy_status"] = "desynchronized"
                m_data[manifest_key]["canonical_copy_error"] = str(canon_err)
                with open(temp_manifest, "w", encoding="utf-8") as mf:
                    json.dump(m_data, mf, indent=2, ensure_ascii=False)
                os.replace(temp_manifest, manifest_json)
                if temp_manifest in tracked_temp_files:
                    tracked_temp_files.remove(temp_manifest)
            except Exception:
                pass
            raise RuntimeError(
                f"Manifest committed successfully to authoritative file {versioned_output}, "
                f"but updating canonical copy {output_json} failed: {canon_err}. "
                f"Canonical copy is marked desynchronized."
            ) from canon_err

        # Step I: Update status to synchronized in manifest
        final_status = "pending"
        status_error = None
        if canonical_ok:
            try:
                if temp_manifest not in tracked_temp_files:
                    tracked_temp_files.append(temp_manifest)
                m_data[manifest_key]["canonical_copy_status"] = "synchronized"
                with open(temp_manifest, "w", encoding="utf-8") as mf:
                    json.dump(m_data, mf, indent=2, ensure_ascii=False)
                os.replace(temp_manifest, manifest_json)
                final_status = "synchronized"
                if temp_manifest in tracked_temp_files:
                    tracked_temp_files.remove(temp_manifest)
            except Exception as status_err:
                status_error = str(status_err)
                final_status = "pending"
                print(f"  WARNING: Failed to update manifest status to synchronized: {status_error}. State remains 'pending'.")
        else:
            final_status = "desynchronized"

        result = {
            "versioned_path": versioned_output,
            "sha256": out_hash,
            "size_bytes": out_size,
            "canonical_copy_path": output_json,
            "canonical_copy_status": final_status
        }
        if status_error:
            result["status_update_error"] = status_error

        return result

    except Exception as e:
        if not manifest_committed and versioned_finalized and os.path.exists(versioned_output):
            try:
                os.remove(versioned_output)
            except Exception:
                pass
        if manifest_committed:
            try:
                with open(manifest_json, "r", encoding="utf-8") as mf:
                    curr_m = json.load(mf)
                if manifest_key in curr_m:
                    curr_m[manifest_key]["canonical_copy_status"] = "desynchronized"
                    curr_m[manifest_key]["canonical_copy_error"] = str(e)
                    with open(temp_manifest, "w", encoding="utf-8") as mf:
                        json.dump(curr_m, mf, indent=2, ensure_ascii=False)
                    os.replace(temp_manifest, manifest_json)
            except Exception:
                pass
        raise
    finally:
        for tmp_path in tracked_temp_files:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

def main():
    print("=" * 80)
    print("CONTROLLED PAIRED A/B BENCHMARK: CONTROL (SiLU) VS REBUILT (ReLU^2)")
    print("Identical HuggingFace Token IDs & Explicit SQL Schemas")
    print("=" * 80)

    # 1. Preflight check: verify manifest
    print("[1/5] Preflight check: verifying manifest JSON...")
    if not os.path.exists(MANIFEST_JSON):
        raise RuntimeError(f"FATAL: Required manifest file not found: {MANIFEST_JSON}")
    with open(MANIFEST_JSON, "r", encoding="utf-8") as f:
        manifest_snapshot = json.load(f)
    if not isinstance(manifest_snapshot, dict) or len(manifest_snapshot) == 0:
        raise ValueError("Manifest is empty or invalid format")
    print(f"Manifest JSON loaded successfully ({len(manifest_snapshot)} entries).")

    # 2. Tokenizer and Test Suite
    print("\n[2/5] Loading HF tokenizer and explicit-schema test suite...")
    tok = AutoTokenizer.from_pretrained(MODEL_HF)
    with open(TEST_SUITE_PATH, "r", encoding="utf-8") as f:
        cases = json.load(f)
    print(f"Loaded {len(cases)} test cases from {TEST_SUITE_PATH}.")

    # Compute provenance hashes
    print("\n[3/5] Computing provenance hashes of binaries, models, and tokenizer...")
    provenance = {
        "server_control_silu": {
            "path": SERVER_SILU,
            "sha256": compute_sha256(SERVER_SILU),
            "size_bytes": os.path.getsize(SERVER_SILU)
        },
        "server_rebuilt_relu2": {
            "path": SERVER_RELU2,
            "sha256": compute_sha256(SERVER_RELU2),
            "size_bytes": os.path.getsize(SERVER_RELU2)
        },
        "model_gguf": {
            "path": MODEL_GGUF,
            "sha256": compute_sha256(MODEL_GGUF),
            "size_bytes": os.path.getsize(MODEL_GGUF)
        },
        "tokenizer_json": {
            "path": TOKENIZER_JSON,
            "sha256": compute_sha256(TOKENIZER_JSON),
            "size_bytes": os.path.getsize(TOKENIZER_JSON)
        },
        "tokenizer_config": {
            "path": TOKENIZER_CONFIG,
            "sha256": compute_sha256(TOKENIZER_CONFIG),
            "size_bytes": os.path.getsize(TOKENIZER_CONFIG)
        },
        "test_suite_explicit_schemas": {
            "path": TEST_SUITE_PATH,
            "sha256": compute_sha256(TEST_SUITE_PATH),
            "size_bytes": os.path.getsize(TEST_SUITE_PATH)
        }
    }
    for k, v in provenance.items():
        print(f"  {k:<28}: {v['sha256']} ({v['size_bytes']} bytes)")

    # 3. Run Configuration A: Control (SiLU)
    print("\n[4/5] Executing paired inference...")
    res_control = run_configuration_inference("Control_SiLU", SERVER_SILU, cases, tok, PORT)

    # Allow port cleanup
    time.sleep(1.0)

    # 4. Run Configuration B: Rebuilt (ReLU^2)
    res_rebuilt = run_configuration_inference("Rebuilt_ReLU2", SERVER_RELU2, cases, tok, PORT)

    # 5. Compute Paired Transitions
    transitions = compute_paired_transitions(res_control["cases"], res_rebuilt["cases"])

    print("\n" + "=" * 90)
    print("PAIRED A/B BENCHMARK SUMMARY (EXPLICIT SQL SCHEMAS, IDENTICAL HF TOKEN IDS)")
    print("=" * 90)
    print(f"{'Metric':<25} | {'Control (SiLU)':<18} | {'Rebuilt (ReLU^2)':<18} | {'Paired Transition (Both/Ctrl/Rebuilt/Neither)'}")
    print("-" * 90)
    s1_t = f"{transitions['s1_tool_attempted']['both_pass_count']}/{transitions['s1_tool_attempted']['control_only_count']}/{transitions['s1_tool_attempted']['rebuilt_only_count']}/{transitions['s1_tool_attempted']['both_fail_count']}"
    s2_t = f"{transitions['s2_schema_valid']['both_pass_count']}/{transitions['s2_schema_valid']['control_only_count']}/{transitions['s2_schema_valid']['rebuilt_only_count']}/{transitions['s2_schema_valid']['both_fail_count']}"
    s3_t = f"{transitions['s3_heuristic_valid']['both_pass_count']}/{transitions['s3_heuristic_valid']['control_only_count']}/{transitions['s3_heuristic_valid']['rebuilt_only_count']}/{transitions['s3_heuristic_valid']['both_fail_count']}"
    s4_t = f"{transitions['s4_executed']['both_pass_count']}/{transitions['s4_executed']['control_only_count']}/{transitions['s4_executed']['rebuilt_only_count']}/{transitions['s4_executed']['both_fail_count']}"
    s5_t = f"{transitions['s5_result_correct']['both_pass_count']}/{transitions['s5_result_correct']['control_only_count']}/{transitions['s5_result_correct']['rebuilt_only_count']}/{transitions['s5_result_correct']['both_fail_count']}"
    neg_t = f"{transitions['negative_control_refusals']['both_refused_count']}/{transitions['negative_control_refusals']['control_only_refused_count']}/{transitions['negative_control_refusals']['rebuilt_only_refused_count']}/{transitions['negative_control_refusals']['both_triggered_count']}"

    print(f"{'S1 (JSON Attempt)':<25} | {res_control['summary']['s1_attempts']:<18} | {res_rebuilt['summary']['s1_attempts']:<18} | {s1_t}")
    print(f"{'S2 (Valid Schema)':<25} | {res_control['summary']['s2_valid_schema']:<18} | {res_rebuilt['summary']['s2_valid_schema']:<18} | {s2_t}")
    print(f"{'S3 (Heuristic Match)':<25} | {res_control['summary']['s3_heuristic']:<18} | {res_rebuilt['summary']['s3_heuristic']:<18} | {s3_t}")
    print(f"{'S4 (Local Execution)':<25} | {res_control['summary']['s4_local_exec']:<18} | {res_rebuilt['summary']['s4_local_exec']:<18} | {s4_t}")
    print(f"{'S5 (Correct Result)':<25} | {res_control['summary']['s5_local_result']:<18} | {res_rebuilt['summary']['s5_local_result']:<18} | {s5_t}")
    print(f"{'Negative Control (Refusal)':<25} | {res_control['summary']['non_tool_refusals']:<18} | {res_rebuilt['summary']['non_tool_refusals']:<18} | {neg_t}")
    print("=" * 90)

    final_payload = {
        "title": "Paired A/B Benchmark Experiment: Control (SiLU) vs Rebuilt (ReLU^2) on Identical HF Tokens & Explicit Schemas",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "provenance": provenance,
        "generation_parameters": {
            "threads": 4,
            "context_limit": 2048,
            "temperature": 0.0,
            "repeat_penalty": 1.15,
            "repeat_last_n": 64,
            "stop_tokens": ["<|eot_id|>", "<|end_of_text|>", "User:"]
        },
        "manifest_snapshot_at_start": manifest_snapshot,
        "control_silu": res_control,
        "rebuilt_relu2": res_rebuilt,
        "paired_transitions": transitions
    }

    # 6. Consistent commit
    print(f"\n[5/5] Performing consistent commit of paired benchmark results...")
    commit_res = commit_results_and_sync_manifest(
        final_payload=final_payload,
        output_json=OUTPUT_JSON,
        manifest_json=MANIFEST_JSON,
        manifest_key="paired_ab_benchmark_results_json"
    )
    print("Commit succeeded:")
    print(f"  Authoritative versioned file: {commit_res['versioned_path']}")
    print(f"  SHA-256: {commit_res['sha256']}, Size: {commit_res['size_bytes']} bytes")
    print(f"  Canonical copy: {commit_res['canonical_copy_path']} (status: {commit_res['canonical_copy_status']})")

if __name__ == "__main__":
    main()
