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

LLAMA_SERVER = r"C:\ai_models\BitNet\build\bin\llama-server-relu2.exe"
MODEL_GGUF = r"C:\ai_models\BitNet\models\BitNet-b1.58-2B-4T\ggml-model-i2_s.gguf"
MODEL_HF = r"C:\ai_models\BitNet-b1.58-2B-4T"
TEST_SUITE_PATH = r"C:\ai_models\test_suite_standardized.json"
OUTPUT_JSON = r"C:\ai_models\paper_artifacts\aligned_tokens_benchmark_results.json"
MANIFEST_JSON = r"C:\ai_models\paper_artifacts\BENCHMARK_ARTIFACTS_MANIFEST.json"
PORT = 8094

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

def commit_results_and_sync_manifest(
    final_payload: dict,
    output_json: str,
    manifest_json: str,
    simulate_failure: str = None
) -> dict:
    """
    Consistently commits benchmark results with the external manifest as the single commit point.

    IMPORTANT NOTE ON ATOMICITY:
    Only the single os.replace of the manifest file is an atomic operation at the OS/filesystem level.
    The multi-file procedure as a whole is NOT atomic. Consistency is achieved by:
    1. Making a unique, immutable versioned file the authoritative target.
    2. Storing the authoritative path, sha256, and size_bytes in the manifest record.
    3. Committing the transaction strictly at the moment os.replace(temp_manifest, manifest_json) succeeds.
    4. Separately updating the canonical copy and recording its synchronization status ('synchronized' vs 'desynchronized').
    """
    run_timestamp = time.strftime("%Y%m%d_%H%M%S")
    unique_suffix = f"{os.getpid()}_{uuid.uuid4().hex[:8]}"
    versioned_filename = f"aligned_tokens_benchmark_results_{run_timestamp}_{unique_suffix}.json"
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

        # Authoritative target is strictly the versioned file: path, sha256, and size_bytes match it!
        record = {
            "path": os.path.abspath(versioned_output),
            "sha256": out_hash,
            "size_bytes": out_size,
            "canonical_copy_path": os.path.abspath(output_json),
            "canonical_copy_status": "pending",
            "committed_at": time.strftime("%Y-%m-%dT%H:%M:%S")
        }
        m_data["aligned_tokens_benchmark_results_json"] = record

        # Step F: write updated manifest to temporary file
        with open(temp_manifest, "w", encoding="utf-8") as mf:
            json.dump(m_data, mf, indent=2, ensure_ascii=False)

        # Optional fault injection before commit
        if simulate_failure == "before_manifest_replace":
            raise RuntimeError("FAULT_INJECTION: Simulated crash BEFORE manifest replace")

        # Step G: SOLE ATOMIC COMMIT POINT (Single atomic filesystem operation)
        os.replace(temp_manifest, manifest_json)
        manifest_committed = True
        if temp_manifest in tracked_temp_files:
            tracked_temp_files.remove(temp_manifest)

        # Step H: update canonical copy and finalize status
        if simulate_failure == "after_manifest_replace":
            raise RuntimeError("FAULT_INJECTION: Simulated crash AFTER manifest replace during canonical copy")

        canonical_ok = False
        try:
            os.replace(temp_canonical, output_json)
            canonical_ok = True
            if temp_canonical in tracked_temp_files:
                tracked_temp_files.remove(temp_canonical)
        except Exception as canon_err:
            # Manifest is already committed. Record desynchronization status in manifest.
            try:
                if temp_manifest not in tracked_temp_files:
                    tracked_temp_files.append(temp_manifest)
                m_data["aligned_tokens_benchmark_results_json"]["canonical_copy_status"] = "desynchronized"
                m_data["aligned_tokens_benchmark_results_json"]["canonical_copy_error"] = str(canon_err)
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
                m_data["aligned_tokens_benchmark_results_json"]["canonical_copy_status"] = "synchronized"
                with open(temp_manifest, "w", encoding="utf-8") as mf:
                    json.dump(m_data, mf, indent=2, ensure_ascii=False)

                if simulate_failure == "manifest_status_update":
                    raise RuntimeError("FAULT_INJECTION: Simulated error during manifest status update")

                os.replace(temp_manifest, manifest_json)
                final_status = "synchronized"
                if temp_manifest in tracked_temp_files:
                    tracked_temp_files.remove(temp_manifest)
            except Exception as status_err:
                status_error = str(status_err)
                final_status = "pending"
                # Note: temp_manifest stays in tracked_temp_files and is removed in finally block
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
        # If manifest was NOT committed, the uncommitted versioned file must be cleaned up
        if not manifest_committed and versioned_finalized and os.path.exists(versioned_output):
            try:
                os.remove(versioned_output)
            except Exception:
                pass

        # If manifest WAS committed and error happened during canonical copy, ensure status is marked desynchronized
        if manifest_committed:
            try:
                with open(manifest_json, "r", encoding="utf-8") as mf:
                    curr_m = json.load(mf)
                if "aligned_tokens_benchmark_results_json" in curr_m:
                    curr_m["aligned_tokens_benchmark_results_json"]["canonical_copy_status"] = "desynchronized"
                    curr_m["aligned_tokens_benchmark_results_json"]["canonical_copy_error"] = str(e)
                    with open(temp_manifest, "w", encoding="utf-8") as mf:
                        json.dump(curr_m, mf, indent=2, ensure_ascii=False)
                    os.replace(temp_manifest, manifest_json)
            except Exception:
                pass

        raise
    finally:
        # Clean up any leftover temporary files across all code paths
        for tmp_path in tracked_temp_files:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

def main():
    print("=" * 75)
    print("REBUILT C++ BITNET BENCHMARK ON IDENTICAL HF TOKEN IDS INPUT")
    print("=" * 75)

    # 1. Preflight check: verify presence and read non-empty manifest JSON before inference
    print("[1/5] Preflight check: loading manifest JSON (presence and non-empty structure check)...")
    if not os.path.exists(MANIFEST_JSON):
        raise RuntimeError(f"FATAL: Required manifest file not found: {MANIFEST_JSON}")

    try:
        with open(MANIFEST_JSON, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        if not isinstance(manifest, dict) or len(manifest) == 0:
            raise ValueError("Manifest is empty or invalid format")
    except Exception as e:
        raise RuntimeError(f"FATAL: Failed to read or parse manifest {MANIFEST_JSON}: {e}") from e

    print(f"Manifest JSON loaded successfully ({len(manifest)} entries).")
    print("  Note: preflight only validates JSON existence and non-empty dictionary structure;")
    print("  it does NOT compute or verify SHA-256 / sizes of the listed files before inference.")

    # 2. Tokenizer
    print("\n[2/5] Loading HF tokenizer for aligned tokenization...")
    tok = AutoTokenizer.from_pretrained(MODEL_HF)

    # 3. Load test cases
    with open(TEST_SUITE_PATH, "r", encoding="utf-8") as f:
        cases = json.load(f)

    if is_port_in_use(PORT):
        alien_pid = get_listening_pid(PORT)
        raise RuntimeError(
            f"Port {PORT} is already occupied (PID {alien_pid}) prior to launching {LLAMA_SERVER}. "
            f"Aborting to avoid directing benchmark requests to an alien server."
        )

    print(f"\n[3/5] Starting llama-server on port {PORT} with -c 2048 -t 4...")
    cmd = [
        LLAMA_SERVER,
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
                f"Server process {LLAMA_SERVER} (PID {proc.pid}) died prematurely with returncode {exit_code} during startup."
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
                    f"Server process {LLAMA_SERVER} (PID {proc.pid}) died immediately after /health responded."
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
        raise RuntimeError(f"Server {LLAMA_SERVER} (PID {proc.pid}) failed to become ready on port {PORT} within timeout")
    print("Server ready!")

    # 5. Run inference with exact token IDs
    print(f"\n[4/5] Executing inference for 20 cases with identical HF token arrays...")
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
            # Exact token IDs matching PyTorch:
            exact_ids = tok(prompt_str).input_ids

            payload = {
                "prompt": exact_ids,
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

            print(f"Case {cid:02d} [{cat:18s}]: {len(exact_ids):3d} in_tok -> {tokens_n:3d} out_tok in {elapsed:5.2f}s ({tps:5.1f} t/s) | {repr(content[:50])}")

            raw_results.append({
                "id": cid,
                "category": cat,
                "needs_tool": needs_tool,
                "expected_tool": exp_tool,
                "prompt_tokens_len": len(exact_ids),
                "generated_tokens": tokens_n,
                "elapsed_s": elapsed,
                "tps": tps,
                "response": content
            })

    finally:
        print("\nStopping server...")
        stop_process(proc)
        print("Server stopped.")

    total_time = time.perf_counter() - t_start

    # 6. Evaluate strictly under 5-stage criteria
    print(f"\n[5/5] Evaluating responses under strict 5-stage criteria...")
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
        raw_text = raw_results[idx]["response"]

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
        "total_elapsed_s": total_time
    }

    print("\n" + "=" * 80)
    print("FINAL 5-STAGE RESULTS ON IDENTICAL TOKEN IDS (REBUILT C++ I2_S)")
    print("=" * 80)
    print(f"S1: Попытки JSON-вызова:        {summary['s1_attempts']}")
    print(f"S2: Валидная схема конверта:    {summary['s2_valid_schema']}")
    print(f"S3: Статическая эвристика:      {summary['s3_heuristic']}")
    print(f"S4: Локальное исполнение:       {summary['s4_local_exec']}")
    print(f"S5: Корректный результат:       {summary['s5_local_result']}")
    print(f"Негативный контроль (отказы):   {summary['non_tool_refusals']}")
    print("=" * 80)

    final_payload = {
        "config": {
            "model": "BitNet-b1.58-2B-4T (i2_s GGUF, rebuilt with LLM_FFN_RELU_SQR)",
            "input_mode": "Identical HuggingFace Token IDs directly to C++ server",
            "threads": 4,
            "context_limit": 2048,
            "temperature": 0.0,
            "repeat_penalty": 1.15,
            "repeat_last_n": 64,
            "stop_tokens": stop_tokens
        },
        "summary": summary,
        "manifest_snapshot_note": (
            "Встроенный объект 'manifest' является предварительным входным снимком состояния манифеста на момент старта раннера "
            "до запуска инференса (preflight проверяет только наличие и чтение JSON-словаря, без валидации хешей файлов). "
            "Канонический внешний манифест BENCHMARK_ARTIFACTS_MANIFEST.json выступает единственной точкой фиксации. "
            "Атомарной операцией на уровне файловой системы является исключительно одиночный os.replace файла манифеста; "
            "многофайловая процедура опирается на то, что авторитетным объектом объявляется уникальный версионированный файл, "
            "а статус синхронизации канонической копии отслеживается и сохраняется в манифесте отдельно."
        ),
        "manifest": manifest,
        "cases": evaluated_cases
    }

    # 7. Consistent commit of results and manifest synchronization
    print(f"\n[5/5] Performing consistent commit of versioned results and manifest synchronization...")
    commit_res = commit_results_and_sync_manifest(
        final_payload=final_payload,
        output_json=OUTPUT_JSON,
        manifest_json=MANIFEST_JSON
    )
    print("Commit succeeded:")
    print(f"  Authoritative versioned file: {commit_res['versioned_path']}")
    print(f"  SHA-256: {commit_res['sha256']}, Size: {commit_res['size_bytes']} bytes")
    print(f"  Canonical copy: {commit_res['canonical_copy_path']} (status: {commit_res['canonical_copy_status']})")

if __name__ == "__main__":
    main()

