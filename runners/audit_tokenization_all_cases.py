import os
import sys
import json
import time
import socket
import requests
import subprocess
import psutil
import transformers
import tokenizers
from transformers import AutoTokenizer

MODEL_DIR = r"C:\ai_models\BitNet-b1.58-2B-4T"
GGUF_PATH = r"C:\ai_models\BitNet\models\BitNet-b1.58-2B-4T\ggml-model-i2_s.gguf"
SERVER_EXE = r"C:\ai_models\BitNet\build\bin\llama-server-relu2.exe"
TEST_SUITE = r"C:\ai_models\test_suite_standardized.json"
OUT_JSON = r"C:\ai_models\paper_artifacts\string_vs_hf_tokenization_audit_20_cases.json"
PORT = 8080

# Exact prompt from run_2x2_factorial_benchmark.py (WITH "AI assistant")
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

def decompose_tokenization_diff(hf_ids, hf_pieces, cpp_ids, cpp_pieces):
    """
    Performs full text reconstruction alignment to identify every single split or substitution.
    """
    diff_events = []
    
    # We can align by advancing text offsets
    hf_idx = 0
    cpp_idx = 0
    
    hf_len = len(hf_ids)
    cpp_len = len(cpp_ids)
    
    # Simple sliding alignment
    # Check at each step if tokens match directly or if one side has multiple tokens covering the same text
    while hf_idx < hf_len and cpp_idx < cpp_len:
        h_id = hf_ids[hf_idx]
        c_id = cpp_ids[cpp_idx]
        
        if h_id == c_id:
            hf_idx += 1
            cpp_idx += 1
            continue
            
        # Mismatch found: find how many tokens on each side match up to the next sync point
        found_sync = False
        for h_ahead in range(1, 10):
            if hf_idx + h_ahead > hf_len:
                break
            for c_ahead in range(1, 10):
                if cpp_idx + c_ahead > cpp_len:
                    break
                # Check if text reconstructed matches
                h_text = "".join(hf_pieces[hf_idx : hf_idx + h_ahead])
                c_text = "".join(cpp_pieces[cpp_idx : cpp_idx + c_ahead])
                if h_text == c_text:
                    diff_events.append({
                        "hf_range": [hf_idx, hf_idx + h_ahead - 1],
                        "cpp_range": [cpp_idx, cpp_idx + c_ahead - 1],
                        "text": h_text,
                        "hf_tokens": [{"id": hf_ids[i], "piece": hf_pieces[i]} for i in range(hf_idx, hf_idx + h_ahead)],
                        "cpp_tokens": [{"id": cpp_ids[i], "piece": cpp_pieces[i]} for i in range(cpp_idx, cpp_idx + c_ahead)],
                        "extra_tokens_in_cpp": c_ahead - h_ahead,
                        "description": f"HF {h_ahead} token(s) vs CPP {c_ahead} token(s) for {repr(h_text)}"
                    })
                    hf_idx += h_ahead
                    cpp_idx += c_ahead
                    found_sync = True
                    break
            if found_sync:
                break
                
        if not found_sync:
            # Fallback advance by 1
            diff_events.append({
                "hf_index": hf_idx,
                "cpp_index": cpp_idx,
                "hf_token": {"id": h_id, "piece": hf_pieces[hf_idx]},
                "cpp_token": {"id": c_id, "piece": cpp_pieces[cpp_idx]},
                "extra_tokens_in_cpp": 0,
                "description": f"Mismatch at HF {hf_idx} vs CPP {cpp_idx}: {repr(hf_pieces[hf_idx])} vs {repr(cpp_pieces[cpp_idx])}"
            })
            hf_idx += 1
            cpp_idx += 1
            
    # Remaining tokens if any
    if hf_idx < hf_len or cpp_idx < cpp_len:
        diff_events.append({
            "tail_hf_remaining": hf_len - hf_idx,
            "tail_cpp_remaining": cpp_len - cpp_idx,
            "extra_tokens_in_cpp": (cpp_len - cpp_idx) - (hf_len - hf_idx),
            "description": f"Tail mismatch: {hf_len - hf_idx} HF tokens vs {cpp_len - cpp_idx} CPP tokens"
        })
        
    return diff_events

def main():
    print(f"Loading HF tokenizer (default fast PreTrainedTokenizerFast)...")
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    print(f"Tokenizer class: {tok.__class__.__name__}, is_fast: {tok.is_fast}")
    print(f"Transformers version: {transformers.__version__}, tokenizers: {tokenizers.__version__}")
    
    with open(TEST_SUITE, "r", encoding="utf-8") as f:
        cases = json.load(f)
        
    if is_port_in_use(PORT):
        alien_pid = get_listening_pid(PORT)
        raise RuntimeError(
            f"Port {PORT} is already occupied (PID {alien_pid}) prior to launching {SERVER_EXE}. "
            f"Aborting to avoid directing audit queries to an alien server."
        )

    print("Starting llama-server-relu2.exe to query /tokenize...")
    cmd = [
        SERVER_EXE,
        "-m", GGUF_PATH,
        "-c", "2048",
        "-t", "4",
        "--host", "127.0.0.1",
        "--port", str(PORT)
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
                f"Server process {SERVER_EXE} (PID {proc.pid}) died prematurely with returncode {exit_code} during startup."
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
                    f"Server process {SERVER_EXE} (PID {proc.pid}) died immediately after /health responded."
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
        print("Failed to start server within timeout!")
        sys.exit(1)
        
    print("Server ready. Running exact tokenization audit on 20 cases...")
    audit_results = []
    
    try:
        for idx, c in enumerate(cases, 1):
            prompt_str = format_prompt(c["messages"], SYSTEM_PROMPT_BASE)
            
            # HF tokenization (identical to run_2x2_factorial_benchmark.py)
            hf_tokens = tok(prompt_str).input_ids
            hf_pieces = [tok.decode([tid]) for tid in hf_tokens]
            
            # C++ tokenization via /tokenize (identical to what server processes on string prompt)
            r = session.post(f"http://127.0.0.1:{PORT}/tokenize", json={
                "content": prompt_str,
                "add_special": True,
                "parse_special": True,
                "with_pieces": True
            })
            cpp_data = r.json()
            cpp_raw = cpp_data.get("tokens", [])
            
            cpp_tokens = []
            cpp_pieces = []
            for item in cpp_raw:
                if isinstance(item, dict):
                    cpp_tokens.append(item.get("id"))
                    cpp_pieces.append(item.get("piece", ""))
                else:
                    cpp_tokens.append(item)
                    cpp_pieces.append("")
                    
            exact_match = (hf_tokens == cpp_tokens)
            
            first_mismatch_idx_0 = None
            if not exact_match:
                min_len = min(len(hf_tokens), len(cpp_tokens))
                for i in range(min_len):
                    if hf_tokens[i] != cpp_tokens[i]:
                        first_mismatch_idx_0 = i
                        break
                if first_mismatch_idx_0 is None and len(hf_tokens) != len(cpp_tokens):
                    first_mismatch_idx_0 = min_len
                    
            first_mismatch_pos_1 = (first_mismatch_idx_0 + 1) if first_mismatch_idx_0 is not None else None
            
            # Full sequential diff breakdown
            diff_events = decompose_tokenization_diff(hf_tokens, hf_pieces, cpp_tokens, cpp_pieces)
            extra_sum = sum(e.get("extra_tokens_in_cpp", 0) for e in diff_events)
            
            # Categorize the diff causes
            category_counts = {}
            for e in diff_events:
                desc = e.get("description", "")
                text = e.get("text", "")
                if ".\n" in text or text == ".\n":
                    k = "dot_newline_split ('.\\n' -> '.' + '\\n')"
                elif "\n\n" in text or text == "\n\n":
                    k = "double_newline_split ('\\n\\n' -> '\\n' + '\\n')"
                elif "\n" in text:
                    k = "other_newline_split (newline combination)"
                elif "```" in text:
                    k = "markdown_code_fence_split ('```')"
                elif any(c.isdigit() for c in text):
                    k = "digit_split (numbers/digits)"
                else:
                    k = f"subword_or_punct_split ({repr(text[:20])})"
                category_counts[k] = category_counts.get(k, 0) + e.get("extra_tokens_in_cpp", 0)

            audit_results.append({
                "case_id": c["id"],
                "category": c["category"],
                "prompt_snippet": c["messages"][-1]["content"][:60],
                "hf_len": len(hf_tokens),
                "cpp_len": len(cpp_tokens),
                "len_diff": len(cpp_tokens) - len(hf_tokens),
                "exact_match": exact_match,
                "first_mismatch_index_0_based": first_mismatch_idx_0,
                "first_mismatch_position_1_based": first_mismatch_pos_1,
                "first_mismatch_hf_token": {
                    "id": hf_tokens[first_mismatch_idx_0] if first_mismatch_idx_0 < len(hf_tokens) else None,
                    "piece": hf_pieces[first_mismatch_idx_0] if first_mismatch_idx_0 < len(hf_tokens) else None
                },
                "first_mismatch_cpp_token": {
                    "id": cpp_tokens[first_mismatch_idx_0] if first_mismatch_idx_0 < len(cpp_tokens) else None,
                    "piece": cpp_pieces[first_mismatch_idx_0] if first_mismatch_idx_0 < len(cpp_tokens) else None
                },
                "extra_tokens_explained_sum": extra_sum,
                "diff_category_breakdown": category_counts,
                "diff_events_count": len(diff_events),
                "diff_events": diff_events,
                "hf_tokens": hf_tokens,
                "cpp_tokens": cpp_tokens,
                "hf_pieces": hf_pieces,
                "cpp_pieces": cpp_pieces
            })
            
            print(f"Case {idx:02d}: diff={len(cpp_tokens)-len(hf_tokens):+2d} tok | 1st mismatch @ 0-idx={first_mismatch_idx_0} (1-pos={first_mismatch_pos_1}) | HF='{hf_pieces[first_mismatch_idx_0]}' vs CPP='{cpp_pieces[first_mismatch_idx_0]}'")
            
    finally:
        stop_process(proc)
        
    full_output = {
        "metadata": {
            "title": "Exact Tokenization Audit: Hugging Face PreTrainedTokenizerFast vs bitnet.cpp /tokenize",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "model_dir": MODEL_DIR,
            "gguf_model": GGUF_PATH,
            "transformers_version": transformers.__version__,
            "tokenizers_version": tokenizers.__version__,
            "tokenizer_class": tok.__class__.__name__,
            "is_fast": tok.is_fast,
            "system_prompt_used": "SYSTEM_PROMPT_BASE with 'helpful AI assistant'"
        },
        "summary": {
            "total_cases": len(audit_results),
            "exact_matches": sum(1 for a in audit_results if a["exact_match"]),
            "mismatches": sum(1 for a in audit_results if not a["exact_match"]),
            "length_difference_range": [min(a["len_diff"] for a in audit_results), max(a["len_diff"] for a in audit_results)],
            "first_mismatch_position_1_based_all": list(set(a["first_mismatch_position_1_based"] for a in audit_results)),
            "first_mismatch_index_0_based_all": list(set(a["first_mismatch_index_0_based"] for a in audit_results))
        },
        "cases": audit_results
    }
    
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(full_output, f, indent=2, ensure_ascii=False)
        
    print(f"\nAudit saved to {OUT_JSON}")
    print(f"Summary: {full_output['summary']}")

if __name__ == "__main__":
    main()
