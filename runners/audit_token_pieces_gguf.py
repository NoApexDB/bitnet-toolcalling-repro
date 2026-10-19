import sys
import os
import json
import subprocess
import time
import requests

# Add gguf library path
sys.path.insert(0, r"C:\ai_models\BitNet\3rdparty\llama.cpp\gguf-py")
import gguf
from transformers import AutoTokenizer

LLAMA_SERVER = r"C:\ai_models\BitNet\build\bin\llama-server.exe"
MODEL_GGUF = r"C:\ai_models\BitNet\models\BitNet-b1.58-2B-4T\ggml-model-i2_s.gguf"
MODEL_HF = r"C:\ai_models\BitNet-b1.58-2B-4T"
TEST_SUITE_PATH = r"C:\ai_models\test_suite_standardized.json"
OUTPUT_JSON = r"C:\ai_models\paper_artifacts\token_alignment_full_audit.json"
PORT = 8092

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
    print("=" * 70)
    print("AUDIT: TOKEN IDS AND PIECES (HF VS GGUF DIRECT VOCAB)")
    print("=" * 70)

    # 1. Load GGUF vocabulary directly from binary GGUF
    print("[1/4] Reading GGUF vocabulary directly from file...")
    reader = gguf.GGUFReader(MODEL_GGUF)
    tokens_field = reader.fields.get("tokenizer.ggml.tokens")
    if tokens_field is None:
        raise RuntimeError("tokenizer.ggml.tokens not found in GGUF!")
    
    # Extract actual token strings using contents()
    gguf_vocab = tokens_field.contents()
    print(f"Loaded {len(gguf_vocab)} tokens directly from GGUF binary metadata.")

    # 2. Load HuggingFace tokenizer
    print("[2/4] Loading HuggingFace tokenizer...")
    tok_hf = AutoTokenizer.from_pretrained(MODEL_HF)

    # 3. Start llama-server on port 8092
    print(f"[3/4] Starting llama-server on port {PORT}...")
    proc = subprocess.Popen([
        LLAMA_SERVER,
        "-m", MODEL_GGUF,
        "--host", "127.0.0.1",
        "--port", str(PORT),
        "-t", "4",
        "-c", "2048"
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    try:
        ready = False
        for _ in range(40):
            try:
                r = requests.get(f"http://127.0.0.1:{PORT}/health", timeout=1)
                if r.status_code == 200:
                    ready = True
                    break
            except Exception:
                pass
            time.sleep(0.5)

        if not ready:
            print("ERROR: Failed to start llama-server!")
            return

        with open(TEST_SUITE_PATH, "r", encoding="utf-8") as f:
            cases = json.load(f)

        print(f"[4/4] Comparing tokenization across all {len(cases)} cases...")
        audit_results = []
        delta_summary = {}

        for case in cases:
            cid = case["id"]
            cat = case["category"]
            messages = case["messages"]
            prompt_str = format_prompt(messages, SYSTEM_PROMPT_BASE)

            # PyTorch HF tokens
            hf_with_bos = tok_hf(prompt_str).input_ids
            hf_no_bos = tok_hf.encode(prompt_str, add_special_tokens=False)

            # C++ tokens via /tokenize
            resp = requests.post(f"http://127.0.0.1:{PORT}/tokenize", json={"content": prompt_str})
            cpp_tokens = resp.json().get("tokens", [])

            # Check BOS presence
            cpp_has_bos = (len(cpp_tokens) > 0 and cpp_tokens[0] == 128000)
            hf_has_bos = (len(hf_with_bos) > 0 and hf_with_bos[0] == 128000)

            # Pieces
            hf_pieces = tok_hf.convert_ids_to_tokens(hf_no_bos)
            cpp_pieces = [gguf_vocab[tid] if tid < len(gguf_vocab) else f"<UNK_{tid}>" for tid in cpp_tokens]

            delta_len = len(cpp_tokens) - len(hf_no_bos)
            delta_summary[delta_len] = delta_summary.get(delta_len, 0) + 1

            # Find first difference against hf_no_bos
            first_diff_idx = None
            diff_details = []
            for i, (h_id, c_id) in enumerate(zip(hf_no_bos, cpp_tokens)):
                if h_id != c_id:
                    if first_diff_idx is None:
                        first_diff_idx = i
                    diff_details.append({
                        "pos": i,
                        "hf_id": h_id,
                        "hf_piece": hf_pieces[i],
                        "cpp_id": c_id,
                        "cpp_piece": cpp_pieces[i]
                    })
                    if len(diff_details) >= 10:
                        break

            audit_results.append({
                "case_id": cid,
                "category": cat,
                "prompt_char_len": len(prompt_str),
                "hf_len_with_bos": len(hf_with_bos),
                "hf_len_no_bos": len(hf_no_bos),
                "cpp_len": len(cpp_tokens),
                "cpp_has_bos": cpp_has_bos,
                "delta_cpp_minus_hf_nobos": delta_len,
                "delta_cpp_minus_hf_bos": len(cpp_tokens) - len(hf_with_bos),
                "first_diff_idx": first_diff_idx,
                "sample_diffs": diff_details,
                "hf_tokens": hf_no_bos,
                "cpp_tokens": cpp_tokens,
                "hf_pieces": hf_pieces,
                "cpp_pieces": cpp_pieces
            })

            print(f"Case {cid:2d} | HF(no_bos): {len(hf_no_bos):3d} | HF(+bos): {len(hf_with_bos):3d} | CPP: {len(cpp_tokens):3d} | Delta: +{delta_len:2d} | 1st diff: {first_diff_idx}")

        # Save full audit
        os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
        with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
            json.dump({
                "delta_distribution": delta_summary,
                "cases": audit_results
            }, f, indent=2, ensure_ascii=False)

        print("\n" + "=" * 70)
        print("DELTA DISTRIBUTION (CPP len minus HF no_bos len):")
        for delta, count in sorted(delta_summary.items()):
            cases_with_delta = [c["case_id"] for c in audit_results if c["delta_cpp_minus_hf_nobos"] == delta]
            print(f"  Delta +{delta}: {count:2d} cases (IDs: {cases_with_delta})")
        print(f"Saved full audit JSON to: {OUTPUT_JSON}")
        print("=" * 70)

    finally:
        proc.terminate()
        proc.wait()

if __name__ == "__main__":
    main()
