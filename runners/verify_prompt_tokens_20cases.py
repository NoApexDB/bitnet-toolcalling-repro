import json
import os
import subprocess
import time
import requests
from transformers import AutoTokenizer

LLAMA_SERVER = r"C:\ai_models\BitNet\build\bin\llama-server.exe"
MODEL_GGUF = r"C:\ai_models\BitNet\models\BitNet-b1.58-2B-4T\ggml-model-i2_s.gguf"
MODEL_HF = r"C:\ai_models\BitNet-b1.58-2B-4T"
TEST_SUITE_PATH = r"C:\ai_models\test_suite_standardized.json"
PORT = 8089

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
    print("VERIFYING PROMPT TOKEN IDS ACROSS ALL 20 CASES: PYTORCH VS C++")
    print("=" * 60)

    # Start llama-server on port 8089
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

        tok_hf = AutoTokenizer.from_pretrained(MODEL_HF)

        with open(TEST_SUITE_PATH, "r", encoding="utf-8") as f:
            cases = json.load(f)

        all_matched = True
        case_results = []

        for case in cases:
            cid = case["id"]
            messages = case["messages"]
            prompt_str = format_prompt(messages, SYSTEM_PROMPT_BASE)

            # PyTorch HF tokens
            hf_tokens = tok_hf.encode(prompt_str, add_special_tokens=False)

            # C++ llama-server tokens
            resp = requests.post(f"http://127.0.0.1:{PORT}/tokenize", json={"content": prompt_str})
            cpp_tokens = resp.json().get("tokens", [])

            is_match = (hf_tokens == cpp_tokens)
            if not is_match:
                all_matched = False

            case_results.append({
                "id": cid,
                "hf_len": len(hf_tokens),
                "cpp_len": len(cpp_tokens),
                "match": is_match,
                "diff_idx": next((i for i, (x, y) in enumerate(zip(hf_tokens, cpp_tokens)) if x != y), None) if not is_match else None
            })

            status_str = "MATCH" if is_match else f"MISMATCH (diff at {case_results[-1]['diff_idx']})"
            print(f"Case {cid:02d}: HF={len(hf_tokens)} tokens, C++={len(cpp_tokens)} tokens | {status_str}")

        print("=" * 60)
        print(f"OVERALL PROMPT TOKEN MATCH: {all_matched} (20/20 cases verified)")
        print("=" * 60)

        out_path = r"C:\ai_models\prompt_token_verification_20cases.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"all_matched": all_matched, "cases": case_results}, f, indent=2)
        print(f"Details saved to {out_path}")

    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()

if __name__ == "__main__":
    main()
