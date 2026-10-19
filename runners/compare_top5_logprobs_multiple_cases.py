import os
import sys
import json
import time
import subprocess
import requests
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.stdout.reconfigure(encoding="utf-8")

LLAMA_SERVER = r"C:\ai_models\BitNet\build\bin\llama-server.exe"
MODEL_GGUF = r"C:\ai_models\BitNet\models\BitNet-b1.58-2B-4T\ggml-model-i2_s.gguf"
MODEL_HF = r"C:\ai_models\BitNet-b1.58-2B-4T"
TEST_SUITE_PATH = r"C:\ai_models\test_suite_standardized.json"
OUTPUT_JSON = r"C:\ai_models\paper_artifacts\top5_logprob_comparison_cases.json"
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

def format_prompt(messages, system_prompt):
    prompt = f"System: {system_prompt}<|eot_id|>"
    for m in messages:
        role = "User" if m["role"] == "user" else "Assistant"
        prompt += f"{role}: {m['content']}<|eot_id|>"
    prompt += "Assistant: "
    return prompt

def main():
    print("=" * 75)
    print("STEP 3: TOP-5 LOGPROB ON FIRST TOKEN ACROSS MULTIPLE CASES (HF TOKENS INPUT)")
    print("=" * 75)

    # 1. Start llama-server on port 8094
    print(f"Starting llama-server on port {PORT}...")
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

        # 2. Load PyTorch model
        print("Loading PyTorch model and tokenizer on CPU...")
        tok = AutoTokenizer.from_pretrained(MODEL_HF)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_HF,
            torch_dtype=torch.float32,
            device_map="cpu",
            low_cpu_mem_usage=True
        )
        model.eval()

        with open(TEST_SUITE_PATH, "r", encoding="utf-8") as f:
            cases = json.load(f)

        target_cases = [1, 2, 11, 16] # Weather, Weather(EN), SQL, Negative
        comparison_results = []

        for cid in target_cases:
            case = next(c for c in cases if c["id"] == cid)
            prompt_str = format_prompt(case["messages"], SYSTEM_PROMPT_BASE)
            # HF token IDs (including BOS 128000)
            hf_input_ids = tok(prompt_str).input_ids
            prompt_len = len(hf_input_ids)

            # PyTorch forward pass
            t_in = torch.tensor([hf_input_ids], dtype=torch.long)
            with torch.no_grad():
                pt_out = model(t_in)
                pt_logits = pt_out.logits[0, -1, :]
                pt_logprobs = torch.log_softmax(pt_logits, dim=-1)
                pt_topk = torch.topk(pt_logprobs, k=5)
                pt_top_ids = pt_topk.indices.tolist()
                pt_top_lps = pt_topk.values.tolist()

            pt_distribution = [
                {"id": tid, "token": tok.decode([tid]), "logprob": lp}
                for tid, lp in zip(pt_top_ids, pt_top_lps)
            ]

            # C++ forward pass via /completion
            payload = {
                "prompt": hf_input_ids,
                "n_predict": 1,
                "temperature": 0.0,
                "n_probs": 5
            }
            resp_cpp = requests.post(f"http://127.0.0.1:{PORT}/completion", json=payload).json()
            cpp_probs = resp_cpp.get("completion_probabilities", [{}])[0].get("top_logprobs", [])
            cpp_distribution = [
                {"id": e.get("id"), "token": e.get("token"), "logprob": e.get("logprob")}
                for e in cpp_probs
            ]

            comparison_results.append({
                "case_id": cid,
                "category": case["category"],
                "user_query": case["messages"][-1]["content"],
                "prompt_token_count": prompt_len,
                "pytorch_top5": pt_distribution,
                "cpp_i2s_top5": cpp_distribution
            })

            print(f"\n--- CASE {cid:02d} ({case['category']}): {repr(case['messages'][-1]['content'])} ---")
            print("  PyTorch Top-5:")
            for rank, item in enumerate(pt_distribution, 1):
                print(f"    #{rank}: ID {item['id']:6d} | Logprob: {item['logprob']:8.4f} | Token: {repr(item['token'])}")
            print("  C++ I2_S Top-5:")
            for rank, item in enumerate(cpp_distribution, 1):
                print(f"    #{rank}: ID {item['id']:6d} | Logprob: {item['logprob']:8.4f} | Token: {repr(item['token'])}")

        os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
        with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
            json.dump(comparison_results, f, indent=2, ensure_ascii=False)
        print(f"\nSaved multi-case top-5 distributions to: {OUTPUT_JSON}")

    finally:
        proc.terminate()
        proc.wait()

if __name__ == "__main__":
    main()
