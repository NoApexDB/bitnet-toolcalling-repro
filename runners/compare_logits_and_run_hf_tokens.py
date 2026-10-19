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
OUTPUT_BENCHMARK = r"C:\ai_models\benchmark_results_bitnet_i2s_with_hf_tokens.json"
PORT = 8093

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
    print("EXPERIMENT: DIRECT HF TOKEN IDS IN C++ I2_S (LOGITS & BENCHMARK)")
    print("=" * 70)

    # 1. Start llama-server on port 8093
    print(f"[1/4] Starting llama-server on port {PORT}...")
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
        print("llama-server is ready.")

        # 2. Load HuggingFace Model & Tokenizer on CPU
        print("[2/4] Loading PyTorch HF model and tokenizer on CPU (torch_dtype=float32)...")
        tok = AutoTokenizer.from_pretrained(MODEL_HF)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_HF,
            torch_dtype=torch.float32,
            device_map="cpu",
            low_cpu_mem_usage=True
        )
        model.eval()
        torch.set_num_threads(4)
        print("PyTorch model loaded successfully.")

        # 3. Step A: Compare Logits on Identical Input (Case 1)
        with open(TEST_SUITE_PATH, "r", encoding="utf-8") as f:
            cases = json.load(f)

        case1 = cases[0]
        prompt1 = format_prompt(case1["messages"], SYSTEM_PROMPT_BASE)
        # Exactly what PyTorch receives in run_pytorch_benchmark_cpu.py:
        hf_input_ids = tok(prompt1).input_ids # List of ints starting with 128000 (BOS)
        print(f"\n[3/4] Step A: Comparing Logits on Case 1 prompt ({len(hf_input_ids)} tokens, BOS={hf_input_ids[0]})...")

        # PyTorch forward pass
        t_pt_in = torch.tensor([hf_input_ids], dtype=torch.long)
        with torch.no_grad():
            pt_out = model(t_pt_in)
            pt_logits = pt_out.logits[0, -1, :] # shape [vocab_size]
            pt_logprobs = torch.log_softmax(pt_logits, dim=-1)
            pt_topk = torch.topk(pt_logprobs, k=5)
            pt_top_ids = pt_topk.indices.tolist()
            pt_top_logprobs = pt_topk.values.tolist()

        # C++ forward pass via /completion with n_predict=1, n_probs=5
        payload_cpp = {
            "prompt": hf_input_ids,
            "n_predict": 1,
            "temperature": 0.0,
            "n_probs": 5
        }
        resp_cpp = requests.post(f"http://127.0.0.1:{PORT}/completion", json=payload_cpp).json()
        cpp_probs = resp_cpp.get("completion_probabilities", [{}])[0].get("top_logprobs", [])

        print("\n--- PYTORCH TOP 5 NEXT-TOKEN PREDICTIONS ---")
        for rank, (tid, lp) in enumerate(zip(pt_top_ids, pt_top_logprobs), 1):
            tok_str = tok.decode([tid])
            print(f"  #{rank}: ID {tid:6d} | Logprob: {lp:8.4f} | Token: {repr(tok_str)}")

        print("\n--- C++ I2_S TOP 5 NEXT-TOKEN PREDICTIONS (SAME TOKEN IDS INPUT) ---")
        for rank, entry in enumerate(cpp_probs, 1):
            tid = entry.get("id")
            lp = entry.get("logprob")
            tok_str = entry.get("token")
            print(f"  #{rank}: ID {tid:6d} | Logprob: {lp:8.4f} | Token: {repr(tok_str)}")

        # 4. Step B: Run Full Generation Benchmark on 20 Cases with exact HF token IDs
        print(f"\n[4/4] Step B: Running Full 20-Case Benchmark in C++ I2_S with Direct HF Token IDs...")
        
        benchmark_results = []
        stop_tokens = ["<|eot_id|>", "<|end_of_text|>", "User:"]

        for case in cases:
            cid = case["id"]
            cat = case["category"]
            exp_tool = case["expected_tool"]
            needs_tool = case["needs_tool"]
            prompt_str = format_prompt(case["messages"], SYSTEM_PROMPT_BASE)
            exact_ids = tok(prompt_str).input_ids # includes BOS

            payload = {
                "prompt": exact_ids,
                "n_predict": 128,
                "temperature": 0.0,
                "repeat_penalty": 1.15,
                "repeat_last_n": 64,
                "stop": stop_tokens
            }

            t0 = time.perf_counter()
            r = requests.post(f"http://127.0.0.1:{PORT}/completion", json=payload, timeout=60).json()
            elapsed = time.perf_counter() - t0

            content = r.get("content", "").strip()
            timings = r.get("timings", {})
            tokens_n = timings.get("predicted_n", 0)
            tps = timings.get("predicted_per_second", 0.0)

            benchmark_results.append({
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

            first_line = content.split('\n')[0][:60] if content else "<EMPTY>"
            print(f"Case {cid:2d} | Tokens: {tokens_n:3d} | {elapsed:.2f}s | Resp: {repr(first_line)}")

        with open(OUTPUT_BENCHMARK, "w", encoding="utf-8") as f:
            json.dump({
                "config": {
                    "model": "BitNet-b1.58-2B-4T (ggml-model-i2_s.gguf)",
                    "mode": "Direct HF Token IDs (includes BOS=128000)",
                    "temperature": 0.0,
                    "repeat_penalty": 1.15,
                    "repeat_last_n": 64,
                    "stop": stop_tokens
                },
                "results": benchmark_results
            }, f, indent=2, ensure_ascii=False)
        print(f"\nSaved direct HF tokens benchmark to: {OUTPUT_BENCHMARK}")

    finally:
        proc.terminate()
        proc.wait()

if __name__ == "__main__":
    main()
