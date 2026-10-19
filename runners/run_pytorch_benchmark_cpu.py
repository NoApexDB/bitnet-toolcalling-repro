import functools
import json
import os
import sys
import time
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, StoppingCriteria, StoppingCriteriaList

# Unbuffered print
print = functools.partial(print, flush=True)

# Force UTF-8 stdout
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# Force 4 CPU threads to match C++ -t 4
torch.set_num_threads(4)

MODEL_DIR = r"C:\ai_models\BitNet-b1.58-2B-4T"
TEST_SUITE_PATH = r"C:\ai_models\test_suite_standardized.json"
OUTPUT_PATH = r"C:\ai_models\benchmark_results_pytorch_cpu.json"

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

class LlamaCppRepetitionPenaltyLogitsProcessor(LogitsProcessor):
    """
    Matches llama.cpp repetition penalty behavior:
    Applies penalty only to tokens appearing in the sliding window of last_n tokens (default 64).
    """
    def __init__(self, penalty: float = 1.15, last_n: int = 64):
        self.penalty = penalty
        self.last_n = last_n

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        window = input_ids[:, -self.last_n:]
        for i in range(scores.shape[0]):
            unique_tokens = torch.unique(window[i])
            token_scores = scores[i, unique_tokens]
            penalized = torch.where(token_scores < 0, token_scores * self.penalty, token_scores / self.penalty)
            scores[i, unique_tokens] = penalized
        return scores

class StopOnUserCriteria(StoppingCriteria):
    """
    Stops generation when the token sequence for 'User:' or '\nUser:' is produced,
    matching llama-server -r 'User:'.
    """
    def __init__(self, stop_token_sequences):
        self.stop_token_sequences = stop_token_sequences
        self.triggered = False

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for seq in self.stop_token_sequences:
            seq_len = len(seq)
            if input_ids.shape[1] >= seq_len:
                if input_ids[0, -seq_len:].tolist() == seq:
                    self.triggered = True
                    return True
        return False

def format_prompt(messages, system_prompt):
    prompt = f"System: {system_prompt}<|eot_id|>"
    for m in messages:
        role = "User" if m["role"] == "user" else "Assistant"
        prompt += f"{role}: {m['content']}<|eot_id|>"
    prompt += "Assistant: "
    return prompt

def main():
    print("=" * 60)
    print("BITNET b1.58 PYTORCH CPU BENCHMARK (ALIGNED PROTOCOL)")
    print("=" * 60)
    print(f"Transformers version: {transformers.__version__}")
    print(f"PyTorch version:      {torch.__version__}")
    print(f"Configured threads:   {torch.get_num_threads()}")
    print(f"Model path:           {MODEL_DIR}")
    print(f"Test suite path:      {TEST_SUITE_PATH}")

    # 1. Load tokenizer
    print("\n[1/3] Loading tokenizer...")
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)

    # 2. Load model in float32 on CPU
    print("\n[2/3] Loading BitNetForCausalLM in float32 on CPU...")
    t0_load = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        torch_dtype=torch.float32,
        device_map="cpu",
        low_cpu_mem_usage=True
    )
    model.eval()
    t_load = time.perf_counter() - t0_load
    print(f"Model loaded in {t_load:.2f} s")
    print(f"Model architecture: {model.config.architectures}")
    print(f"Model class: {model.__class__.__name__}")

    # Prepare stopping criteria
    # Tokens for "User:" and "\nUser:"
    user_seqs = [
        tok.encode("User:", add_special_tokens=False),
        tok.encode("\nUser:", add_special_tokens=False)
    ]
    eos_ids = [tok.eos_token_id, 128001, 128009]
    eos_ids = list(set([x for x in eos_ids if x is not None]))

    # Load test cases
    with open(TEST_SUITE_PATH, "r", encoding="utf-8") as f:
        test_cases = json.load(f)

    print(f"\n[3/3] Running benchmark across {len(test_cases)} cases...")
    print("Alignment: max_new_tokens=128, temp=0.0, repeat_penalty=1.15 (window 64), stop=['<|eot_id|>', '<|end_of_text|>', 'User:']")
    print("-" * 60)

    detailed_results = []
    total_tokens_all = 0
    t_start_bench = time.perf_counter()

    for idx, case in enumerate(test_cases):
        case_id = case["id"]
        category = case["category"]
        needs_tool = case["needs_tool"]
        exp_tool = case["expected_tool"]
        messages = case["messages"]
        user_query = messages[-1]["content"]

        prompt_str = format_prompt(messages, SYSTEM_PROMPT_BASE)
        inputs = tok(prompt_str, return_tensors="pt").to("cpu")
        prompt_len = inputs.input_ids.shape[1]

        # Check context limit
        if prompt_len > 2048:
            print(f"Warning: prompt length {prompt_len} exceeds 2048!")

        # Custom processors and criteria per case
        rep_proc = LlamaCppRepetitionPenaltyLogitsProcessor(penalty=1.15, last_n=64)
        stop_crit = StopOnUserCriteria(user_seqs)

        t0_case = time.perf_counter()
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                logits_processor=[rep_proc],
                stopping_criteria=StoppingCriteriaList([stop_crit]),
                pad_token_id=tok.eos_token_id,
                eos_token_id=eos_ids
            )
        elapsed = time.perf_counter() - t0_case

        gen_tokens = out[0][prompt_len:]
        num_gen = len(gen_tokens)
        total_tokens_all += num_gen
        tps = num_gen / elapsed if elapsed > 0 else 0.0

        raw_decoded = tok.decode(gen_tokens, skip_special_tokens=False)

        # Clean response matching llama-server behavior:
        # Strip trailing stop sequences (<|eot_id|>, <|end_of_text|>, User:)
        cleaned_text = raw_decoded
        for stop_str in ["<|eot_id|>", "<|end_of_text|>", "User:"]:
            if cleaned_text.endswith(stop_str):
                cleaned_text = cleaned_text[:-len(stop_str)]
        cleaned_text = cleaned_text.strip()

        stop_reason = "eos_token"
        if stop_crit.triggered:
            stop_reason = "stop_sequence_user"
        elif num_gen >= 128:
            stop_reason = "max_tokens"

        print(f"Case {case_id:02d} [{category:18s}]: {num_gen:3d} tokens in {elapsed:5.2f}s ({tps:4.1f} t/s) | Stop: {stop_reason}")
        print(f"    Raw preview: {repr(cleaned_text[:70])}")

        detailed_results.append({
            "id": case_id,
            "category": category,
            "needs_tool": needs_tool,
            "expected_tool": exp_tool,
            "user_query": user_query,
            "prompt_tokens": prompt_len,
            "generated_tokens": num_gen,
            "elapsed_sec": round(elapsed, 4),
            "tokens_per_sec": round(tps, 2),
            "response": cleaned_text,
            "raw_output": raw_decoded,
            "stop_reason": stop_reason
        })

    total_bench_time = time.perf_counter() - t_start_bench
    mean_tps = total_tokens_all / total_bench_time if total_bench_time > 0 else 0.0

    print("=" * 60)
    print("BENCHMARK COMPLETED")
    print(f"Total time:       {total_bench_time:.2f} s")
    print(f"Total tokens:     {total_tokens_all}")
    print(f"Mean generation:  {mean_tps:.2f} t/s")
    print("=" * 60)

    output_payload = {
        "model_name": "BitNet b1.58 Base (PyTorch CPU)",
        "model_path": MODEL_DIR,
        "runtime": "PyTorch / HuggingFace Transformers (CPU)",
        "environment": {
            "transformers_version": transformers.__version__,
            "torch_version": torch.__version__,
            "cpu_threads": torch.get_num_threads(),
            "device": "cpu",
            "model_class": model.__class__.__name__,
            "compute_dtype": "torch.float32",
            "repetition_penalty": 1.15,
            "repeat_last_n": 64,
            "max_context_length": 2048,
            "max_new_tokens": 128,
            "temperature": 0.0,
            "stop_sequences": ["<|eot_id|>", "<|end_of_text|>", "User:"]
        },
        "performance": {
            "total_tokens": total_tokens_all,
            "wall_clock_sec": round(total_bench_time, 2),
            "mean_tps": round(mean_tps, 2)
        },
        "detailed_results": detailed_results
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, ensure_ascii=False, indent=2)

    print(f"Results successfully saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
