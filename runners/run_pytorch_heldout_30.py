# -*- coding: utf-8 -*-
"""
PyTorch (FP32, CPU, 4 threads) run of BitNet b1.58 2B on the 30-case challenge suite.

Closes the gap flagged in the audit chain: "PyTorch was not run on the 30-case set".
Protocol matches run_heldout_factorial_benchmark.py exactly:
  * same system prompt (without 'AI') as the 30-case C++ cells
  * same generation params (T=0.0, rep_pen 1.15 window 64, max_new_tokens 128, context 2048)
  * same S1..S5 evaluator and same denominators (8 local-applicable tasks, reached executor,
    valid conversational refusals vs degenerated echo)
Output: NEW file paper_artifacts/pytorch_heldout_30_results.json (nothing overwritten).
"""
import functools
import json
import os
import sys
import time

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, StoppingCriteria, StoppingCriteriaList

print = functools.partial(print, flush=True)
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, r"C:\ai_models")
from evaluator_rigorous import (
    parse_tool_envelope,
    evaluate_static_heuristic_arguments,
    evaluate_execution_and_result_sandboxed,
)

torch.set_num_threads(4)

MODEL_DIR = r"C:\ai_models\BitNet-b1.58-2B-4T"
TEST_SUITE = r"C:\ai_models\heldout_test_suite_30.json"
OUTPUT = r"C:\ai_models\paper_artifacts\pytorch_heldout_30_results.json"
OUTPUT_DESKTOP = r"C:\Users\Рам\Desktop\BitNet_Research_Reports\pytorch_heldout_30_results.json"
LOCAL_TOOLS = {"file_read", "query_database"}

SYSTEM_PROMPT_BASE = """You are a helpful assistant with access to tools.
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
    def __init__(self, penalty: float = 1.15, last_n: int = 64):
        self.penalty = penalty
        self.last_n = last_n

    def __call__(self, input_ids, scores):
        window = input_ids[:, -self.last_n:]
        for i in range(scores.shape[0]):
            unique_tokens = torch.unique(window[i])
            token_scores = scores[i, unique_tokens]
            penalized = torch.where(token_scores < 0, token_scores * self.penalty, token_scores / self.penalty)
            scores[i, unique_tokens] = penalized
        return scores


class StopOnUserCriteria(StoppingCriteria):
    def __init__(self, stop_token_sequences):
        self.stop_token_sequences = stop_token_sequences
        self.triggered = False

    def __call__(self, input_ids, scores, **kwargs):
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


def classify_state(case, s1, s2, s3, s4_status, s5_status, raw_text):
    if not case["needs_tool"]:
        if s1:
            return "FALSE_POSITIVE_TOOL_CALL"
        if "1." in raw_text or "Available tools" in raw_text or len(raw_text) < 5:
            return "DEGENERATED_ECHO_NO_TOOL"
        return "VALID_CONVERSATIONAL_REFUSAL"
    if not s1:
        return "DEGENERATED_LOOP_NO_TOOL"
    if not s2:
        return "HALLUCINATED_OR_INVALID_SCHEMA"
    if not s3:
        return "VALID_SCHEMA_BAD_ARGS"
    if s4_status == "[+]" and s5_status == "[+]" :
        return "FULL_EXECUTION_SUCCESS"
    return "VALID_TOOL_CALL"


def main():
    print("=" * 70)
    print("PYTORCH FP32 CPU ON 30-CASE CHALLENGE SUITE (no-'AI' prompt, aligned protocol)")
    print("=" * 70)
    print(f"transformers {transformers.__version__} | torch {torch.__version__} | threads {torch.get_num_threads()}")

    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, torch_dtype=torch.float32, device_map="cpu", low_cpu_mem_usage=True
    )
    model.eval()
    print(f"Model loaded in {time.perf_counter()-t0:.1f}s | {model.__class__.__name__}")

    with open(TEST_SUITE, encoding="utf-8") as f:
        cases = json.load(f)
    print(f"{len(cases)} cases | positives {sum(1 for c in cases if c['needs_tool'])} | negatives {sum(1 for c in cases if not c['needs_tool'])}")

    user_seqs = [tok.encode("User:", add_special_tokens=False), tok.encode("\nUser:", add_special_tokens=False)]
    eos_ids = [tok.eos_token_id, 128001, 128009]
    eos_ids = list({x for x in eos_ids if x is not None})

    evaluated = []
    total_tokens = 0
    t_start = time.perf_counter()

    for case in cases:
        cid = case["id"]
        prompt_str = format_prompt(case["messages"], SYSTEM_PROMPT_BASE)
        inputs = tok(prompt_str, return_tensors="pt").to("cpu")
        prompt_len = inputs.input_ids.shape[1]

        rep_proc = LlamaCppRepetitionPenaltyLogitsProcessor(1.15, 64)
        stop_crit = StopOnUserCriteria(user_seqs)

        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                logits_processor=[rep_proc],
                stopping_criteria=StoppingCriteriaList([stop_crit]),
                pad_token_id=tok.eos_token_id,
                eos_token_id=eos_ids,
            )
        elapsed = time.perf_counter() - t0

        gen_tokens = out[0][prompt_len:]
        num_gen = len(gen_tokens)
        total_tokens += num_gen
        raw_decoded = tok.decode(gen_tokens, skip_special_tokens=False)
        cleaned = raw_decoded
        for s in ["<|eot_id|>", "<|end_of_text|>", "User:"]:
            if cleaned.endswith(s):
                cleaned = cleaned[: -len(s)]
        cleaned = cleaned.strip()

        s1, s2, envelope, env_err = parse_tool_envelope(cleaned)
        s3 = False
        s4_status = "N/A"
        s5_status = "N/A"
        detail = ""
        if case["needs_tool"]:
            if s2:
                s3, s3_detail = evaluate_static_heuristic_arguments(case, envelope)
            if s3:
                s4_res, s5_res, detail = evaluate_execution_and_result_sandboxed(case, envelope)
                s4_status = "[+]" if s4_res is True else ("[-]" if s4_res is False else str(s4_res))
                s5_status = "[+]" if s5_res is True else ("[-]" if s5_res is False else str(s5_res))
            elif not s2:
                detail = env_err or "Невалидная схема"
            else:
                detail = s3_detail
        state = classify_state(case, s1, s2, s3, s4_status, s5_status, cleaned)
        print(f"Case {cid:2d} [{case['category']:16s}]: {num_gen:3d} tok {elapsed:5.1f}s | {state:32s} | {repr(cleaned[:38])}")

        evaluated.append({
            "id": cid,
            "category": case["category"],
            "needs_tool": case["needs_tool"],
            "expected_tool": case.get("expected_tool"),
            "prompt_tokens": prompt_len,
            "generated_tokens": num_gen,
            "elapsed_s": round(elapsed, 2),
            "response": cleaned,
            "s1_tool_attempted": s1,
            "s2_schema_valid": s2,
            "s3_heuristic_valid": s3,
            "s4_executed": s4_status,
            "s5_result_correct": s5_status,
            "case_state": state,
            "detail": detail,
        })

    wall = time.perf_counter() - t_start

    tool_cases = [e for e in evaluated if e["needs_tool"]]
    non_tool_cases = [e for e in evaluated if not e["needs_tool"]]
    local_cases = [e for e in tool_cases if e["expected_tool"] in LOCAL_TOOLS]
    local_reached = [e for e in local_cases if e["s3_heuristic_valid"]]
    local_s5_success = [e for e in local_cases if e["s5_result_correct"] == "[+]"]
    valid_refusals = [e for e in non_tool_cases if e["case_state"] == "VALID_CONVERSATIONAL_REFUSAL"]
    degenerated = [e for e in non_tool_cases if e["case_state"] == "DEGENERATED_ECHO_NO_TOOL"]

    def pct(n, d):
        return f"{n}/{d} ({n/d*100:.1f}%)" if d else "N/A"

    summary = {
        "s1_attempts": pct(sum(1 for e in tool_cases if e["s1_tool_attempted"]), len(tool_cases)),
        "s2_valid_schema": pct(sum(1 for e in tool_cases if e["s2_schema_valid"]), len(tool_cases)),
        "s3_heuristic": pct(sum(1 for e in tool_cases if e["s3_heuristic_valid"]), len(tool_cases)),
        "s4_s5_local_applicable_tasks": len(local_cases),
        "local_applicable_case_ids": [e["id"] for e in local_cases],
        "s4_s5_local_reached_executor": len(local_reached),
        "s4_s5_success_among_actual_runs": pct(len(local_s5_success), len(local_reached)) if local_reached else "0/0 (N/A)",
        "s4_s5_overall_local_applicable": pct(len(local_s5_success), len(local_cases)),
        "s4_s5_end_to_end_all_tool_queries": pct(sum(1 for e in tool_cases if e["s5_result_correct"] == "[+]"), len(tool_cases)),
        "external_api_and_python_unmocked": f"{len(tool_cases) - len(local_cases)} (N/A)",
        "valid_conversational_refusals": pct(len(valid_refusals), len(non_tool_cases)),
        "degenerated_no_tool": pct(len(degenerated), len(non_tool_cases)),
        "total_elapsed_s": round(wall, 1),
        "s1_n": sum(1 for e in tool_cases if e["s1_tool_attempted"]),
        "s2_n": sum(1 for e in tool_cases if e["s2_schema_valid"]),
        "s3_n": sum(1 for e in tool_cases if e["s3_heuristic_valid"]),
        "s5_local_n": len(local_s5_success),
        "valid_refusals_n": len(valid_refusals),
        "degenerated_no_tool_n": len(degenerated),
    }

    payload = {
        "title": "BitNet b1.58 2B, PyTorch FP32 CPU, 30-case challenge suite (no-'AI' prompt)",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "environment": {
            "transformers_version": transformers.__version__,
            "torch_version": torch.__version__,
            "cpu_threads": torch.get_num_threads(),
            "device": "cpu",
            "model_class": model.__class__.__name__,
            "compute_dtype": "torch.float32",
            "temperature": 0.0,
            "repeat_penalty": 1.15,
            "repeat_last_n": 64,
            "max_new_tokens": 128,
            "stop_sequences": ["<|eot_id|>", "<|end_of_text|>", "User:"],
            "system_prompt_note": "identical to run_heldout_factorial_benchmark.py (without 'AI')",
        },
        "performance": {
            "total_tokens": total_tokens,
            "wall_clock_sec": round(wall, 1),
            "mean_tps": round(total_tokens / wall, 2) if wall else 0,
        },
        "summary": summary,
        "cases": evaluated,
    }

    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    try:
        import shutil
        shutil.copyfile(OUTPUT, OUTPUT_DESKTOP)
    except Exception as e:
        print(f"[warn] desktop copy failed: {e}")

    print("\n" + "=" * 70)
    print(f"SUMMARY: S2 {summary['s2_valid_schema']} | S3 {summary['s3_heuristic']} | "
          f"local e2e {summary['s4_s5_overall_local_applicable']} | refusals {summary['valid_conversational_refusals']} | "
          f"degenerated {summary['degenerated_no_tool']}")
    print(f"Saved -> {OUTPUT}")


if __name__ == "__main__":
    main()
