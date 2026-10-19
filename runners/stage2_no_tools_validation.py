# -*- coding: utf-8 -*-
"""
PHASE 2.1b: Stage-2 validation of the two-stage agent architecture.

Stage 1 (router, keyword heuristic): decides tool-vs-direct. Validated in
binary_router_experiment.py — FP 0/10, FN 1/20 (fixed to 0/20 by a train-derived rule).

Stage 2 (this script): for NEGATIVE queries the router sends the conversation to the
SAME BitNet model but with a NO-TOOLS system prompt (laziness protocol). Question:
does the model produce a useful direct answer without any tool JSON?

Protocol: BitNet b1.58 2B, llama-server-relu2.exe, exact HF token IDs, T=0.0,
rep_pen 1.15 (win 64), max 128 tokens. Input: the 10 negative cases of the
30-case challenge suite. A valid direct answer = no tool-call syntax (S1=False)
AND non-degenerate text (>= 5 chars, not a tool-list echo).
"""
import json
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"C:\ai_models")

import benchmark_postfix_symmetric as B
from evaluator_rigorous import parse_tool_envelope

HELD = r"C:\ai_models\heldout_test_suite_30.json"
OUT = r"C:\ai_models\paper_artifacts\stage2_no_tools_validation.json"
OUT_DESKTOP = r"C:\Users\Рам\Desktop\BitNet_Research_Reports\stage2_no_tools_validation.json"

SYSTEM_PROMPT_NO_TOOLS = """You are a helpful AI assistant. You have no external tools.
Answer the user directly and concisely in natural language.
Never output JSON or tool calls."""


def main():
    cases = [c for c in json.load(open(HELD, encoding="utf-8")) if not c["needs_tool"]]
    print(f"{len(cases)} negative cases loaded")

    tok = B.AutoTokenizer.from_pretrained(B.MODEL_BITNET_HF)
    proc, session = B.start_server(B.MODEL_BITNET_GGUF)
    print("Server ready.")
    results = []
    t0 = time.perf_counter()
    try:
        for case in cases:
            prompt_str, stop_tokens = B.format_prompt(case["messages"], SYSTEM_PROMPT_NO_TOOLS, "bitnet")
            ids = tok(prompt_str).input_ids
            r = session.post(f"http://127.0.0.1:{B.PORT}/completion", json={
                "prompt": ids, "n_predict": 128, "temperature": 0.0,
                "repeat_penalty": 1.15, "repeat_last_n": 64, "stop": stop_tokens,
            }, timeout=120).json()
            content = r.get("content", "").strip()
            s1, s2, _, err = parse_tool_envelope(content)
            echo = ("1." in content) or ("Available tools" in content) or len(content) < 5
            valid = (not s1) and (not echo)
            state = "VALID_DIRECT_ANSWER" if valid else ("TOOL_JSON_DESPITE_BAN" if s1 else "DEGENERATED_ECHO")
            print(f"Case {case['id']:2d}: {state:26s} | {repr(content[:60])}")
            results.append({
                "id": case["id"],
                "query": case["messages"][-1]["content"][:80],
                "response": content,
                "s1_tool_syntax": s1,
                "echo_or_degenerate": echo,
                "state": state,
            })
    finally:
        B.stop_process(proc)

    valid_n = sum(1 for r in results if r["state"] == "VALID_DIRECT_ANSWER")
    payload = {
        "title": "Stage-2 (no-tools prompt) validation: can BitNet answer directly when the router blocks tools?",
        "protocol": {
            "model": "BitNet-b1.58-2B-4T gguf i2_s, llama-server-relu2.exe, exact HF token IDs",
            "system_prompt": SYSTEM_PROMPT_NO_TOOLS,
            "cases": "10 negatives of the 30-case challenge suite",
            "valid_direct_answer": "no tool-call syntax AND non-degenerate text",
        },
        "summary": {
            "valid_direct_answers": f"{valid_n}/{len(results)} ({valid_n/len(results)*100:.1f}%)",
            "valid_n": valid_n,
            "total": len(results),
            "elapsed_s": round(time.perf_counter() - t0, 1),
        },
        "cases": results,
    }
    print("\n" + payload["summary"]["valid_direct_answers"])
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    try:
        import shutil
        shutil.copyfile(OUT, OUT_DESKTOP)
    except Exception as e:
        print(f"[warn] desktop copy failed: {e}")
    print(f"Saved -> {OUT}")


if __name__ == "__main__":
    main()
