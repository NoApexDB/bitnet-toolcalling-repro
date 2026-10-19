# -*- coding: utf-8 -*-
"""
Review remark #4: mini-factorial on the NEGATIVE-control prompt factors.

The internal prompt (0/10 refusals) and the BFCL prompt (19/20 refusals) differ in
at least three things at once. This experiment isolates TWO of them on the SAME
10 negative cases of the heldout 30-case suite (BitNet, corrected runtime, T=0):

  Factor A (tool menu):  5 always-available tools  vs  1 explicitly irrelevant tool
  Factor B (framing):    aggressive (MUST ONLY JSON, no extra text)  vs  soft
                         (JSON allowed; if no tool relevant -> plain-text answer)

Cells:
  A1B1 = 5 tools + aggressive   (≈ internal prompt; expected ~0/10)
  A1B2 = 5 tools + soft
  A2B1 = 1 tool  + aggressive
  A2B2 = 1 tool  + soft         (≈ BFCL-style)
Output: paper_artifacts/prompt_factorial_negatives.json
"""
import json
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"C:\ai_models")

import benchmark_postfix_symmetric as B

OUT = r"C:\ai_models\paper_artifacts\prompt_factorial_negatives.json"
OUT_DESKTOP = r"C:\Users\Рам\Desktop\BitNet_Research_Reports\prompt_factorial_negatives.json"

TOOLS_5 = """Available tools:
1. get_current_weather(location: str, unit: str = "celsius") - Retrieve current weather data for a specific city.
2. web_search(query: str, max_results: int = 5) - Search the web for up-to-date information.
3. execute_python(code: str) - Execute Python script in a secure sandbox and return standard output.
4. file_read(path: str) - Read text content of a local file.
5. query_database(sql: str) - Execute a read-only SQL query against the database."""

TOOLS_1 = """Available tools:
1. get_current_weather(location: str, unit: str = "celsius") - Retrieve current weather data for a specific city."""

PROMPT_AGGRESSIVE = """You are a helpful AI assistant with access to tools.
When you need to use a tool to assist the user, you MUST respond ONLY with a JSON object in the following format:
```json
{"tool": "tool_name", "parameters": {"param1": "value1"}}
```
Do not include any extra thoughts or text before or after the JSON when invoking a tool.
If no tool is required, answer the user directly and concisely in natural language.

{tools}"""

PROMPT_SOFT = """You are a helpful AI assistant with access to tools.
When you need to use a tool to assist the user, you may respond with a JSON object in the following format:
```json
{"tool": "tool_name", "parameters": {"param1": "value1"}}
```
If no tool is relevant to the user's request, answer the user directly in plain text without any JSON.

{tools}"""


def valid_refusal(content):
    """JSON-envelope criterion: a refusal = no tool envelope AND non-degenerate text."""
    text = content.strip()
    has_envelope = ('"tool"' in text and "{" in text) or ("'tool'" in text)
    echo = ("1." in text) or ("Available tools" in text) or len(text) < 5
    return (not has_envelope) and (not echo)


def main():
    cases = [c for c in json.load(open(r"C:\ai_models\heldout_test_suite_30.json", encoding="utf-8"))
             if not c["needs_tool"]]
    print(f"{len(cases)} negative cases")

    tok = B.AutoTokenizer.from_pretrained(B.MODEL_BITNET_HF)
    proc, session = B.start_server(B.MODEL_BITNET_GGUF)
    print("Server ready.")

    cells = {
        "A1B1_5tools_aggressive": PROMPT_AGGRESSIVE.replace("{tools}", TOOLS_5),
        "A1B2_5tools_soft": PROMPT_SOFT.replace("{tools}", TOOLS_5),
        "A2B1_1tool_aggressive": PROMPT_AGGRESSIVE.replace("{tools}", TOOLS_1),
        "A2B2_1tool_soft": PROMPT_SOFT.replace("{tools}", TOOLS_1),
    }

    results = {}
    try:
        for cell_name, sys_prompt in cells.items():
            rows = []
            refusals = 0
            for case in cases:
                prompt_str, stop_tokens = B.format_prompt(case["messages"], sys_prompt, "bitnet")
                ids = tok(prompt_str).input_ids
                r = session.post(f"http://127.0.0.1:{B.PORT}/completion", json={
                    "prompt": ids, "n_predict": 128, "temperature": 0.0,
                    "repeat_penalty": 1.15, "repeat_last_n": 64, "stop": stop_tokens,
                }, timeout=120).json()
                content = r.get("content", "").strip()
                ok = valid_refusal(content)
                refusals += ok
                rows.append({"id": case["id"], "refusal": ok, "output": content[:200]})
                print(f"[{cell_name}] {case['id']:3d}: {'REFUSAL' if ok else 'TOOL-CALL'} | {repr(content[:60])}")
            results[cell_name] = {
                "valid_refusals": f"{refusals}/{len(cases)}",
                "refusals_n": refusals, "cases": rows,
            }
            print(f"  -> {cell_name}: {refusals}/{len(cases)}\n")
    finally:
        B.stop_process(proc)

    def n(cell):
        return results[cell]["refusals_n"]

    payload = {
        "title": "Mini-factorial on negative-control prompt factors (tool menu x framing)",
        "protocol": {
            "model": "BitNet-b1.58-2B-4T i2_s, llama-server-relu2.exe, exact HF token IDs, T=0",
            "cases": "10 negatives of the heldout 30-case suite",
            "refusal_criterion": "no JSON tool envelope AND non-degenerate text",
            "factor_A": "tool menu: 5 tools vs 1 (weather, irrelevant to all 10 queries)",
            "factor_B": "framing: MUST-ONLY-JSON vs may-JSON + plain-text-affordance",
        },
        "cells": results,
        "effects": {
            "framing_effect_5tools (B2-B1)": n("A1B2_5tools_soft") - n("A1B1_5tools_aggressive"),
            "framing_effect_1tool (B2-B1)": n("A2B2_1tool_soft") - n("A2B1_1tool_aggressive"),
            "menu_effect_aggressive (A2-A1)": n("A2B1_1tool_aggressive") - n("A1B1_5tools_aggressive"),
            "menu_effect_soft (A2-A1)": n("A2B2_1tool_soft") - n("A1B2_5tools_soft"),
        },
    }
    print("EFFECTS:", json.dumps(payload["effects"], ensure_ascii=False))
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
