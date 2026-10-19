# -*- coding: utf-8 -*-
"""
PHASE 2.4: EXTERNAL BLIND HOLDOUT — Berkeley Function Calling Leaderboard (BFCL v3).

Genuinely external set: none of these prompts/tools were seen during evaluator tuning,
router building, or any prior stage of the project. Sample (deterministic, first N):
  * simple (400 -> first 40): single function call, ground truth = name + acceptable
    parameter values;
  * irrelevance (240 -> first 20): function doc provided but NO call should be made.

Models: BitNet 2B I2_S (corrected runtime: ReLU2 build + exact HF token IDs) vs
Qwen2.5-3B Q4_K_M (its canonical string path). Same engine, same decoding params.
Prompt: neutral English system prompt with JSON-schema function docs; expected output
envelope {"name": ..., "parameters": {...}}.

Strict matcher: function name == GT name; every GT param value in its acceptable list
(numeric equality, case-insensitive strings); no extra params beyond GT's set.
"""
import json
import re
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"C:\ai_models")

import benchmark_postfix_symmetric as B

SCRATCH = r"C:\ai_models\scratch"
OUT = r"C:\ai_models\paper_artifacts\bfcl_external_holdout_results.json"
OUT_DESKTOP = r"C:\Users\Рам\Desktop\BitNet_Research_Reports\bfcl_external_holdout_results.json"

N_SIMPLE = 40
N_IRR = 20

SYSTEM_PROMPT = """You are an assistant with access to the following functions:

{funcs}

When a function call is needed, respond ONLY with a JSON object in the format:
{{"name": "function_name", "parameters": {{...}}}}
If no function is relevant to the user's request, answer directly in plain text without any JSON."""


def render_funcs(func_docs):
    return json.dumps(func_docs, ensure_ascii=False)


def last_user_content(entry):
    turn = entry["question"][0]
    for m in reversed(turn):
        if m["role"] == "user":
            return m["content"]
    return turn[-1]["content"]


def parse_call(text, allow_prose_fallback=True):
    """Extract {"name":..., "parameters": {...}} from the output (or None).
    Prose fallback (regex over explanatory text) is DISABLED for negative scoring:
    mentioning 'cos(theta)' or a function name in an explanation is NOT a call."""
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    cands = []
    if m:
        cands.append(m.group(1))
    idx = text.find("{")
    while idx != -1:
        cands.append(text[idx:])
        idx = text.find("{", idx + 1)
    for cand in cands:
        try:
            obj, _ = json.JSONDecoder().raw_decode(cand)
            if isinstance(obj, dict) and ("name" in obj or "tool" in obj):
                return obj, "json"
        except Exception:
            pass
    if allow_prose_fallback:
        m = re.search(r"([A-Za-z_][A-Za-z0-9_]*)\s*\((.*)\)", text, re.DOTALL)
        if m:
            return {"name": m.group(1), "parameters": {"__raw__": m.group(2)[:200]}}, "prose"
    return None, None


def values_match(model_v, acceptable):
    for acc in acceptable:
        if isinstance(acc, (int, float)) and isinstance(model_v, (int, float)):
            if abs(float(model_v) - float(acc)) < 1e-6:
                return True
        elif isinstance(acc, str) and isinstance(model_v, str):
            if model_v.strip().lower() == acc.strip().lower():
                return True
        elif str(model_v).strip().lower() == str(acc).strip().lower():
            return True
    return False


def eval_simple(model_call, gt_list, required_params=None):
    """gt_list: [{"func": {"param": [acceptable...]}}] — any alternative counts.
    Returns (strict_ok, lenient_ok, detail). Lenient: an omitted param is acceptable
    if it is NOT in the function's 'required' list (BFCL default-param convention)."""
    if model_call is None:
        return False, False, "no call parsed"
    name = model_call.get("name") or model_call.get("tool")
    params = model_call.get("parameters") or model_call.get("arguments") or {}
    for alt in gt_list:
        for gt_name, gt_params in alt.items():
            if gt_name != name:
                continue
            if "__raw__" in params:
                return False, False, "raw call syntax, not JSON"
            extra = set(params.keys()) - set(gt_params.keys())
            if extra:
                return False, False, f"extra params {sorted(extra)}"
            strict_ok = True
            for p, acceptable in gt_params.items():
                optional = p not in (required_params or [])
                if p not in params:
                    if optional:
                        strict_ok = False  # omitted optional: lenient passes, strict fails
                        continue
                    return False, False, f"missing required param {p}"
                if not values_match(params[p], acceptable):
                    if optional and "" in [str(a) for a in acceptable]:
                        strict_ok = False  # defaultable value: lenient passes, strict fails
                        continue
                    return False, False, f"param {p}: {params[p]!r} not in {acceptable!r}"
            return strict_ok, True, "match"
    return False, False, f"name {name!r} not in GT"


def eval_irrelevance(text, model_call):
    if model_call is None:
        # any residual call-like syntax?
        if re.search(r"```json|\{\"name\"|'\w+'\s*\(", text[:400]):
            return False, "call-like syntax despite no parse"
        return True, "no call"
    return False, f"false call to {(model_call.get('name') or '?')!r}"


def run_model(label, model_path, model_type, input_mode, entries, answers, tok):
    proc, session = B.start_server(model_path)
    print(f"Server ready for {label}")
    results = []
    t0 = time.perf_counter()
    try:
        for e in entries:
            funcs = render_funcs(e["function"])
            sys_prompt = SYSTEM_PROMPT.format(funcs=funcs)
            msgs = e["question"][0]
            prompt_str, stop_tokens = B.format_prompt(msgs, sys_prompt, model_type)
            if input_mode == "tokens":
                payload_prompt = tok(prompt_str).input_ids
            else:
                payload_prompt = prompt_str
            r = session.post(f"http://127.0.0.1:{B.PORT}/completion", json={
                "prompt": payload_prompt, "n_predict": 128, "temperature": 0.0,
                "repeat_penalty": 1.15, "repeat_last_n": 64, "stop": stop_tokens,
            }, timeout=120).json()
            content = r.get("content", "").strip()

            if e["id"].startswith("simple"):
                call, syntax = parse_call(content, allow_prose_fallback=True)
                req = e["function"][0].get("parameters", {}).get("required", []) or []
                s_ok, l_ok, detail = eval_simple(call, answers.get(e["id"], []), required_params=req)
                cat = "simple"
            else:
                # JSON-envelope criterion only: prose mentions of functions are NOT calls
                call, syntax = parse_call(content, allow_prose_fallback=False)
                ok, detail = eval_irrelevance(content, call)
                s_ok = l_ok = ok
                cat = "irrelevance"
            results.append({"id": e["id"], "category": cat, "ok": s_ok, "ok_lenient": l_ok,
                            "detail": detail, "syntax": syntax, "output": content})
            print(f"[{label}] {e['id']:16s}: {'OK ' if s_ok else ('L~ ' if l_ok else 'FAIL')} | {detail[:60]} | {repr(content[:45])}")
    finally:
        B.stop_process(proc)

    simple = [r for r in results if r["category"] == "simple"]
    irr = [r for r in results if r["category"] == "irrelevance"]
    summary = {
        "label": label,
        "simple_strict": f"{sum(1 for r in simple if r['ok'])}/{len(simple)}",
        "simple_lenient_optional": f"{sum(1 for r in simple if r['ok_lenient'])}/{len(simple)}",
        "irrelevance_correct_refusal_json_only": f"{sum(1 for r in irr if r['ok'])}/{len(irr)}",
        "false_positive_rate_pct_json_only": round(100 * sum(1 for r in irr if not r["ok"]) / max(len(irr), 1), 1),
        "elapsed_s": round(time.perf_counter() - t0, 1),
    }
    return summary, results


def main():
    q = [json.loads(l) for l in open(SCRATCH + r"\bfcl_simple_q.json", encoding="utf-8")]
    irr = [json.loads(l) for l in open(SCRATCH + r"\bfcl_irrelevance.json", encoding="utf-8")]
    a = [json.loads(l) for l in open(SCRATCH + r"\bfcl_simple_ans.json", encoding="utf-8")]
    answers = {x["id"]: x["ground_truth"] for x in a}

    simple_sample = q[:N_SIMPLE]
    irr_sample = irr[:N_IRR]
    entries = simple_sample + irr_sample
    print(f"BFCL external holdout: {len(simple_sample)} simple + {len(irr_sample)} irrelevance")

    tok = B.AutoTokenizer.from_pretrained(B.MODEL_BITNET_HF)

    s1, r1 = run_model("bitnet_2b_i2_s_relu2_hf", B.MODEL_BITNET_GGUF, "bitnet", "tokens", entries, answers, tok)
    s2, r2 = run_model("qwen2.5_3b_q4_k_m", B.MODEL_QWEN3, "qwen", "string", entries, answers, tok)

    payload = {
        "title": "EXTERNAL BLIND HOLDOUT: BFCL v3 (simple + irrelevance subsets)",
        "source": "huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard "
                  "(BFCL_v3_simple.json first 40, BFCL_v3_irrelevance.json first 20; deterministic sample)",
        "blindness": "neither prompts nor tool docs nor ground truth were seen at any stage of the project",
        "protocol": {
            "engine": "llama-server-relu2.exe, 4 threads, T=0.0, rep_pen 1.15/64, 128 max tokens",
            "bitnet_input": "exact HF token IDs",
            "matcher": "strict: name match; every GT param in acceptable values; no extra params",
        },
        "bitnet_2b_fixed_runtime": {"summary": s1, "cases": r1},
        "qwen2.5_3b": {"summary": s2, "cases": r2},
    }
    print("\nbitnet:", json.dumps(s1, ensure_ascii=False))
    print("qwen3b:", json.dumps(s2, ensure_ascii=False))
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
