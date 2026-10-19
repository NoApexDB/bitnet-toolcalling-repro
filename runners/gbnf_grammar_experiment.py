# -*- coding: utf-8 -*-
"""
PHASE 2.2: GBNF grammar-constrained decoding experiment on BitNet 2B (fixed runtime).

Question 1 (syntax): does a strict JSON-envelope grammar lift S2 to 100% on positives,
and what remains broken in S3 (semantic residue)?
Question 2 (selectivity): with an extended grammar offering {"tool":"none"}, does the
model CHOOSE "none" on negative queries, or does it still pick a hallucinated tool?
(Grammar constrains syntax; the choice remains the model's — this separates the two.)

Runtime: llama-server-relu2.exe + exact HF token IDs, canonical 20-case suite,
SYSTEM_PROMPT_BASE (with 'AI'), T=0.0, rep_pen 1.15/64, n_predict 128.
"""
import json
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"C:\ai_models")

import benchmark_postfix_symmetric as B
from evaluator_rigorous import (
    parse_tool_envelope,
    evaluate_static_heuristic_arguments,
    evaluate_execution_and_result_sandboxed,
)

OUT = r"C:\ai_models\paper_artifacts\gbnf_grammar_benchmark.json"
OUT_DESKTOP = r"C:\Users\Рам\Desktop\BitNet_Research_Reports\gbnf_grammar_benchmark.json"

GRAMMAR_STRICT = r"""
root ::= "{" ws "\"tool\"" ws ":" ws tool ws "," ws "\"parameters\"" ws ":" ws obj ws "}"
tool ::= "\"get_current_weather\"" | "\"web_search\"" | "\"execute_python\"" | "\"file_read\"" | "\"query_database\""
obj ::= "{" ws (str ws ":" ws val (ws "," ws str ws ":" ws val)*)? ws "}"
val ::= str | num | obj | arr
arr ::= "[" ws (val (ws "," ws val)*)? ws "]"
str ::= "\"" ( [^"\\\x7f\x00-\x1f] | "\\" (["\\bfnrt] | "u" [0-9a-fA-F]{4}) )* "\""
num ::= ("-"? ([0-9] | [1-9] [0-9]{0,15})) ("." [0-9]+)? ([eE] [-+]? [0-9] [0-9] [0-9]{0,14})?
ws ::= [ \t\n\r]*
""".strip()

GRAMMAR_WITH_NONE = GRAMMAR_STRICT.replace(
    '"\\"query_database\\""',
    '"\\"query_database\\"" | "\\"none\\""',
)


def run_cases(cases, grammar, tok, session, label):
    results = []
    for case in cases:
        prompt_str, stop_tokens = B.format_prompt(case["messages"], B.SYSTEM_PROMPT_BASE, "bitnet")
        ids = tok(prompt_str).input_ids
        payload = {
            "prompt": ids, "n_predict": 128, "temperature": 0.0,
            "repeat_penalty": 1.15, "repeat_last_n": 64, "stop": stop_tokens,
        }
        if grammar:
            payload["grammar"] = grammar
        r = session.post(f"http://127.0.0.1:{B.PORT}/completion", json=payload, timeout=120).json()
        content = r.get("content", "").strip()

        s1, s2, envelope, env_err = parse_tool_envelope(content)
        s3 = False
        s4_status = s5_status = "N/A"
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
                detail = s3_detail if s3 is False else ""
            picked_none = (envelope or {}).get("tool") == "none"
        else:
            picked_none = ('"tool": "none"' in content) or ('"tool":"none"' in content)
            detail = "Текст: " + repr(content[:60])

        results.append({
            "id": case["id"], "category": case["category"],
            "needs_tool": case["needs_tool"],
            "expected_tool": case.get("expected_tool"),
            "response": content,
            "s1": s1, "s2": s2, "s3": s3,
            "s4": s4_status, "s5": s5_status,
            "picked_none": picked_none,
            "detail": detail,
        })
        print(f"[{label}] Case {case['id']:2d}: S1={int(s1)} S2={int(s2)} S3={int(s3)} "
              f"none={int(picked_none)} | {repr(content[:55])}")
    return results


def summarize(results):
    pos = [r for r in results if r["needs_tool"]]
    neg = [r for r in results if not r["needs_tool"]]
    return {
        "s2_valid_schema": f"{sum(1 for r in pos if r['s2'])}/{len(pos)}",
        "s3_heuristic": f"{sum(1 for r in pos if r['s3'])}/{len(pos)}",
        "s4_local_exec": f"{sum(1 for r in pos if r['s4'] == '[+]')}/5",
        "s5_local_result": f"{sum(1 for r in pos if r['s5'] == '[+]')}/5",
        "negatives_picked_none": f"{sum(1 for r in neg if r['picked_none'])}/{len(neg)}",
        "positives_picked_none": f"{sum(1 for r in pos if r['picked_none'])}/{len(pos)}",
    }


def main():
    cases = json.load(open(B.TEST_SUITE_PATH, encoding="utf-8"))
    positives = [c for c in cases if c["needs_tool"]]

    tok = B.AutoTokenizer.from_pretrained(B.MODEL_BITNET_HF)
    proc, session = B.start_server(B.MODEL_BITNET_GGUF)
    print("Server ready.")

    try:
        # smoke test: does the server accept the "grammar" field at all?
        smoke_ids = tok(B.format_prompt(positives[0]["messages"], B.SYSTEM_PROMPT_BASE, "bitnet")[0]).input_ids
        try:
            sr = session.post(f"http://127.0.0.1:{B.PORT}/completion", json={
                "prompt": smoke_ids, "n_predict": 24, "temperature": 0.0,
                "grammar": GRAMMAR_STRICT,
            }, timeout=60).json()
            smoke_ok = "content" in sr and "error" not in sr
            print(f"Grammar support smoke test: {'OK' if smoke_ok else 'FAILED'} | {repr(str(sr.get('content',''))[:60])}")
            smoke_content = str(sr.get("content", ""))
        except Exception as e:
            print(f"Grammar smoke test threw: {e}")
            smoke_ok = False
            smoke_content = ""

        if not smoke_ok:
            print("Server does not support request-level 'grammar'. Aborting grammar runs.")
            return

        # A) strict envelope grammar on the 15 positives
        print("\n=== A) STRICT envelope grammar, 15 positives ===")
        res_a = run_cases(positives, GRAMMAR_STRICT, tok, session, "A")
        # B) grammar with "none" option on all 20 cases
        print("\n=== B) Extended grammar with tool:'none', all 20 cases ===")
        res_b = run_cases(cases, GRAMMAR_WITH_NONE, tok, session, "B")
    finally:
        B.stop_process(proc)

    payload = {
        "title": "GBNF grammar-constrained decoding on BitNet 2B (fixed runtime, canonical 20-case suite)",
        "grammar_strict": GRAMMAR_STRICT,
        "grammar_with_none": GRAMMAR_WITH_NONE,
        "smoke": {"supported": smoke_ok, "sample": smoke_content[:120]},
        "A_strict_envelope_positives": {"summary": summarize(res_a), "cases": res_a},
        "B_with_none_all_cases": {"summary": summarize(res_b), "cases": res_b},
        "interpretation_notes": [
            "Condition A isolates syntax: grammar forces a valid envelope -> S2 should be 15/15;",
            "whatever remains failing in S3 is semantic (wrong entities/args), not syntactic;",
            "Condition B probes selectivity: syntax allows 'none', the CHOICE is still the model's;",
            "grammar-constrained decoding therefore cannot substitute the router (stage 1).",
        ],
    }
    print("\nA) strict:", json.dumps(summarize(res_a), ensure_ascii=False))
    print("B) none-opt:", json.dumps(summarize(res_b), ensure_ascii=False))

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
