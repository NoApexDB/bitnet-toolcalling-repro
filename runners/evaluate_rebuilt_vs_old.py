import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"C:\ai_models")

# Import evaluation logic directly from evaluator_rigorous
from evaluator_rigorous import (
    parse_tool_envelope,
    evaluate_static_heuristic_arguments,
    evaluate_execution_and_result_sandboxed
)

TEST_SUITE_PATH = r"C:\ai_models\test_suite_standardized.json"
OLD_CPP_PATH = r"C:\ai_models\paper_artifacts\symmetric_dense_vs_ternary.json"
REBUILT_CPP_PATH = r"C:\ai_models\paper_artifacts\rebuilt_cpp_benchmark_20cases.json"
PYTORCH_PATH = r"C:\ai_models\benchmark_results_pytorch_cpu.json"

def evaluate_run(run_name, detailed_results, cases):
    s1_cnt = 0
    s2_cnt = 0
    s3_cnt = 0
    s4_local_cnt = 0
    s5_local_cnt = 0
    local_total = 5  # 2 SQL (cases 11, 12) + 3 File (cases 9, 10, 14)
    non_tool_clean = 0

    per_case_eval = []

    for idx, case in enumerate(cases):
        cid = case["id"]
        exp_tool = case["expected_tool"]
        raw_text = detailed_results[idx]["response"] if "response" in detailed_results[idx] else detailed_results[idx]["model_response"]

        s1, s2, envelope, env_err = parse_tool_envelope(raw_text)

        s3 = False
        s3_detail = ""
        s4_status = "N/A"
        s5_status = "N/A"
        rule_str = "—"
        exec_detail = ""

        if case["needs_tool"]:
            if s1: s1_cnt += 1
            if s2: s2_cnt += 1

            if s2:
                s3, s3_detail = evaluate_static_heuristic_arguments(case, envelope)
            if s3: s3_cnt += 1

            if s3 and envelope:
                s4_res, s5_res, exec_detail = evaluate_execution_and_result_sandboxed(case, envelope)
                if s4_res is True:
                    s4_local_cnt += 1
                    s4_status = "[+]"
                elif s4_res is False:
                    s4_status = "[-]"
                else:
                    s4_status = str(s4_res)

                if s5_res is True:
                    s5_local_cnt += 1
                    s5_status = "[+]"
                elif s5_res is False:
                    s5_status = "[-]"
                else:
                    s5_status = str(s5_res)
            elif not s2:
                exec_detail = env_err or "Невалидная схема"
            elif not s3:
                exec_detail = s3_detail
        else:
            # Negative control
            if not s1:
                non_tool_clean += 1
                rule_str = "[OK] Нет вызова"
            else:
                if '{"tool": "none"' in raw_text:
                    rule_str = "[FAIL] Фиктивный tool:none"
                else:
                    rule_str = "[FAIL] Вызов инструмента"
            exec_detail = "Текст: " + repr(raw_text[:50])

        per_case_eval.append({
            "id": cid,
            "exp_tool": exp_tool,
            "s1": s1,
            "s2": s2,
            "s3": s3,
            "s4": s4_status,
            "s5": s5_status,
            "neg_rule": rule_str,
            "detail": exec_detail,
            "raw": raw_text
        })

    return {
        "run_name": run_name,
        "s1_cnt": s1_cnt,
        "s2_cnt": s2_cnt,
        "s3_cnt": s3_cnt,
        "s4_local_cnt": s4_local_cnt,
        "s5_local_cnt": s5_local_cnt,
        "non_tool_clean": non_tool_clean,
        "local_total": local_total,
        "per_case": per_case_eval
    }

def main():
    with open(TEST_SUITE_PATH, "r", encoding="utf-8") as f:
        cases = json.load(f)

    # 1. Rebuilt C++ BitNet
    with open(REBUILT_CPP_PATH, "r", encoding="utf-8") as f:
        rebuilt_cpp_data = json.load(f)
    rebuilt_eval = evaluate_run("BitNet C++ (Rebuilt, ReLU^2)", rebuilt_cpp_data["detailed_results"], cases)

    # 2. Old C++ BitNet
    with open(OLD_CPP_PATH, "r", encoding="utf-8") as f:
        old_cpp_data = json.load(f)
    old_eval = evaluate_run("BitNet C++ (Old Broken, SwiGLU)", old_cpp_data["bitnet_2b_i2_s_base"]["detailed_results"], cases)
    qwen15_eval = evaluate_run("Qwen 2.5 1.5B (Q4_K_M)", old_cpp_data["qwen2.5_1.5b_q4_k_m"]["detailed_results"], cases)
    qwen3_eval = evaluate_run("Qwen 2.5 3B (Q4_K_M)", old_cpp_data["qwen2.5_3b_q4_k_m"]["detailed_results"], cases)

    # 3. PyTorch BitNet (if available)
    pt_eval = None
    if os.path.exists(PYTORCH_PATH):
        with open(PYTORCH_PATH, "r", encoding="utf-8") as f:
            pt_data = json.load(f)
        pt_eval = evaluate_run("BitNet PyTorch CPU (Reference)", pt_data["detailed_results"], cases)

    runs = [old_eval, rebuilt_eval]
    if pt_eval:
        runs.append(pt_eval)
    runs.extend([qwen15_eval, qwen3_eval])

    print("\n" + "=" * 110)
    print("5-STAGE EVALUATION COMPARISON ACROSS RUNS")
    print("=" * 110)
    print(f"{'Run / Model':<35} | {'S1 (Tool)':<12} | {'S2 (Schema)':<12} | {'S3 (Heuristic)':<15} | {'S4 (Exec)':<10} | {'S5 (Result)':<10} | {'Non-Tool':<10}")
    print("-" * 110)
    for r in runs:
        s1_str = f"{r['s1_cnt']}/15 ({r['s1_cnt']/15*100:.1f}%)"
        s2_str = f"{r['s2_cnt']}/15 ({r['s2_cnt']/15*100:.1f}%)"
        s3_str = f"{r['s3_cnt']}/15 ({r['s3_cnt']/15*100:.1f}%)"
        s4_str = f"{r['s4_local_cnt']}/{r['local_total']} ({r['s4_local_cnt']/r['local_total']*100:.1f}%)"
        s5_str = f"{r['s5_local_cnt']}/{r['local_total']} ({r['s5_local_cnt']/r['local_total']*100:.1f}%)"
        neg_str = f"{r['non_tool_clean']}/5 ({r['non_tool_clean']/5*100:.1f}%)"
        print(f"{r['run_name']:<35} | {s1_str:<12} | {s2_str:<12} | {s3_str:<15} | {s4_str:<10} | {s5_str:<10} | {neg_str:<10}")

    print("\n" + "=" * 110)
    print("PER-CASE COMPARISON (Old Broken C++ vs Rebuilt C++ vs PyTorch Reference)")
    print("=" * 110)
    print(f"{'ID':<3} | {'Exp Tool':<20} | {'Old C++ (SwiGLU)':<25} | {'Rebuilt C++ (ReLU^2)':<25} | {'PyTorch Reference':<25}")
    print("-" * 110)
    for idx in range(len(cases)):
        cid = cases[idx]["id"]
        exp_t = cases[idx]["expected_tool"]
        old_c = old_eval["per_case"][idx]
        reb_c = rebuilt_eval["per_case"][idx]
        pt_c = pt_eval["per_case"][idx] if pt_eval else None

        old_desc = f"S1:{'+' if old_c['s1'] else '-'} S2:{'+' if old_c['s2'] else '-'} S3:{'+' if old_c['s3'] else '-'}"
        reb_desc = f"S1:{'+' if reb_c['s1'] else '-'} S2:{'+' if reb_c['s2'] else '-'} S3:{'+' if reb_c['s3'] else '-'}"
        pt_desc = f"S1:{'+' if pt_c['s1'] else '-'} S2:{'+' if pt_c['s2'] else '-'} S3:{'+' if pt_c['s3'] else '-'}" if pt_c else "N/A"

        exp_t_str = str(exp_t) if exp_t is not None else "[Non-Tool]"
        print(f"{cid:02d} | {exp_t_str:<20} | {old_desc:<25} | {reb_desc:<25} | {pt_desc:<25}")

    out_file = r"C:\ai_models\paper_artifacts\REBUILT_CPP_VS_OLD_AUDIT.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "rebuilt_eval": rebuilt_eval,
            "old_eval": old_eval,
            "pt_eval": pt_eval,
            "qwen15_eval": qwen15_eval,
            "qwen3_eval": qwen3_eval
        }, f, indent=2, ensure_ascii=False)
    print(f"\nSaved full comparative evaluation to: {out_file}")

if __name__ == "__main__":
    main()
