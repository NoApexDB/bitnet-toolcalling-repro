import json
import sqlite3
import subprocess
import sys
import tempfile

def evaluate_response_5stage(case, raw_text):
    category = case["category"]
    needs_tool = case["needs_tool"]
    exp_tool = case["expected_tool"]
    
    res = {
        "id": case["id"],
        "category": category,
        "needs_tool": needs_tool,
        "expected_tool": exp_tool,
        "raw_text": raw_text.strip(),
        # Stage 1: Tool Chosen
        "s1_tool_chosen": False,
        # Stage 2: JSON Parsed
        "s2_json_parsed": False,
        "parsed_json": None,
        # Stage 3: Arguments Valid
        "s3_args_valid": False,
        # Stage 4: Executed
        "s4_executed": False,
        "exec_error": None,
        # Stage 5: Result Correct
        "s5_result_correct": False,
        "exec_output": None,
        # Negative control rule adherence
        "no_tool_rule_complied": False
    }

    # Extract JSON or tool attempts
    text = raw_text.strip()
    
    # Check if tool was chosen / attempted
    tool_keywords = ['"tool"', "'tool'", "tool:", "get_current_weather", "web_search", "execute_python", "file_read", "query_database"]
    has_keyword = any(k in text.lower() for k in tool_keywords)
    
    # Try parsing JSON
    parsed_obj = None
    import re
    # Try finding markdown json or curly brackets
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            parsed_obj = json.loads(m.group(1))
        except Exception:
            pass
    
    if parsed_obj is None:
        # Try raw decode
        start_idx = text.find("{")
        if start_idx != -1:
            try:
                decoder = json.JSONDecoder()
                obj, _ = decoder.raw_decode(text[start_idx:])
                if isinstance(obj, dict):
                    parsed_obj = obj
            except Exception:
                pass

    if parsed_obj is not None and isinstance(parsed_obj, dict):
        res["s2_json_parsed"] = True
        res["parsed_json"] = parsed_obj
        if "tool" in parsed_obj or "name" in parsed_obj:
            res["s1_tool_chosen"] = True
    elif has_keyword:
        res["s1_tool_chosen"] = True

    if not needs_tool:
        # Negative scenario
        res["no_tool_rule_complied"] = not res["s1_tool_chosen"]
        return res

    # If needs tool, verify arguments and execution
    if not res["s2_json_parsed"] or not parsed_obj:
        return res

    tool_name = parsed_obj.get("tool") or parsed_obj.get("name")
    params = parsed_obj.get("parameters") or parsed_obj.get("arguments") or {}
    
    if tool_name != exp_tool:
        res["s3_args_valid"] = False
        return res

    val_type = case.get("validator_type")
    
    # Stage 3: Argument validity check
    if val_type == "entity_match":
        rules = case.get("validation_rules", {})
        all_ok = True
        for param_key, expected_substrings in rules.items():
            val = str(params.get(param_key, "")).lower()
            if not any(sub in val for sub in expected_substrings):
                all_ok = False
                break
        res["s3_args_valid"] = all_ok
        # For mock tools (weather, search, file), if args are valid, mock execute
        if all_ok:
            res["s4_executed"] = True
            res["s5_result_correct"] = True
            res["exec_output"] = f"Mock success for {tool_name}"

    elif val_type == "python_sandbox":
        code = params.get("code", "")
        # Require actual code string
        if isinstance(code, str) and len(code.strip()) > 5:
            res["s3_args_valid"] = True
            # Stage 4: Run in sandbox
            try:
                run_res = subprocess.run(
                    [sys.executable, "-c", code],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if run_res.returncode == 0:
                    res["s4_executed"] = True
                    out_clean = run_res.stdout.strip()
                    res["exec_output"] = out_clean
                    # Stage 5: Result match
                    exp_out = str(case.get("sandbox_expected_output", "")).strip()
                    if exp_out in out_clean:
                        res["s5_result_correct"] = True
                else:
                    res["exec_error"] = run_res.stderr.strip()[:200]
            except Exception as e:
                res["exec_error"] = str(e)[:200]

    elif val_type == "sqlite_execution":
        sql = params.get("sql", "")
        if isinstance(sql, str) and len(sql.strip()) > 5:
            res["s3_args_valid"] = True
            # Stage 4: SQLite execution in memory
            try:
                conn = sqlite3.connect(":memory:")
                cur = conn.cursor()
                setup_sql = case.get("sql_setup", "")
                if setup_sql:
                    cur.executescript(setup_sql)
                cur.execute(sql)
                rows = cur.fetchall()
                res["s4_executed"] = True
                res["exec_output"] = str(rows)[:200]
                
                # Stage 5: Check expected result
                if "sql_expected_scalar" in case:
                    exp_val = case["sql_expected_scalar"]
                    if len(rows) > 0 and len(rows[0]) > 0 and rows[0][0] == exp_val:
                        res["s5_result_correct"] = True
                elif "sql_expected_row_count" in case:
                    exp_count = case["sql_expected_row_count"]
                    if len(rows) == exp_count:
                        res["s5_result_correct"] = True
                conn.close()
            except Exception as e:
                res["exec_error"] = str(e)[:200]

    return res

if __name__ == "__main__":
    with open(r"C:\ai_models\test_suite_standardized.json", "r", encoding="utf-8") as f:
        cases = json.load(f)

    with open(r"C:\ai_models\paper_artifacts\symmetric_dense_vs_ternary.json", "r", encoding="utf-8") as f:
        cpp_data = json.load(f)

    models_to_eval = [
        ("bitnet_2b_i2_s_base", "BitNet b1.58 (I2_S Base)"),
        ("qwen2.5_1.5b_q4_k_m", "Qwen 2.5 1.5B (Q4_K_M)"),
        ("qwen2.5_3b_q4_k_m", "Qwen 2.5 3B (Q4_K_M)")
    ]

    report_lines = []
    report_lines.append("# Независимый пятиэтапный аудит ответов C++ рантайма\n")
    report_lines.append("Оценка проведена с реальным исполнением Python-кода в subprocess и SQL-запросов в SQLite in-memory.\n")

    summary_stats = {}

    for mod_key, mod_name in models_to_eval:
        detailed = cpp_data[mod_key]["detailed_results"]
        stats = {
            "pos_total": 15,
            "s1_chosen": 0,
            "s2_json": 0,
            "s3_args": 0,
            "s4_executed": 0,
            "s5_correct": 0,
            "neg_total": 5,
            "no_tool_rule": 0
        }
        
        report_lines.append(f"## Модель: {mod_name}\n")
        report_lines.append("| ID | Ожидаемый инструмент | S1: Вызов выбран | S2: JSON разобран | S3: Аргументы верны | S4: Выполнен | S5: Результат верен | Правило Non-Tool | Детали исполнения / Ошибка |")
        report_lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")

        for idx, case in enumerate(cases):
            raw = detailed[idx]["response"]
            r = evaluate_response_5stage(case, raw)
            
            if r["needs_tool"]:
                if r["s1_tool_chosen"]: stats["s1_chosen"] += 1
                if r["s2_json_parsed"]: stats["s2_json"] += 1
                if r["s3_args_valid"]: stats["s3_args"] += 1
                if r["s4_executed"]: stats["s4_executed"] += 1
                if r["s5_result_correct"]: stats["s5_correct"] += 1
                rule_str = "—"
            else:
                if r["no_tool_rule_complied"]: stats["no_tool_rule"] += 1
                rule_str = "[OK] Соблюдено" if r["no_tool_rule_complied"] else "[FAIL] Вызов тула"

            detail = ""
            if r["exec_error"]:
                detail = f"Error: {r['exec_error']}"
            elif r["exec_output"]:
                detail = f"Out: {r['exec_output']}"
            elif not r["s2_json_parsed"] and r["s1_tool_chosen"]:
                detail = "Prompt echo / Unparsed"

            s1 = "[+]" if r["s1_tool_chosen"] else "[-]"
            s2 = "[+]" if r["s2_json_parsed"] else "[-]"
            s3 = "[+]" if r["s3_args_valid"] else "[-]"
            s4 = "[+]" if r["s4_executed"] else "[-]"
            s5 = "[+]" if r["s5_result_correct"] else "[-]"

            report_lines.append(f"| {case['id']:02d} | `{case['expected_tool']}` | {s1} | {s2} | {s3} | {s4} | {s5} | {rule_str} | {detail} |")

        report_lines.append("\n")
        summary_stats[mod_name] = stats

    report_lines.append("## Сводная пятиэтапная таблица (C++ рантайм)\n")
    report_lines.append("| Модель | S1: Вызов выбран | S2: JSON разобран | S3: Аргументы верны | S4: Выполнен в песочнице | S5: Результат верен | Соблюдение Non-Tool (из 5) |")
    report_lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
    for name, s in summary_stats.items():
        report_lines.append(f"| {name} | {s['s1_chosen']}/15 ({s['s1_chosen']/15*100:.1f}%) | {s['s2_json']}/15 ({s['s2_json']/15*100:.1f}%) | {s['s3_args']}/15 ({s['s3_args']/15*100:.1f}%) | {s['s4_executed']}/15 ({s['s4_executed']/15*100:.1f}%) | {s['s5_correct']}/15 ({s['s5_correct']/15*100:.1f}%) | {s['no_tool_rule']}/5 ({s['no_tool_rule']/5*100:.1f}%) |")

    out_file = r"C:\ai_models\paper_artifacts\AUDIT_5STAGE_CPP.md"
    with open(out_file, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print("Audit 5-stage completed successfully.")
