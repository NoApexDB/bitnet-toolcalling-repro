import ast
import json
import os
import re
import sqlite3
import subprocess
import sys

FORBIDDEN_MODULES = {
    "os", "sys", "subprocess", "shutil", "socket", "ctypes", 
    "requests", "urllib", "pathlib", "builtins", "importlib",
    "pty", "platform", "multiprocessing", "threading"
}

FORBIDDEN_CALLS = {
    "eval", "exec", "open", "__import__", "globals", "locals", 
    "getattr", "setattr", "delattr", "compile"
}

def check_python_security(code_str):
    try:
        tree = ast.parse(code_str)
    except SyntaxError as e:
        return False, f"SyntaxError: {e}"

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                mod_base = alias.name.split('.')[0]
                if mod_base in FORBIDDEN_MODULES:
                    return False, f"Forbidden import: {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            mod_base = (node.module or "").split('.')[0]
            if mod_base in FORBIDDEN_MODULES:
                return False, f"Forbidden import from: {node.module}"
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id in FORBIDDEN_CALLS:
                    return False, f"Forbidden call: {node.func.id}"
            elif isinstance(node.func, ast.Attribute):
                if node.func.attr in FORBIDDEN_CALLS:
                    return False, f"Forbidden attribute call: {node.func.attr}"
    return True, "Safe"

def resolve_fixture_file(path_str):
    path_lower = path_str.lower().replace("\\", "/")
    if "settings.json" in path_lower:
        return r"C:\ai_models\test_fixtures\settings.json"
    elif "hosts" in path_lower:
        return r"C:\ai_models\test_fixtures\hosts"
    elif "production_error.log" in path_lower or "error.log" in path_lower:
        return r"C:\ai_models\test_fixtures\production_error.log"
    return None

def evaluate_response_strict(case, raw_text):
    category = case["category"]
    needs_tool = case["needs_tool"]
    exp_tool = case["expected_tool"]
    
    res = {
        "id": case["id"],
        "category": category,
        "needs_tool": needs_tool,
        "expected_tool": exp_tool,
        "raw_text": raw_text.strip(),
        # 5 distinct stages
        "s1_tool_attempted": False,
        "s2_format_parsed": False,
        "parsed_json": None,
        "s3_args_valid": False,
        "s4_executed": "N/A",  # True, False, or 'N/A'
        "s5_result_correct": "N/A",  # True, False, or 'N/A'
        "exec_detail": "",
        "non_tool_compliance": False
    }

    text = raw_text.strip()

    # Step 1: Detect explicit tool call structure (JSON or function syntax)
    # Reject prompt echo: if text starts with "1. get_current_weather" or "1. get_weather" without a JSON object
    has_json_structure = False
    parsed_obj = None

    # Try extracting markdown JSON
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            candidate = json.loads(m.group(1))
            if isinstance(candidate, dict) and ("tool" in candidate or "name" in candidate):
                parsed_obj = candidate
                has_json_structure = True
        except Exception:
            # Malformed JSON inside markdown block
            has_json_structure = True

    if parsed_obj is None:
        # Check raw JSON starting with {"tool" or {"name"
        m2 = re.search(r'\{\s*"(?:tool|name)"\s*:\s*"[^"]+"\s*,\s*"(?:parameters|arguments)"\s*:\s*\{.*?\}\s*\}', text, re.DOTALL)
        if m2:
            try:
                candidate = json.loads(m2.group(0))
                parsed_obj = candidate
                has_json_structure = True
            except Exception:
                has_json_structure = True

    if parsed_obj is None:
        # Raw decode first curly bracket if it contains "tool"
        start_idx = text.find("{")
        if start_idx != -1 and ('"tool"' in text or '"name"' in text):
            try:
                decoder = json.JSONDecoder()
                obj, _ = decoder.raw_decode(text[start_idx:])
                if isinstance(obj, dict) and ("tool" in obj or "name" in obj):
                    parsed_obj = obj
                    has_json_structure = True
            except Exception:
                # If there was a { with "tool" but failed decode, it's an attempted tool structure
                has_json_structure = True

    # Check function call syntax like web_search("...")
    fn_syntax = re.search(r'\b(get_current_weather|web_search|execute_python|file_read|query_database)\s*\(.*?\)', text)
    if fn_syntax and not text.startswith("1."):
        # Function call attempt outside of numbered list echo
        res["s1_tool_attempted"] = True
    elif has_json_structure:
        res["s1_tool_attempted"] = True

    if parsed_obj is not None:
        res["s1_tool_attempted"] = True
        res["s2_format_parsed"] = True
        res["parsed_json"] = parsed_obj

    # Handle negative control cases
    if not needs_tool:
        res["non_tool_compliance"] = not res["s1_tool_attempted"]
        res["s4_executed"] = "N/A"
        res["s5_result_correct"] = "N/A"
        return res

    # If it needs tool, but format is not parsed or not attempted
    if not res["s2_format_parsed"] or not parsed_obj:
        res["s4_executed"] = False
        res["s5_result_correct"] = False
        res["exec_detail"] = "Format not parsed"
        return res

    called_tool = parsed_obj.get("tool") or parsed_obj.get("name")
    params = parsed_obj.get("parameters") or parsed_obj.get("arguments") or {}

    # Stage 3: Tool & Arguments Matched
    if called_tool != exp_tool:
        res["s3_args_valid"] = False
        res["s4_executed"] = False
        res["s5_result_correct"] = False
        res["exec_detail"] = f"Tool mismatch: expected {exp_tool}, got {called_tool}"
        return res

    val_type = case.get("validator_type")

    if exp_tool in ["get_current_weather", "web_search"]:
        # Semantic argument validation
        rules = case.get("validation_rules", {})
        args_ok = True
        for param_key, expected_substrings in rules.items():
            val = str(params.get(param_key, "")).lower()
            if not any(sub in val for sub in expected_substrings):
                args_ok = False
                break
        res["s3_args_valid"] = args_ok
        # Stage 4 & 5 CANNOT be evaluated for external APIs in offline testbed
        res["s4_executed"] = "N/A"
        res["s5_result_correct"] = "N/A"
        res["exec_detail"] = "N/A (External API)"
        return res

    elif exp_tool == "file_read":
        path_str = str(params.get("path", ""))
        rules = case.get("validation_rules", {})
        expected_subs = rules.get("path", [])
        if any(sub in path_str.lower() for sub in expected_subs):
            res["s3_args_valid"] = True
        else:
            res["s3_args_valid"] = False
            res["s4_executed"] = False
            res["s5_result_correct"] = False
            res["exec_detail"] = f"Path '{path_str}' missing expected target"
            return res

        # Stage 4: Real execution on disk
        target_file = resolve_fixture_file(path_str)
        if target_file and os.path.exists(target_file):
            try:
                with open(target_file, "r", encoding="utf-8") as f:
                    content = f.read()
                res["s4_executed"] = True
                # Stage 5: Content verified
                if len(content) > 0:
                    res["s5_result_correct"] = True
                    res["exec_detail"] = f"Read {len(content)} bytes successfully"
                else:
                    res["s5_result_correct"] = False
                    res["exec_detail"] = "File empty"
            except Exception as e:
                res["s4_executed"] = False
                res["s5_result_correct"] = False
                res["exec_detail"] = f"Read error: {e}"
        else:
            res["s4_executed"] = False
            res["s5_result_correct"] = False
            res["exec_detail"] = f"Fixture file not found for path: {path_str}"

    elif exp_tool == "execute_python":
        code_str = params.get("code", "")
        if not isinstance(code_str, str) or len(code_str.strip()) < 5:
            res["s3_args_valid"] = False
            res["s4_executed"] = False
            res["s5_result_correct"] = False
            res["exec_detail"] = "Empty or invalid code string"
            return res

        res["s3_args_valid"] = True

        # Security check before execution
        is_safe, sec_reason = check_python_security(code_str)
        if not is_safe:
            res["s4_executed"] = False
            res["s5_result_correct"] = False
            res["exec_detail"] = f"Security rejected: {sec_reason}"
            return res

        # Isolated execution
        try:
            sub = subprocess.run(
                [sys.executable, "-I", "-c", code_str],
                capture_output=True,
                text=True,
                timeout=2.0
            )
            if sub.returncode == 0:
                res["s4_executed"] = True
                stdout_clean = sub.stdout.strip()
                exp_out = str(case.get("sandbox_expected_output", "")).strip()
                if exp_out in stdout_clean:
                    res["s5_result_correct"] = True
                    res["exec_detail"] = f"Output: '{stdout_clean}' matched '{exp_out}'"
                else:
                    res["s5_result_correct"] = False
                    res["exec_detail"] = f"Output mismatch: got '{stdout_clean}', expected '{exp_out}'"
            else:
                res["s4_executed"] = False
                res["s5_result_correct"] = False
                res["exec_detail"] = f"Execution error: {sub.stderr.strip()[:150]}"
        except subprocess.TimeoutExpired:
            res["s4_executed"] = False
            res["s5_result_correct"] = False
            res["exec_detail"] = "Timeout (>2.0s)"
        except Exception as e:
            res["s4_executed"] = False
            res["s5_result_correct"] = False
            res["exec_detail"] = f"Subprocess error: {e}"

    elif exp_tool == "query_database":
        sql_str = params.get("sql", "")
        if not isinstance(sql_str, str) or len(sql_str.strip()) < 5:
            res["s3_args_valid"] = False
            res["s4_executed"] = False
            res["s5_result_correct"] = False
            res["exec_detail"] = "Empty or invalid SQL string"
            return res

        res["s3_args_valid"] = True

        try:
            conn = sqlite3.connect(":memory:")
            cur = conn.cursor()
            setup_sql = case.get("sql_setup", "")
            if setup_sql:
                cur.executescript(setup_sql)
            cur.execute(sql_str)
            rows = cur.fetchall()
            res["s4_executed"] = True

            # Stage 5: Strict Ground Truth Verification
            if case["id"] == 11:
                # Active users count: expected scalar 2
                if len(rows) > 0 and len(rows[0]) > 0 and rows[0][0] == 2:
                    res["s5_result_correct"] = True
                    res["exec_detail"] = f"Count match: {rows[0][0]}"
                else:
                    res["s5_result_correct"] = False
                    res["exec_detail"] = f"Count mismatch: got {rows}"

            elif case["id"] == 12:
                # Top 5 orders by amount descending: expected amounts [950.0, 800.0, 500.0, 300.0, 200.0]
                expected_amounts = [950.0, 800.0, 500.0, 300.0, 200.0]
                if len(rows) != 5:
                    res["s5_result_correct"] = False
                    res["exec_detail"] = f"Row count mismatch: expected 5, got {len(rows)}"
                else:
                    # Find which column contains the amount
                    # Rows could be (id, total_amount) or (total_amount,) or (id, name, total_amount)
                    matched_amounts = []
                    # Check each column
                    col_count = len(rows[0])
                    found_col = None
                    for c in range(col_count):
                        col_vals = [float(r[c]) for r in rows if isinstance(r[c], (int, float))]
                        if col_vals == expected_amounts:
                            found_col = c
                            break

                    if found_col is not None:
                        res["s5_result_correct"] = True
                        res["exec_detail"] = f"Strict sorted match: {expected_amounts}"
                    else:
                        res["s5_result_correct"] = False
                        res["exec_detail"] = f"Values or ordering mismatch: got {rows}"
            conn.close()
        except Exception as e:
            res["s4_executed"] = False
            res["s5_result_correct"] = False
            res["exec_detail"] = f"SQLite error: {e}"

    return res

if __name__ == "__main__":
    with open(r"C:\ai_models\test_suite_standardized.json", "r", encoding="utf-8") as f:
        cases = json.load(f)

    with open(r"C:\ai_models\paper_artifacts\symmetric_dense_vs_ternary.json", "r", encoding="utf-8") as f:
        cpp_data = json.load(f)

    models_to_eval = [
        ("bitnet_2b_i2_s_base", "BitNet b1.58 Base (I2_S)"),
        ("qwen2.5_1.5b_q4_k_m", "Qwen 2.5 1.5B (Q4_K_M)"),
        ("qwen2.5_3b_q4_k_m", "Qwen 2.5 3B (Q4_K_M)")
    ]

    report = []
    report.append("# Строгий пятиэтапный аудит C++ рантайма с изоляцией окружения\n")
    report.append("- S1: Учитывается только структурная попытка вызова (JSON или синтаксис функции). Повтор промпта не засчитывается.")
    report.append("- S2: Валидный JSON с полями tool/parameters.")
    report.append("- S3: Точное совпадение имени инструмента и семантических сущностей.")
    report.append("- S4: Реальное исполнение: Python (AST проверка + sandbox subprocess), SQLite (:memory:), Files (файловые фикстуры). Для внешних API (weather/search) = N/A.")
    report.append("- S5: Сверка результата с эталоном: точные значения и сортировка в SQL, эталонный stdout в Python, чтение фикстуры. Для внешних API = N/A.\n")

    summary_table = []
    summary_table.append("| Модель | S1: Вызов запрошен | S2: Формат разобран | S3: Аргументы верны | S4: Выполнен в среде (из 7 локальных) | S5: Результат верен (из 7 локальных) | Запрет вызова (из 5) |")
    summary_table.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")

    for mod_key, mod_name in models_to_eval:
        detailed = cpp_data[mod_key]["detailed_results"]
        
        s1_cnt = 0
        s2_cnt = 0
        s3_cnt = 0
        s4_cnt = 0  # out of 7 local tools (Python: 2, SQL: 2, File: 3)
        s5_cnt = 0  # out of 7 local tools
        non_tool_cnt = 0

        report.append(f"## {mod_name}\n")
        report.append("| ID | Инструмент | S1 | S2 | S3 | S4 | S5 | Non-Tool | Детали / Ошибка |")
        report.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")

        for idx, case in enumerate(cases):
            raw = detailed[idx]["response"]
            r = evaluate_response_strict(case, raw)

            if r["needs_tool"]:
                if r["s1_tool_attempted"]: s1_cnt += 1
                if r["s2_format_parsed"]: s2_cnt += 1
                if r["s3_args_valid"]: s3_cnt += 1
                if r["s4_executed"] is True: s4_cnt += 1
                if r["s5_result_correct"] is True: s5_cnt += 1
                non_tool_str = "—"
            else:
                if r["non_tool_compliance"]: non_tool_cnt += 1
                non_tool_str = "[OK]" if r["non_tool_compliance"] else "[FAIL]"

            s1 = "[+]" if r["s1_tool_attempted"] else "[-]"
            s2 = "[+]" if r["s2_format_parsed"] else "[-]"
            s3 = "[+]" if r["s3_args_valid"] else "[-]"
            s4 = "[+]" if r["s4_executed"] is True else ("N/A" if r["s4_executed"] == "N/A" else "[-]")
            s5 = "[+]" if r["s5_result_correct"] is True else ("N/A" if r["s5_result_correct"] == "N/A" else "[-]")

            report.append(f"| {case['id']:02d} | `{case['expected_tool']}` | {s1} | {s2} | {s3} | {s4} | {s5} | {non_tool_str} | {r['exec_detail']} |")

        report.append("\n")
        summary_table.append(f"| {mod_name} | {s1_cnt}/15 ({s1_cnt/15*100:.1f}%) | {s2_cnt}/15 ({s2_cnt/15*100:.1f}%) | {s3_cnt}/15 ({s3_cnt/15*100:.1f}%) | {s4_cnt}/7 ({s4_cnt/7*100:.1f}%) | {s5_cnt}/7 ({s5_cnt/7*100:.1f}%) | {non_tool_cnt}/5 ({non_tool_cnt/5*100:.1f}%) |")

    report.append("## Сводная таблица по строгим критериям\n")
    report.extend(summary_table)

    out_path = r"C:\ai_models\paper_artifacts\STRICT_5STAGE_AUDIT.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report))

    print("Strict 5-stage audit completed.")
