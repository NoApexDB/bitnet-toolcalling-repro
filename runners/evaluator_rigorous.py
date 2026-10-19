import ast
import json
import os
import re
import sqlite3

# Registered tools specification with strict schemas
REGISTERED_TOOLS = {
    "get_current_weather": {
        "allowed_params": {"location", "unit"},
        "required_params": {"location"},
        "types": {"location": str, "unit": str}
    },
    "web_search": {
        "allowed_params": {"query", "max_results"},
        "required_params": {"query"},
        "types": {"query": str, "max_results": int}
    },
    "execute_python": {
        "allowed_params": {"code"},
        "required_params": {"code"},
        "types": {"code": str}
    },
    "file_read": {
        "allowed_params": {"path"},
        "required_params": {"path"},
        "types": {"path": str}
    },
    "query_database": {
        "allowed_params": {"sql"},
        "required_params": {"sql"},
        "types": {"sql": str}
    }
}

# Stand test fixtures
FIXTURE_DIR = r"C:\ai_models\test_fixtures"
FIXTURE_MAP = {
    "settings.json": (os.path.join(FIXTURE_DIR, "settings.json"), "app_name"),
    "hosts": (os.path.join(FIXTURE_DIR, "hosts"), "127.0.0.1"),
    "production_error.log": (os.path.join(FIXTURE_DIR, "production_error.log"), "ERROR [DatabasePool]")
}

def parse_tool_envelope(text):
    """
    S1 & S2 Evaluation:
    S1: Structural tool invocation attempt.
        - Excludes numbered list echoes ('1. get_current_weather...').
        - Excludes placeholder echoes ('tool': 'name' or 'tool': 'tool_name').
    S2: Schema validation:
        - Tool name: strictly one of the 5 registered tools.
        - Envelope keys: allows 'tool' or 'name', and 'parameters' or 'arguments'.
        - Parameters: must be a dict.
        - Extra/unknown parameters: STRICTLY FORBIDDEN (extra_fields = False).
        - Required parameters: must be present.
        - Types: must match schema types.
    """
    text = text.strip()
    s1_attempted = False
    s2_valid = False
    parsed_envelope = None
    envelope_error = ""

    # Check for function-call syntax outside list echo
    fn_match = re.search(r'\b(get_current_weather|web_search|execute_python|file_read|query_database)\s*\(.*?\)', text)
    if fn_match and not text.startswith("1."):
        s1_attempted = True

    # Search for JSON candidate
    json_candidate_str = None
    m_code = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m_code:
        candidate_text = m_code.group(1).strip()
        # If the block inside is just an echo of prompt placeholder {"tool": "name"...} preceded by prompt text
        if text.startswith("1.") and ('"tool": "name"' in candidate_text or '"tool": "tool_name"' in candidate_text):
            # Prompt echo with example placeholder
            pass
        else:
            json_candidate_str = candidate_text
            s1_attempted = True
    else:
        # Search for raw JSON starting with {"tool" or {"name"
        m_raw = re.search(r'\{\s*"(?:tool|name)"\s*:\s*"[^"]+".*?\}', text, re.DOTALL)
        if m_raw:
            # Check if this is an echo of prompt example
            raw_substr = m_raw.group(0)
            if '"tool": "name"' in raw_substr or '"tool": "tool_name"' in raw_substr:
                # Prompt placeholder echo
                pass
            else:
                s1_attempted = True
                start = text.find("{")
                depth = 0
                end = -1
                for i in range(start, len(text)):
                    if text[i] == '{': depth += 1
                    elif text[i] == '}':
                        depth -= 1
                        if depth == 0:
                            end = i + 1
                            break
                if end != -1:
                    json_candidate_str = text[start:end]

    if json_candidate_str:
        try:
            obj = json.loads(json_candidate_str)
            if isinstance(obj, dict):
                tool_name = obj.get("tool") or obj.get("name")
                params = obj.get("parameters") if "parameters" in obj else obj.get("arguments")

                # If tool_name is the prompt placeholder literally
                if tool_name in ["name", "tool_name"]:
                    s1_attempted = False
                    envelope_error = "Prompt placeholder echo ('name')"
                elif not tool_name:
                    envelope_error = "Missing tool identifier"
                elif tool_name not in REGISTERED_TOOLS:
                    envelope_error = f"Unregistered tool name '{tool_name}'"
                elif not isinstance(params, dict):
                    envelope_error = f"'parameters' must be a dict, got {type(params).__name__}"
                else:
                    schema = REGISTERED_TOOLS[tool_name]
                    # Check for unknown parameters (forbidden)
                    unknown_params = set(params.keys()) - schema["allowed_params"]
                    if unknown_params:
                        envelope_error = f"Unknown parameter(s) forbidden: {unknown_params}"
                    else:
                        # Check required parameters
                        missing_required = schema["required_params"] - set(params.keys())
                        if missing_required:
                            envelope_error = f"Missing required parameter(s): {missing_required}"
                        else:
                            # Check types
                            type_mismatch = False
                            for pk, pv in params.items():
                                exp_t = schema["types"][pk]
                                if not isinstance(pv, exp_t):
                                    type_mismatch = True
                                    envelope_error = f"Param '{pk}' expected {exp_t.__name__}, got {type(pv).__name__}"
                                    break
                            if not type_mismatch:
                                s2_valid = True
                                parsed_envelope = {"tool": tool_name, "parameters": params}
        except Exception as e:
            envelope_error = f"JSON parse error: {e}"

    return s1_attempted, s2_valid, parsed_envelope, envelope_error

def evaluate_static_heuristic_arguments(case, parsed_envelope):
    """
    S3: Статическая эвристическая проверка аргументов.
    Проверяет семантические сущности, синтаксис AST для Python и целевые структуры SQL.
    """
    if not parsed_envelope:
        return False, "Отсутствует валидный конверт вызова"

    tool_name = parsed_envelope["tool"]
    params = parsed_envelope["parameters"]
    exp_tool = case["expected_tool"]

    if tool_name != exp_tool:
        return False, f"Несоответствие инструмента: ожидался {exp_tool}, получен {tool_name}"

    if tool_name == "get_current_weather":
        loc = str(params.get("location", "")).lower()
        exp_locs = case.get("validation_rules", {}).get("location", [])
        if not any(el in loc for el in exp_locs):
            return False, f"Локация '{loc}' не содержит целевой город {exp_locs}"
        if "unit" in case.get("validation_rules", {}):
            unit = str(params.get("unit", "")).lower()
            exp_units = case["validation_rules"]["unit"]
            if not any(eu in unit for eu in exp_units):
                return False, f"Единицы '{unit}' не соответствуют запрошенным (fahrenheit)"
        return True, "Сущности локации/единиц подтверждены"

    elif tool_name == "web_search":
        query = str(params.get("query", "")).lower()
        exp_subs = case.get("validation_rules", {}).get("query", [])
        if not any(sub in query for sub in exp_subs):
            return False, f"Поисковый запрос '{query}' не содержит ключевые сущности {exp_subs}"
        return True, "Поисковые сущности подтверждены"

    elif tool_name == "file_read":
        path_str = str(params.get("path", "")).lower().replace("\\", "/")
        exp_subs = case.get("validation_rules", {}).get("path", [])
        if not any(sub in path_str for sub in exp_subs):
            return False, f"Путь '{path_str}' не содержит ожидаемый файл {exp_subs}"
        return True, "Целевой файл подтвержден"

    elif tool_name == "execute_python":
        code = params.get("code", "")
        if not isinstance(code, str) or len(code.strip()) < 10:
            return False, "Код слишком короткий или пустой"
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return False, f"Синтаксическая ошибка кода: {e}"

        if case["id"] == 7:  # Fibonacci
            has_algo = any(kw in code.lower() for kw in ["fib", "for ", "while ", "def "])
            has_50 = "50" in code
            if not (has_algo and has_50):
                return False, "Синтаксис разобран, но отсутствуют текстовые признаки Фибоначчи или таргета 50"
            return True, "Синтаксис разобран (ast.parse OK), найдены заданные текстовые признаки задачи"

        elif case["id"] == 8:  # Primes sum 1..500
            code_lower = code.lower()
            has_prime_check = ("%" in code_lower and ("def " in code_lower or "for " in code_lower)) or "is_prime" in code_lower or "prime" in code_lower
            if "range(2, 501)" in code_lower and "%" not in code_lower and "is_prime" not in code_lower:
                return False, "Код суммирует все числа подряд range(2, 501), пропуская проверку простоты чисел"
            if not has_prime_check:
                return False, "Синтаксис разобран, но отсутствуют текстовые признаки проверки чисел на простоту"
            return True, "Синтаксис разобран (ast.parse OK), найдены заданные текстовые признаки задачи"

    elif tool_name == "query_database":
        sql = params.get("sql", "")
        if not isinstance(sql, str) or len(sql.strip()) < 10:
            return False, "SQL-запрос пуст или слишком короткий"
        sql_lower = sql.lower()
        if not sql_lower.startswith("select") and not sql_lower.startswith("with"):
            return False, "Запрос не является SELECT-запросом на чтение"

        if case["id"] == 11:
            if "users" not in sql_lower or "count" not in sql_lower:
                return False, "Запрос не содержит агрегации COUNT по таблице users"
            if not any(f in sql_lower for f in ["active", "is_active", "= 1", "= true", "=true"]):
                return False, "Отсутствует условие фильтрации активных пользователей"
            return True, "SQL-запрос корректно адресует таблицу users с фильтром активности"

        elif case["id"] == 12:
            if "orders" not in sql_lower:
                return False, "Запрос не адресует таблицу orders"
            if "5" not in sql_lower:
                return False, "В запросе отсутствует ограничение 5 заказов"
            return True, "SQL-запрос адресует orders с ограничением 5"

    return False, "Неизвестный сценарий"

def evaluate_execution_and_result_sandboxed(case, parsed_envelope):
    """
    S4 (Выполнение) и S5 (Результат):
    - SQL: Выполнение в SQLite :memory: под authorizer (строго SELECT к разрешенной таблице)
           и progress_handler (таймаут/лимит инструкций).
    - Файлы: Чтение дисковых фикстур стенда и проверка маркеров.
    - Python: Отключено (нет ОС-изоляции).
    - Внешние API: N/A.
    """
    tool_name = parsed_envelope["tool"]
    params = parsed_envelope["parameters"]

    if tool_name in ["get_current_weather", "web_search"]:
        return "N/A (Внешний API)", "N/A (Внешний API)", "Внешний веб-сервис не эмулируется"

    if tool_name == "execute_python":
        return "Не исполнялось (нет изоляции ОС)", "Не исполнялось (нет изоляции ОС)", "Произвольный запуск кода отключен для безопасности хоста"

    if tool_name == "file_read":
        raw_path = params.get("path", "")
        fixture_key = None
        for k in FIXTURE_MAP:
            if k in raw_path.lower():
                fixture_key = k
                break

        if not fixture_key:
            return False, False, f"Путь '{raw_path}' не сопоставлен с фикстурами стенда"

        fixture_file, exp_sub = FIXTURE_MAP[fixture_key]
        if not os.path.exists(fixture_file):
            return False, False, f"Файл фикстуры '{fixture_file}' отсутствует на диске"

        try:
            with open(fixture_file, "r", encoding="utf-8") as f:
                content = f.read()
            s4_exec = True
            s5_res = (exp_sub in content)
            detail = f"Фикстура '{fixture_key}' прочитана ({len(content)} байт), маркер найден: {s5_res}"
            return s4_exec, s5_res, detail
        except Exception as e:
            return False, False, f"Ошибка чтения фикстуры: {e}"

    if tool_name == "query_database":
        sql = params.get("sql", "")
        target_table = "users" if case["id"] == 11 else "orders"

        try:
            conn = sqlite3.connect(":memory:")
            cur = conn.cursor()
            
            # Setup test tables
            setup = case.get("sql_setup", "")
            if setup:
                cur.executescript(setup)

            # Security: Set authorizer permitting ONLY SELECT from the designated table
            def authorizer(action, arg1, arg2, db_name, trigger_name):
                if action == sqlite3.SQLITE_SELECT:
                    return sqlite3.SQLITE_OK
                if action == sqlite3.SQLITE_READ and arg1 == target_table:
                    return sqlite3.SQLITE_OK
                if action == sqlite3.SQLITE_FUNCTION:
                    return sqlite3.SQLITE_OK
                return sqlite3.SQLITE_DENY

            conn.set_authorizer(authorizer)

            # Security: Instruction limiter / execution timeout
            calls = [0]
            def progress_limiter():
                calls[0] += 1
                if calls[0] > 50000: # Abort long loops
                    return 1
                return 0

            conn.set_progress_handler(progress_limiter, 10)

            cur.execute(sql)
            rows = cur.fetchall()
            s4_exec = True
            conn.close()

            if case["id"] == 11:
                # Active users count: strictly 1 row, 1 column with value 2: [(2,)]
                if len(rows) == 1 and len(rows[0]) == 1 and rows[0][0] == 2:
                    return s4_exec, True, f"Точное совпадение формы и значения: {rows}"
                else:
                    return s4_exec, False, f"Форма или значение не совпали: получено {rows}, ожидалось [(2,)]"

            elif case["id"] == 12:
                # Top 5 orders by total_amount descending on this fixture
                expected_desc = [950.0, 800.0, 500.0, 300.0, 200.0]
                if len(rows) != 5:
                    return s4_exec, False, f"Возвращено строк: {len(rows)}, ожидалось ровно 5"

                num_cols = len(rows[0])
                matched = False
                for c in range(num_cols):
                    col_vals = [float(r[c]) for r in rows if isinstance(r[c], (int, float))]
                    if col_vals == expected_desc:
                        matched = True
                        break

                if matched:
                    return s4_exec, True, f"Значения фикстуры строго соответствуют нисходящему топу: {expected_desc}"
                else:
                    return s4_exec, False, f"Значения/сортировка строк {rows} не соответствуют эталону фикстуры {expected_desc}"

        except Exception as e:
            return False, False, f"Ошибка SQLite / блокировка безопасности: {e}"

    return False, False, "Неизвестный поток исполнения"

def run_evaluation():
    with open(r"C:\ai_models\test_suite_standardized.json", "r", encoding="utf-8") as f:
        cases = json.load(f)

    with open(r"C:\ai_models\paper_artifacts\symmetric_dense_vs_ternary.json", "r", encoding="utf-8") as f:
        cpp_data = json.load(f)

    models = [
        ("bitnet_2b_i2_s_base", "BitNet b1.58 Base (I2_S)"),
        ("qwen2.5_1.5b_q4_k_m", "Qwen 2.5 1.5B (Q4_K_M)"),
        ("qwen2.5_3b_q4_k_m", "Qwen 2.5 3B (Q4_K_M)")
    ]

    report = []
    report.append("# Выверенный покейсовый аудит C++ рантайма (Без завышений)\n")
    report.append("## Обновленные правила классификации:")
    report.append("1. **S1 (Текстовая попытка вызова)**: фиксирует наличие в сгенерированном тексте синтаксического JSON-конверта или функционального вызова, а не факт нативного вызова инструмента в рантайме. Постобработка JSON не является реальным исполнением. Эхо списка инструментов промпта и плейсхолдеров (`'tool': 'name'`) исключены.")
    report.append("2. **S2 (Схема конверта валидна)**: зарегистрированное имя инструмента (из 5 разрешенных), словарь `parameters`/`arguments`, отсутствие неизвестных параметров (`extra_fields = False`) и соответствие типов.")
    report.append("3. **S3 (Статическая эвристическая проверка аргументов)**: проверка наличия ожидаемых сущностей и синтаксический разбор `ast.parse` для Python без выполнения кода и без анализа графа алгоритма.")
    report.append("4. **S4 (Выполнение в изолированной среде)**:")
    report.append("   - SQL: запуск в SQLite :memory: под authorizer (только SELECT к целевой таблице) и progress_handler (прерывание после ~500 000 инструкций виртуальной машины SQLite: 50 000 срабатываний с шагом 10 инструкций; не является секундным тайм-аутом).")
    report.append("   - Файлы: чтение локальных дисковых фикстур стенда (test_fixtures).")
    report.append("   - Python: **Не исполнялось (нет изоляции ОС)**. Произвольный запуск отключен по соображениям безопасности.")
    report.append("   - Погода / Поиск: **N/A (Внешний API)**. Внешние сервисы не эмулируются.")
    report.append("5. **S5 (Результат верен на фикстуре)**:")
    report.append("   - SQL: проверка формы [(2,)] для кейса 11; совпадение упорядоченных сумм [950, 800, 500, 300, 200] для кейса 12 на данной фикстуре (наличие конструкции ORDER BY DESC в AST отдельно не проверяется).")
    report.append("   - Файлы: проверка маркеров в прочитанном файле фикстуры.")
    report.append("6. **Отсутствие синтаксиса вызова (Non-Tool)**: оценивается на 5 негативных кейсах (штраф за синтаксис инструмента, включая фиктивный `{'tool': 'none'}`).\n")

    summary_rows = []

    for mod_key, mod_name in models:
        detailed = cpp_data[mod_key]["detailed_results"]
        
        # Verify alignment of case IDs
        for idx, case in enumerate(cases):
            assert case["id"] == detailed[idx]["id"], f"Case ID mismatch at index {idx}: {case['id']} != {detailed[idx]['id']}"

        s1_cnt = 0
        s2_cnt = 0
        s3_cnt = 0
        s4_local_cnt = 0
        s5_local_cnt = 0
        local_total = 5  # 2 SQL + 3 Files
        non_tool_clean = 0

        report.append(f"### Модель: {mod_name}\n")
        report.append("| ID | Ожидаемый инструмент | S1: Вызов запрошен | S2: Схема конверта | S3: Статическая эвристика | S4: Выполнение (SQL/File) | S5: Результат (SQL/File) | Негативный контроль (Non-Tool) | Детали / Ошибка |")
        report.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")

        for idx, case in enumerate(cases):
            cid = case["id"]
            exp_tool = case["expected_tool"]
            raw_text = detailed[idx]["response"]

            s1, s2, envelope, env_err = parse_tool_envelope(raw_text)

            if case["needs_tool"]:
                if s1: s1_cnt += 1
                if s2: s2_cnt += 1

                s3 = False
                s3_detail = ""
                if s2:
                    s3, s3_detail = evaluate_static_heuristic_arguments(case, envelope)
                if s3: s3_cnt += 1

                s4_status = "N/A"
                s5_status = "N/A"
                exec_detail = ""

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

                s1_s = "[+]" if s1 else "[-]"
                s2_s = "[+]" if s2 else "[-]"
                s3_s = "[+]" if s3 else "[-]"
                rule_str = "—"
            else:
                # Negative control
                if not s1:
                    non_tool_clean += 1
                    rule_str = "[OK] Нет вызова"
                else:
                    if '{"tool": "none"' in raw_text:
                        rule_str = "[FAIL] Вывод фиктивного tool:none"
                    else:
                        rule_str = "[FAIL] Вызов инструмента"

                s1_s = "[+]" if s1 else "[-]"
                s2_s = "[+]" if s2 else "[-]"
                s3_s = "—"
                s4_status = "—"
                s5_status = "—"
                exec_detail = "Ответ модели: " + repr(raw_text[:60])

            report.append(f"| {cid:02d} | `{exp_tool}` | {s1_s} | {s2_s} | {s3_s} | {s4_status} | {s5_status} | {rule_str} | {exec_detail} |")

        report.append("\n")
        summary_rows.append(
            f"| {mod_name} | {s1_cnt}/15 ({s1_cnt/15*100:.1f}%) | {s2_cnt}/15 ({s2_cnt/15*100:.1f}%) | {s3_cnt}/15 ({s3_cnt/15*100:.1f}%) | {s4_local_cnt}/{local_total} ({s4_local_cnt/local_total*100:.1f}%) | {s5_local_cnt}/{local_total} ({s5_local_cnt/local_total*100:.1f}%) | {non_tool_clean}/5 ({non_tool_clean/5*100:.1f}%) |"
        )

    report.append("## Сводная таблица по выверенным критериям\n")
    report.append("| Модель | S1: Вызов запрошен | S2: Схема конверта | S3: Статическая эвристика | S4: Выполнение (из 5 локальных SQL/File) | S5: Результат (из 5 локальных SQL/File) | Отсутствие синтаксиса вызова (из 5) |")
    report.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
    report.extend(summary_rows)

    out_file = r"C:\ai_models\paper_artifacts\RIGOROUS_CPP_AUDIT.md"
    with open(out_file, "w", encoding="utf-8") as f:
        f.write("\n".join(report))

    print("Rigorous audit generated successfully at:", out_file)

if __name__ == "__main__":
    run_evaluation()
