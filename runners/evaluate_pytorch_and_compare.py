import json
import os
import sys

# Import functions directly from evaluator_rigorous
sys.path.append(r"C:\ai_models")
from evaluator_rigorous import (
    parse_tool_envelope,
    evaluate_static_heuristic_arguments,
    evaluate_execution_and_result_sandboxed,
    REGISTERED_TOOLS
)

TEST_SUITE_PATH = r"C:\ai_models\test_suite_standardized.json"
CPP_I2S_PATH = r"C:\ai_models\paper_artifacts\symmetric_dense_vs_ternary.json"
PYTORCH_DATA_PATH = r"C:\ai_models\benchmark_results_pytorch_cpu.json"
CPP_F32_PATH = r"C:\ai_models\benchmark_results_bitnet_f32_cpp.json"
REPORT_MD_PATH = r"C:\ai_models\paper_artifacts\COMPARATIVE_AUDIT_PYTORCH_VS_CPP.md"

def run_comparative_evaluation():
    with open(TEST_SUITE_PATH, "r", encoding="utf-8") as f:
        cases = json.load(f)

    with open(CPP_I2S_PATH, "r", encoding="utf-8") as f:
        cpp_i2s_data = json.load(f)

    with open(PYTORCH_DATA_PATH, "r", encoding="utf-8") as f:
        pytorch_data = json.load(f)

    cpp_f32_data = None
    if os.path.exists(CPP_F32_PATH):
        try:
            with open(CPP_F32_PATH, "r", encoding="utf-8") as f:
                cpp_f32_data = json.load(f)
        except Exception:
            pass

    models_to_eval = [
        ("BitNet b1.58 2B (bitnet.cpp CPU I2_S)", cpp_i2s_data["bitnet_2b_i2_s_base"]["detailed_results"], "bitnet.cpp I2_S"),
        ("BitNet b1.58 2B (PyTorch CPU FP32)", pytorch_data["detailed_results"], "PyTorch CPU FP32"),
    ]

    if cpp_f32_data:
        models_to_eval.append(
            ("BitNet b1.58 2B (bitnet.cpp CPU F32)", cpp_f32_data["detailed_results"], "bitnet.cpp F32 (Диагностический)")
        )

    models_to_eval.extend([
        ("Qwen 2.5 1.5B (bitnet.cpp CPU Q4_K_M)", cpp_i2s_data["qwen2.5_1.5b_q4_k_m"]["detailed_results"], "bitnet.cpp Q4_K_M"),
        ("Qwen 2.5 3B (bitnet.cpp CPU Q4_K_M)", cpp_i2s_data["qwen2.5_3b_q4_k_m"]["detailed_results"], "bitnet.cpp Q4_K_M")
    ])

    report = []
    report.append("# Сравнительный аудит инференса: PyTorch/Transformers CPU vs bitnet.cpp CPU (Пилотный стенд)\n")
    report.append("## Параметры тестового стенда:")
    report.append("- CPU: AMD Ryzen 5 5500 (6 ядер / 12 потоков), бенчмарк ограничен ровно 4 потоками во всех прогонах.")
    report.append("- Набор кейсов: `test_suite_standardized.json` (20 кейсов: 15 позитивных вызовов инструментов + 5 негативных контролей).")
    report.append("- Протокол генерации: greedy decoding (`temperature=0.0`), `max_new_tokens=128`, лимит контекста 2048.")
    report.append("- Регулирование повторов: `repetition_penalty=1.15` со скользящим окном `repeat_last_n=64`.")
    report.append("- Стоп-последовательности: `<|eot_id|>`, `<|end_of_text|>`, `User:`.\n")

    summary_rows = []

    for label, detailed, runtime_desc in models_to_eval:
        s1_cnt = 0
        s2_cnt = 0
        s3_cnt = 0
        s4_local_cnt = 0
        s5_local_cnt = 0
        local_total = 5  # 2 SQL + 3 Files
        non_tool_clean = 0

        report.append(f"### Модель и среда: {label} [{runtime_desc}]\n")
        report.append("| ID | Ожидаемый инструмент | S1: Вызов запрошен | S2: Схема конверта | S3: Статическая эвристика | S4: Выполнение (SQL/File) | S5: Результат (SQL/File) | Нет структурного вызова (Non-Tool) | Детали / Ошибка |")
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
                if not s1:
                    non_tool_clean += 1
                    # Distinguish between plain answer and prompt echo
                    if raw_text.startswith("1."):
                        rule_str = "[OK] Нет вызова (эхо промпта)"
                    else:
                        rule_str = "[OK] Нет вызова (текст)"
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
            f"| {label} | {s1_cnt}/15 ({s1_cnt/15*100:.1f}%) | {s2_cnt}/15 ({s2_cnt/15*100:.1f}%) | {s3_cnt}/15 ({s3_cnt/15*100:.1f}%) | {s4_local_cnt}/{local_total} ({s4_local_cnt/local_total*100:.1f}%) | {s5_local_cnt}/{local_total} ({s5_local_cnt/local_total*100:.1f}%) | {non_tool_clean}/5 ({non_tool_clean/5*100:.1f}%) |"
        )

    report.append("## Сводная таблица по выверенным критериям (Пилотный результат)\n")
    report.append("| Модель / Стек | S1: Вызов запрошен | S2: Схема конверта | S3: Статическая эвристика | S4: Выполнение (из 5 локальных SQL/File) | S5: Результат (из 5 локальных SQL/File) | Нет структурного вызова инструмента (из 5) |")
    report.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
    report.extend(summary_rows)

    with open(REPORT_MD_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(report))

    print("Comparative audit generated successfully at:", REPORT_MD_PATH)

if __name__ == "__main__":
    run_comparative_evaluation()
