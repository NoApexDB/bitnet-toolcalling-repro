import json
import os

cpp_json_path = r"C:\ai_models\paper_artifacts\symmetric_dense_vs_ternary.json"
output_md = r"C:\ai_models\paper_artifacts\REAL_CPP_AUDIT.md"

with open(cpp_json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Extract actual C++ prompts from benchmark_dense_vs_ternary.py
cpp_prompts = {
    1: ("get_current_weather", "Какая сейчас погода в Токио?"),
    2: ("get_current_weather", "What is the temperature in London right now?"),
    3: ("get_current_weather", "Посмотри погоду в Берлине в градусах Фаренгейта"),
    4: ("web_search", "Кто стал чемпионом мира по футболу в 2022 году?"),
    5: ("web_search", "Найди последние новости об искусственном интеллекте за эту неделю"),
    6: ("web_search", "Поищи в интернете информацию о BitNet b1.58 от Microsoft"),
    7: ("execute_python", "Вычисли 50-е число Фибоначчи с помощью Python скрипта"),
    8: ("execute_python", "Посчитай сумму всех простых чисел от 1 до 500 через Python"),
    9: ("file_read", "Прочитай системный файл /etc/hosts"),
    10: ("file_read", "Пожалуйста, открой и прочитай файл C:\\ai_models\\settings.json"),
    11: ("query_database", "Сделай SQL запрос для подсчета всех активных пользователей в таблице users"),
    12: ("query_database", "Выполни SQL: найди топ 5 заказов с наибольшей суммой из таблицы orders"),
    13: ("get_current_weather", "[Контекст: Париж] Какая там сейчас погода?"),
    14: ("file_read", "[Контекст: /var/log/production_error.log] Пожалуйста, прочти его."),
    15: ("web_search", "[Контекст: PyTorch 2.5] Поищи в интернете дату официального релиза этой версии."),
    16: (None, "Что такое градиентный спуск в машинном обучении?"),
    17: (None, "Переведи фразу 'Artificial intelligence revolution' на русский язык."),
    18: (None, "Объясни простыми словами разницу между структурой стек (Stack) и очередь (Queue)."),
    19: (None, "Привет! Как поживаешь?"),
    20: (None, "В чем главное концептуальное отличие синхронного выполнения от асинхронного?")
}

bitnet_results = data["bitnet_2b_i2_s_base"]["detailed_results"]
qwen15_results = data["qwen2.5_1.5b_q4_k_m"]["detailed_results"]
qwen3_results = data["qwen2.5_3b_q4_k_m"]["detailed_results"]

lines = []
lines.append("# Аудит фактических ответов C++ рантайма по реальным промптам\n")
lines.append("| ID | Реальный запрос C++ теста | Ожидаемый инструмент | BitNet b1.58 (I2_S Base) | Qwen 2.5 1.5B (Q4_K_M) | Qwen 2.5 3B (Q4_K_M) |")
lines.append("| :--- | :--- | :--- | :--- | :--- | :--- |")

for i in range(1, 21):
    exp_tool, query = cpp_prompts[i]
    b_resp = bitnet_results[i-1]["response"].replace("\n", " ").replace("|", "\\|")[:90]
    q15_resp = qwen15_results[i-1]["response"].replace("\n", " ").replace("|", "\\|")[:90]
    q3_resp = qwen3_results[i-1]["response"].replace("\n", " ").replace("|", "\\|")[:90]
    
    b_call = bitnet_results[i-1]["parsed_call"]
    q15_call = qwen15_results[i-1]["parsed_call"]
    q3_call = qwen3_results[i-1]["parsed_call"]

    lines.append(f"| {i:02d} | {query} | `{exp_tool}` | Call: {b_call}<br>Raw: `{b_resp}...` | Call: {q15_call}<br>Raw: `{q15_resp}...` | Call: {q3_call}<br>Raw: `{q3_resp}...` |")

with open(output_md, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print("Saved REAL_CPP_AUDIT.md successfully.")
