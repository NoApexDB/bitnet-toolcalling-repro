# Строгий пятиэтапный аудит C++ рантайма с изоляцией окружения

- S1: Учитывается только структурная попытка вызова (JSON или синтаксис функции). Повтор промпта не засчитывается.
- S2: Валидный JSON с полями tool/parameters.
- S3: Точное совпадение имени инструмента и семантических сущностей.
- S4: Реальное исполнение: Python (AST проверка + sandbox subprocess), SQLite (:memory:), Files (файловые фикстуры). Для внешних API (weather/search) = N/A.
- S5: Сверка результата с эталоном: точные значения и сортировка в SQL, эталонный stdout в Python, чтение фикстуры. Для внешних API = N/A.

## BitNet b1.58 Base (I2_S)

| ID | Инструмент | S1 | S2 | S3 | S4 | S5 | Non-Tool | Детали / Ошибка |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 01 | `get_current_weather` | [+] | [+] | [-] | [-] | [-] | — | Tool mismatch: expected get_current_weather, got weather |
| 02 | `get_current_weather` | [+] | [+] | [-] | [-] | [-] | — | Tool mismatch: expected get_current_weather, got temperature |
| 03 | `get_current_weather` | [-] | [-] | [-] | [-] | [-] | — | Format not parsed |
| 04 | `web_search` | [+] | [-] | [-] | [-] | [-] | — | Format not parsed |
| 05 | `web_search` | [+] | [-] | [-] | [-] | [-] | — | Format not parsed |
| 06 | `web_search` | [-] | [-] | [-] | [-] | [-] | — | Format not parsed |
| 07 | `execute_python` | [-] | [-] | [-] | [-] | [-] | — | Format not parsed |
| 08 | `execute_python` | [-] | [-] | [-] | [-] | [-] | — | Format not parsed |
| 09 | `file_read` | [-] | [-] | [-] | [-] | [-] | — | Format not parsed |
| 10 | `file_read` | [-] | [-] | [-] | [-] | [-] | — | Format not parsed |
| 11 | `query_database` | [-] | [-] | [-] | [-] | [-] | — | Format not parsed |
| 12 | `query_database` | [-] | [-] | [-] | [-] | [-] | — | Format not parsed |
| 13 | `get_current_weather` | [-] | [-] | [-] | [-] | [-] | — | Format not parsed |
| 14 | `file_read` | [-] | [-] | [-] | [-] | [-] | — | Format not parsed |
| 15 | `web_search` | [-] | [-] | [-] | [-] | [-] | — | Format not parsed |
| 16 | `None` | [-] | [-] | [-] | N/A | N/A | [OK] |  |
| 17 | `None` | [-] | [-] | [-] | N/A | N/A | [OK] |  |
| 18 | `None` | [-] | [-] | [-] | N/A | N/A | [OK] |  |
| 19 | `None` | [-] | [-] | [-] | N/A | N/A | [OK] |  |
| 20 | `None` | [-] | [-] | [-] | N/A | N/A | [OK] |  |


## Qwen 2.5 1.5B (Q4_K_M)

| ID | Инструмент | S1 | S2 | S3 | S4 | S5 | Non-Tool | Детали / Ошибка |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 01 | `get_current_weather` | [-] | [-] | [-] | [-] | [-] | — | Format not parsed |
| 02 | `get_current_weather` | [+] | [+] | [+] | N/A | N/A | — | N/A (External API) |
| 03 | `get_current_weather` | [+] | [+] | [+] | N/A | N/A | — | N/A (External API) |
| 04 | `web_search` | [-] | [-] | [-] | [-] | [-] | — | Format not parsed |
| 05 | `web_search` | [+] | [-] | [-] | [-] | [-] | — | Format not parsed |
| 06 | `web_search` | [-] | [-] | [-] | [-] | [-] | — | Format not parsed |
| 07 | `execute_python` | [+] | [+] | [+] | [+] | [-] | — | Output mismatch: got '', expected '12586269025' |
| 08 | `execute_python` | [+] | [+] | [+] | [+] | [-] | — | Output mismatch: got '125249', expected '21536' |
| 09 | `file_read` | [-] | [-] | [-] | [-] | [-] | — | Format not parsed |
| 10 | `file_read` | [-] | [-] | [-] | [-] | [-] | — | Format not parsed |
| 11 | `query_database` | [-] | [-] | [-] | [-] | [-] | — | Format not parsed |
| 12 | `query_database` | [+] | [+] | [+] | [+] | [+] | — | Strict sorted match: [950.0, 800.0, 500.0, 300.0, 200.0] |
| 13 | `get_current_weather` | [-] | [-] | [-] | [-] | [-] | — | Format not parsed |
| 14 | `file_read` | [-] | [-] | [-] | [-] | [-] | — | Format not parsed |
| 15 | `web_search` | [-] | [-] | [-] | [-] | [-] | — | Format not parsed |
| 16 | `None` | [-] | [-] | [-] | N/A | N/A | [OK] |  |
| 17 | `None` | [-] | [-] | [-] | N/A | N/A | [OK] |  |
| 18 | `None` | [-] | [-] | [-] | N/A | N/A | [OK] |  |
| 19 | `None` | [-] | [-] | [-] | N/A | N/A | [OK] |  |
| 20 | `None` | [-] | [-] | [-] | N/A | N/A | [OK] |  |


## Qwen 2.5 3B (Q4_K_M)

| ID | Инструмент | S1 | S2 | S3 | S4 | S5 | Non-Tool | Детали / Ошибка |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 01 | `get_current_weather` | [+] | [+] | [-] | N/A | N/A | — | N/A (External API) |
| 02 | `get_current_weather` | [+] | [+] | [+] | N/A | N/A | — | N/A (External API) |
| 03 | `get_current_weather` | [+] | [+] | [+] | N/A | N/A | — | N/A (External API) |
| 04 | `web_search` | [+] | [+] | [+] | N/A | N/A | — | N/A (External API) |
| 05 | `web_search` | [+] | [+] | [+] | N/A | N/A | — | N/A (External API) |
| 06 | `web_search` | [+] | [+] | [+] | N/A | N/A | — | N/A (External API) |
| 07 | `execute_python` | [+] | [+] | [+] | [-] | [-] | — | Execution error: Traceback (most recent call last):
  File "<string>", line 1, in <module>
    fibonacci = [0, 1] + [fibonacci[-2] + fibonacci[-1] for _ in range(50)]  |
| 08 | `execute_python` | [+] | [+] | [+] | [+] | [-] | — | Output mismatch: got '', expected '21536' |
| 09 | `file_read` | [+] | [+] | [+] | [+] | [+] | — | Read 56 bytes successfully |
| 10 | `file_read` | [+] | [+] | [+] | [+] | [+] | — | Read 69 bytes successfully |
| 11 | `query_database` | [+] | [+] | [+] | [+] | [+] | — | Count match: 2 |
| 12 | `query_database` | [+] | [+] | [+] | [+] | [+] | — | Strict sorted match: [950.0, 800.0, 500.0, 300.0, 200.0] |
| 13 | `get_current_weather` | [+] | [+] | [+] | N/A | N/A | — | N/A (External API) |
| 14 | `file_read` | [+] | [+] | [+] | [+] | [+] | — | Read 74 bytes successfully |
| 15 | `web_search` | [+] | [+] | [+] | N/A | N/A | — | N/A (External API) |
| 16 | `None` | [+] | [+] | [-] | N/A | N/A | [FAIL] |  |
| 17 | `None` | [+] | [+] | [-] | N/A | N/A | [FAIL] |  |
| 18 | `None` | [+] | [+] | [-] | N/A | N/A | [FAIL] |  |
| 19 | `None` | [-] | [-] | [-] | N/A | N/A | [OK] |  |
| 20 | `None` | [+] | [+] | [-] | N/A | N/A | [FAIL] |  |


## Сводная таблица по строгим критериям

| Модель | S1: Вызов запрошен | S2: Формат разобран | S3: Аргументы верны | S4: Выполнен в среде (из 7 локальных) | S5: Результат верен (из 7 локальных) | Запрет вызова (из 5) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| BitNet b1.58 Base (I2_S) | 4/15 (26.7%) | 2/15 (13.3%) | 0/15 (0.0%) | 0/7 (0.0%) | 0/7 (0.0%) | 5/5 (100.0%) |
| Qwen 2.5 1.5B (Q4_K_M) | 6/15 (40.0%) | 5/15 (33.3%) | 5/15 (33.3%) | 3/7 (42.9%) | 1/7 (14.3%) | 5/5 (100.0%) |
| Qwen 2.5 3B (Q4_K_M) | 15/15 (100.0%) | 15/15 (100.0%) | 14/15 (93.3%) | 6/7 (85.7%) | 5/7 (71.4%) | 1/5 (20.0%) |