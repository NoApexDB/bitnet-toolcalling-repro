# Независимый пятиэтапный аудит ответов C++ рантайма

Оценка проведена с реальным исполнением Python-кода в subprocess и SQL-запросов в SQLite in-memory.

## Модель: BitNet b1.58 (I2_S Base)

| ID | Ожидаемый инструмент | S1: Вызов выбран | S2: JSON разобран | S3: Аргументы верны | S4: Выполнен | S5: Результат верен | Правило Non-Tool | Детали исполнения / Ошибка |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 01 | `get_current_weather` | [+] | [+] | [-] | [-] | [-] | — |  |
| 02 | `get_current_weather` | [+] | [+] | [-] | [-] | [-] | — |  |
| 03 | `get_current_weather` | [+] | [-] | [-] | [-] | [-] | — | Prompt echo / Unparsed |
| 04 | `web_search` | [+] | [-] | [-] | [-] | [-] | — | Prompt echo / Unparsed |
| 05 | `web_search` | [+] | [-] | [-] | [-] | [-] | — | Prompt echo / Unparsed |
| 06 | `web_search` | [-] | [-] | [-] | [-] | [-] | — |  |
| 07 | `execute_python` | [+] | [-] | [-] | [-] | [-] | — | Prompt echo / Unparsed |
| 08 | `execute_python` | [+] | [-] | [-] | [-] | [-] | — | Prompt echo / Unparsed |
| 09 | `file_read` | [+] | [-] | [-] | [-] | [-] | — | Prompt echo / Unparsed |
| 10 | `file_read` | [+] | [-] | [-] | [-] | [-] | — | Prompt echo / Unparsed |
| 11 | `query_database` | [+] | [-] | [-] | [-] | [-] | — | Prompt echo / Unparsed |
| 12 | `query_database` | [+] | [-] | [-] | [-] | [-] | — | Prompt echo / Unparsed |
| 13 | `get_current_weather` | [-] | [-] | [-] | [-] | [-] | — |  |
| 14 | `file_read` | [+] | [-] | [-] | [-] | [-] | — | Prompt echo / Unparsed |
| 15 | `web_search` | [+] | [-] | [-] | [-] | [-] | — | Prompt echo / Unparsed |
| 16 | `None` | [+] | [-] | [-] | [-] | [-] | [FAIL] Вызов тула | Prompt echo / Unparsed |
| 17 | `None` | [-] | [-] | [-] | [-] | [-] | [OK] Соблюдено |  |
| 18 | `None` | [+] | [-] | [-] | [-] | [-] | [FAIL] Вызов тула | Prompt echo / Unparsed |
| 19 | `None` | [+] | [-] | [-] | [-] | [-] | [FAIL] Вызов тула | Prompt echo / Unparsed |
| 20 | `None` | [+] | [-] | [-] | [-] | [-] | [FAIL] Вызов тула | Prompt echo / Unparsed |


## Модель: Qwen 2.5 1.5B (Q4_K_M)

| ID | Ожидаемый инструмент | S1: Вызов выбран | S2: JSON разобран | S3: Аргументы верны | S4: Выполнен | S5: Результат верен | Правило Non-Tool | Детали исполнения / Ошибка |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 01 | `get_current_weather` | [-] | [-] | [-] | [-] | [-] | — |  |
| 02 | `get_current_weather` | [+] | [+] | [+] | [+] | [+] | — | Out: Mock success for get_current_weather |
| 03 | `get_current_weather` | [+] | [+] | [+] | [+] | [+] | — | Out: Mock success for get_current_weather |
| 04 | `web_search` | [-] | [-] | [-] | [-] | [-] | — |  |
| 05 | `web_search` | [+] | [-] | [-] | [-] | [-] | — | Prompt echo / Unparsed |
| 06 | `web_search` | [-] | [-] | [-] | [-] | [-] | — |  |
| 07 | `execute_python` | [+] | [+] | [+] | [+] | [-] | — |  |
| 08 | `execute_python` | [+] | [+] | [+] | [+] | [-] | — | Out: 125249 |
| 09 | `file_read` | [-] | [-] | [-] | [-] | [-] | — |  |
| 10 | `file_read` | [-] | [-] | [-] | [-] | [-] | — |  |
| 11 | `query_database` | [-] | [-] | [-] | [-] | [-] | — |  |
| 12 | `query_database` | [+] | [+] | [+] | [+] | [+] | — | Out: [(6, 950.0), (4, 800.0), (2, 500.0), (7, 300.0), (3, 200.0)] |
| 13 | `get_current_weather` | [-] | [-] | [-] | [-] | [-] | — |  |
| 14 | `file_read` | [-] | [-] | [-] | [-] | [-] | — |  |
| 15 | `web_search` | [-] | [-] | [-] | [-] | [-] | — |  |
| 16 | `None` | [-] | [-] | [-] | [-] | [-] | [OK] Соблюдено |  |
| 17 | `None` | [-] | [-] | [-] | [-] | [-] | [OK] Соблюдено |  |
| 18 | `None` | [-] | [-] | [-] | [-] | [-] | [OK] Соблюдено |  |
| 19 | `None` | [-] | [-] | [-] | [-] | [-] | [OK] Соблюдено |  |
| 20 | `None` | [-] | [-] | [-] | [-] | [-] | [OK] Соблюдено |  |


## Модель: Qwen 2.5 3B (Q4_K_M)

| ID | Ожидаемый инструмент | S1: Вызов выбран | S2: JSON разобран | S3: Аргументы верны | S4: Выполнен | S5: Результат верен | Правило Non-Tool | Детали исполнения / Ошибка |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 01 | `get_current_weather` | [+] | [+] | [-] | [-] | [-] | — |  |
| 02 | `get_current_weather` | [+] | [+] | [+] | [+] | [+] | — | Out: Mock success for get_current_weather |
| 03 | `get_current_weather` | [+] | [+] | [+] | [+] | [+] | — | Out: Mock success for get_current_weather |
| 04 | `web_search` | [+] | [+] | [+] | [+] | [+] | — | Out: Mock success for web_search |
| 05 | `web_search` | [+] | [+] | [+] | [+] | [+] | — | Out: Mock success for web_search |
| 06 | `web_search` | [+] | [+] | [+] | [+] | [+] | — | Out: Mock success for web_search |
| 07 | `execute_python` | [+] | [+] | [+] | [-] | [-] | — | Error: Traceback (most recent call last):
  File "<string>", line 1, in <module>
    fibonacci = [0, 1] + [fibonacci[-2] + fibonacci[-1] for _ in range(50)] if len(fibonacci) < 51 else fibonacci; print(fibon |
| 08 | `execute_python` | [+] | [+] | [+] | [+] | [-] | — |  |
| 09 | `file_read` | [+] | [+] | [+] | [+] | [+] | — | Out: Mock success for file_read |
| 10 | `file_read` | [+] | [+] | [+] | [+] | [+] | — | Out: Mock success for file_read |
| 11 | `query_database` | [+] | [+] | [+] | [+] | [+] | — | Out: [(2,)] |
| 12 | `query_database` | [+] | [+] | [+] | [+] | [+] | — | Out: [(6, 950.0), (4, 800.0), (2, 500.0), (7, 300.0), (3, 200.0)] |
| 13 | `get_current_weather` | [+] | [+] | [+] | [+] | [+] | — | Out: Mock success for get_current_weather |
| 14 | `file_read` | [+] | [+] | [+] | [+] | [+] | — | Out: Mock success for file_read |
| 15 | `web_search` | [+] | [+] | [+] | [+] | [+] | — | Out: Mock success for web_search |
| 16 | `None` | [+] | [+] | [-] | [-] | [-] | [FAIL] Вызов тула |  |
| 17 | `None` | [+] | [+] | [-] | [-] | [-] | [FAIL] Вызов тула |  |
| 18 | `None` | [+] | [+] | [-] | [-] | [-] | [FAIL] Вызов тула |  |
| 19 | `None` | [-] | [-] | [-] | [-] | [-] | [OK] Соблюдено |  |
| 20 | `None` | [+] | [+] | [-] | [-] | [-] | [FAIL] Вызов тула |  |


## Сводная пятиэтапная таблица (C++ рантайм)

| Модель | S1: Вызов выбран | S2: JSON разобран | S3: Аргументы верны | S4: Выполнен в песочнице | S5: Результат верен | Соблюдение Non-Tool (из 5) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| BitNet b1.58 (I2_S Base) | 13/15 (86.7%) | 2/15 (13.3%) | 0/15 (0.0%) | 0/15 (0.0%) | 0/15 (0.0%) | 1/5 (20.0%) |
| Qwen 2.5 1.5B (Q4_K_M) | 6/15 (40.0%) | 5/15 (33.3%) | 5/15 (33.3%) | 5/15 (33.3%) | 3/15 (20.0%) | 5/5 (100.0%) |
| Qwen 2.5 3B (Q4_K_M) | 15/15 (100.0%) | 15/15 (100.0%) | 14/15 (93.3%) | 13/15 (86.7%) | 12/15 (80.0%) | 1/5 (20.0%) |