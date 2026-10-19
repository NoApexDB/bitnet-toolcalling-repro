import json
import os

POSITIVE_FILE = r"C:\ai_models\paper_artifacts\agent_distill_dataset_clean.jsonl"
OUTPUT_FILE = r"C:\ai_models\paper_artifacts\agent_distill_dataset_balanced.jsonl"

SYSTEM_PROMPT = """You are a helpful AI assistant with access to tools.
When you need to use a tool to assist the user, you MUST respond ONLY with a JSON object in the following format:
```json
{"tool": "tool_name", "parameters": {"param1": "value1"}}
```
Do not include any extra thoughts or text before or after the JSON when invoking a tool.
If no tool is required, answer the user directly and concisely in natural language.

Available tools:
1. get_current_weather(location: str, unit: str = "celsius") - Retrieve current weather data for a specific city.
2. web_search(query: str, max_results: int = 5) - Search the web for up-to-date information.
3. execute_python(code: str) - Execute Python script in a secure sandbox and return standard output.
4. file_read(path: str) - Read text content of a local file.
5. query_database(sql: str) - Execute a read-only SQL query against the database.
"""

# 20 high-quality negative dialogues (no tools needed, direct natural language answer)
NEGATIVE_DIALOGUES = [
    {
        "category": "negative_general_qa",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Что такое градиентный спуск в машинном обучении?"},
            {"role": "assistant", "content": "Градиентный спуск — это метод оптимизации для нахождения локального минимума функции потерь путем последовательного перемещения параметров модели в сторону наискорейшего спуска (противоположно градиенту)."}
        ]
    },
    {
        "category": "negative_greeting",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Привет! Как поживаешь?"},
            {"role": "assistant", "content": "Здравствуйте! Я функционирую в штатном режиме и готов помочь вам с решением задач, кодом или анализом данных."}
        ]
    },
    {
        "category": "negative_translation",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Переведи фразу 'Artificial intelligence revolution' на русский язык."},
            {"role": "assistant", "content": "Революция искусственного интеллекта."}
        ]
    },
    {
        "category": "negative_theory",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Объясни простыми словами разницу между структурой стек (Stack) и очередь (Queue)."},
            {"role": "assistant", "content": "Стек работает по принципу LIFO (последним пришел — первым ушел), как стопка тарелок. Очередь работает по принципу FIFO (первым пришел — первым ушел), как очередь в магазине."}
        ]
    },
    {
        "category": "negative_theory",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "В чем главное концептуальное отличие синхронного выполнения от асинхронного?"},
            {"role": "assistant", "content": "При синхронном выполнении каждая операция блокирует поток до своего завершения. При асинхронном выполнении операция запускается в фоне, освобождая поток для выполнения других задач."}
        ]
    },
    {
        "category": "negative_cs_concept",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Что такое сложность O(n log n)?"},
            {"role": "assistant", "content": "O(n log n) — это квазилинейная временная сложность алгоритма, характерная для оптимальных алгоритмов сортировки сравнением, таких как Merge Sort и Quick Sort в среднем случае."}
        ]
    },
    {
        "category": "negative_programming",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Как устроен декоратор в Python?"},
            {"role": "assistant", "content": "Декоратор в Python — это функция, которая принимает другую функцию в качестве аргумента, расширяет или изменяет ее поведение без модификации исходного кода и возвращает новую функцию."}
        ]
    },
    {
        "category": "negative_math_knowledge",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Напомни теорему Пифагора."},
            {"role": "assistant", "content": "В прямоугольном треугольнике квадрат длины гипотенузы равен сумме квадратов длин катетов: a² + b² = c²."}
        ]
    },
    {
        "category": "negative_facts",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Какая столица у Австралии?"},
            {"role": "assistant", "content": "Столицей Австралии является город Канберра."}
        ]
    },
    {
        "category": "negative_architecture",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Что означают принципы SOLID в разработке ПО?"},
            {"role": "assistant", "content": "SOLID — это 5 принципов объектно-ориентированного проектирования: Single Responsibility (единственная ответственность), Open-Closed (открытость/закрытость), Liskov Substitution (подстановка Барбары Лисков), Interface Segregation (разделение интерфейсов), Dependency Inversion (инверсия зависимостей)."}
        ]
    },
    {
        "category": "negative_hardware",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Чем отличается кэш L1 от кэша L3 в процессоре?"},
            {"role": "assistant", "content": "Кэш L1 — самый быстрый, расположен индивидуально на каждом ядре и имеет наименьший объем (обычно десятки килобайт). Кэш L3 — более медленный, но общий для всех ядер и имеет существенно больший объем (десятки мегабайт)."}
        ]
    },
    {
        "category": "negative_networking",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "В чем разница между TCP и UDP протоколами?"},
            {"role": "assistant", "content": "TCP — протокол с установлением соединения, гарантирующий надежную доставку пакетов и их строгий порядок. UDP — протокол без установления соединения, передающий данные с минимальной задержкой, но без гарантии доставки."}
        ]
    },
    {
        "category": "negative_data_structures",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Какова средняя сложность поиска элемента в хэш-таблице?"},
            {"role": "assistant", "content": "Средняя сложность поиска в хэш-таблице составляет O(1). В худшем случае при возникновении множественных коллизий сложность деградирует до O(n)."}
        ]
    },
    {
        "category": "negative_git",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "В чем разница между git merge и git rebase?"},
            {"role": "assistant", "content": "git merge создает специальный коммит слияния, сохраняя исходную историю обеих веток. git rebase переносит коммиты текущей ветки поверх указанной базовой, формируя линейную историю изменений."}
        ]
    },
    {
        "category": "negative_concurrency",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Что такое состояние гонки (Race Condition)?"},
            {"role": "assistant", "content": "Состояние гонки — это ошибка проектирования многопоточных систем, при которой результат работы программы зависит от неконтролируемого порядка переключения и выполнения потоков."}
        ]
    },
    {
        "category": "negative_ml_concepts",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Что такое переобучение (overfitting) в машинном обучении?"},
            {"role": "assistant", "content": "Переобучение — это ситуация, когда модель слишком детально запоминает шум и особенности обучающей выборки, из-за чего теряет способность к обобщению на новых валидационных данных."}
        ]
    },
    {
        "category": "negative_databases",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Что такое свойства ACID в реляционных базах данных?"},
            {"role": "assistant", "content": "ACID — набор требований к транзакциям: Atomicity (атомарность), Consistency (согласованность), Isolation (изолированность), Durability (долговечность)."}
        ]
    },
    {
        "category": "negative_python_features",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Зачем в Python используется ключевое слово yield?"},
            {"role": "assistant", "content": "Ключевое слово yield превращает обычную функцию в генератор, возвращая значение и приостанавливая выполнение функции с сохранением локального состояния до следующего вызова next()."}
        ]
    },
    {
        "category": "negative_polite_conversation",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Спасибо за помощь!"},
            {"role": "assistant", "content": "Пожалуйста! Если понадобятся дополнительные вычисления, поиск данных или разбор архитектуры — обращайтесь."}
        ]
    },
    {
        "category": "negative_algorithms",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Какое ключевое условие необходимо для применения алгоритма бинарного поиска?"},
            {"role": "assistant", "content": "Ключевое условие для бинарного поиска — массив или последовательность данных должны быть предварительно отсортированы."}
        ]
    }
]

def main():
    combined = []
    # 1. Load existing positive examples
    if os.path.exists(POSITIVE_FILE):
        with open(POSITIVE_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    combined.append(json.loads(line))
        print(f"Loaded {len(combined)} positive dialogues.")
    
    # 2. Add negative dialogues
    combined.extend(NEGATIVE_DIALOGUES)
    print(f"Added {len(NEGATIVE_DIALOGUES)} negative dialogues. Total: {len(combined)}")

    # 3. Save balanced dataset
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for item in combined:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Saved balanced dataset to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
