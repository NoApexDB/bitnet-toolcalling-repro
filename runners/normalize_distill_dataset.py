import json
import re

INPUT_FILE = r"C:\ai_models\agent_distill_dataset.jsonl"
OUTPUT_FILE = r"C:\ai_models\agent_distill_dataset_clean.jsonl"

def normalize_tool_call(content: str) -> str:
    """Нормализует JSON вызова инструмента к каноническому формату: {'tool': ..., 'parameters': {...}}"""
    # Ищем блок ```json ... ```
    match = re.search(r"```json\s*(\{.*?\})\s*```", content, re.DOTALL)
    if not match:
        return content
    
    raw_json = match.group(1)
    try:
        obj = json.loads(raw_json)
    except Exception:
        return content

    # Определяем имя тулза
    tool_name = obj.get("tool") or obj.get("function") or obj.get("name")
    parameters = obj.get("parameters", {})

    if not tool_name and "command" in obj:
        tool_name = "execute_command"
        parameters = {"command": obj["command"]}
    elif not tool_name and "expression" in obj:
        tool_name = "calculator"
        parameters = {"expression": obj["expression"]}

    if not tool_name:
        tool_name = "unknown_tool"

    canonical = {
        "tool": tool_name,
        "parameters": parameters
    }
    
    clean_json_str = json.dumps(canonical, ensure_ascii=False)
    return f"```json\n{clean_json_str}\n```"

def main():
    cleaned_rows = []
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            messages = entry.get("messages", [])
            for msg in messages:
                if msg.get("role") == "assistant" and "```json" in msg.get("content", ""):
                    msg["content"] = normalize_tool_call(msg["content"])
            cleaned_rows.append(entry)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for row in cleaned_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"[Normalized] Очищено и нормализовано {len(cleaned_rows)} строк в {OUTPUT_FILE}.")

if __name__ == "__main__":
    main()
