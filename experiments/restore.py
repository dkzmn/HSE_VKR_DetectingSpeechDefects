import csv
import time
from openai import OpenAI

API_KEY = "PPLX_API_KEY"  # сюда подставь свой ключ
INPUT_FILE = "input.txt"
OUTPUT_FILE = "output.csv"

client = OpenAI(
    api_key=API_KEY,
    base_url="https://api.perplexity.ai"
)

SYSTEM_PROMPT = (
    "Ты помогаешь восстанавливать русские скороговорки по искажённой записи. "
    "Отвечай только одной строкой — каноническим вариантом скороговорки без перевода и комментариев. "
    "Если не уверен или такой скороговорки не существует, напиши ровно: НЕ УВЕРЕН."
)

def restore_tongue_twister(line: str) -> str:
    line = line.strip()
    if not line:
        return ""
    resp = client.chat.completions.create(
        model="sonar",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": line}
        ],
        temperature=0.1,
        max_tokens=128
    )
    return resp.choices[0].message.content.strip()

def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        lines = [l.rstrip("\n") for l in f]

    with open(OUTPUT_FILE, "w", encoding="utf-8", newline="") as csvfile:
        writer = csv.writer(csvfile, delimiter=";")
        writer.writerow(["original", "restored"])

        for i, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            print(f"[{i}/{len(lines)}] Обрабатываю: {line!r}")
            try:
                restored = restore_tongue_twister(line)
            except Exception as e:
                print(f"  Ошибка: {e}")
                restored = "ОШИБКА_API"
            writer.writerow([line, restored])
            # Небольшая пауза, чтобы не упереться в лимиты
            time.sleep(0.3)

    print(f"Готово. Результат в {OUTPUT_FILE}")

if __name__ == "__main__":
    main()