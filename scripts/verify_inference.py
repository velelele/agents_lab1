"""
Verification script for Requirement 1:
- Поднять локально модель
- Убедиться, что инференс работает
"""
import sys
import time
from pathlib import Path

# Set UTF-8 encoding for Windows console
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import LOCAL_MODELS
from src.providers.llama_provider import LlamaCppProvider
from src.providers.maple_server_provider import MapleServerProvider


def test_model(key: str):
    if key not in LOCAL_MODELS:
        print(f"[ERROR] Неизвестная модель: {key}")
        return

    info = LOCAL_MODELS[key]
    model_path = info["path"]

    if not model_path.exists():
        print(f"[SKIP] Модель {key} еще не скачана ({model_path.name})")
        return

    print(f"\n==================================================")
    print(f"Тестирование инференса: {info['name']}")
    print(f"Файл: {model_path}")
    print(f"==================================================")

    if key == "maple":
        provider = MapleServerProvider(model_path=model_path, model_alias=key)
    else:
        provider = LlamaCppProvider(
            model_path=model_path,
            model_alias=key,
            chat_format=info["chat_format"]
        )

    test_messages = [
        {"role": "system", "content": "Ты дружелюбный ИИ-помощник. Отвечай кратко на русском языке."},
        {"role": "user", "content": "Привет! Назови столицу Франции и столицу Японии."}
    ]

    print("\n[+] Запрос к модели: 'Привет! Назови столицу Франции и столицу Японии.'")
    print("[+] Стриминг ответа:")
    print("--------------------------------------------------")

    t0 = time.time()
    tokens_count = 0
    full_response = ""

    for chunk in provider.stream_generate(test_messages, max_tokens=150, temperature=0.3):
        print(chunk, end="", flush=True)
        full_response += chunk
        tokens_count += 1

    elapsed = time.time() - t0
    print("\n--------------------------------------------------")
    print(f"[OK] Инференс успешно выполнен!")
    print(f"Время генерации: {elapsed:.2f} сек. Сгенерировано чанков: {tokens_count}")
    print(f"==================================================\n")


def main():
    target = sys.argv[1] if len(sys.argv) > 1 else "qwen"
    test_model(target)


if __name__ == "__main__":
    main()
