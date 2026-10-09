"""
main.py — Точка входа для лабораторной работы.
Запускает консольный REPL-чат с ассистентом на локальных LLM (llama-cpp).

Использование:
    python main.py                       # Qwen 2.5 3B + tools
    python main.py --model llama         # Llama 3.2 3B + tools
    python main.py --model qwen --no-tools  # Без инструментов (простой чат)
"""
import sys
import argparse

# UTF-8 for Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stdin.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.chat import ConsoleREPLChat


def main():
    parser = argparse.ArgumentParser(
        description="Консольный ассистент с инструментами на локальных LLM (Lab 1)"
    )
    parser.add_argument(
        "--model",
        choices=["qwen", "llama", "maple"],
        default="qwen",
        help="Какую локальную модель использовать (default: qwen)"
    )
    parser.add_argument(
        "--no-tools",
        action="store_true",
        help="Отключить режим инструментов (простой чат со стримингом)"
    )
    args = parser.parse_args()

    chat = None
    try:
        chat = ConsoleREPLChat(
            model_key=args.model,
            enable_tools=not args.no_tools
        )
        chat.run_repl()
    except FileNotFoundError as e:
        print(f"\n[ОШИБКА] {e}")
        print("Скачайте модель командой: python scripts/download_models.py")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nДо свидания!")
        sys.exit(0)
    finally:
        provider = getattr(chat, "provider", None)
        close = getattr(provider, "close", None)
        if close:
            close()


if __name__ == "__main__":
    main()
