import sys
import argparse
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stdin.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.agent import ToolAgent
from src.file_fact_memory import FileFactMemory
from src.providers.llama_provider import LlamaCppProvider
from src.providers.maple_server_provider import MapleServerProvider
from src.config import LOCAL_MODELS

def main():
    parser = argparse.ArgumentParser(description="Личный ассистент с памятью в файле")
    parser.add_argument("--session", default="default_user", help="ID сессии / профиля")
    parser.add_argument("--file", default="memory.json", help="Путь к файлу памяти")
    parser.add_argument("--model", choices=["qwen", "llama", "maple", "none"], default="qwen", help="Модель")
    parser.add_argument("--clear", action="store_true", help="Очистить память сессии перед запуском")
    args = parser.parse_args()

    memory = FileFactMemory(file_path=args.file, session_id=args.session, show_service_cell=True)
    if args.clear:
        memory.clear()

    provider = None
    if args.model != "none" and args.model in LOCAL_MODELS:
        info = LOCAL_MODELS[args.model]
        if not info["path"].exists():
            parser.error(f"Модель не найдена: {info['path']}")
        if args.model == "maple":
            provider = MapleServerProvider(model_path=info["path"], model_alias=args.model)
        else:
            provider = LlamaCppProvider(
                model_path=info["path"],
                model_alias=args.model,
                chat_format=info["chat_format"]
            )

    agent = ToolAgent(
        provider=provider,
        memory=memory,
        session_id=args.session,
        file_path=args.file,
        show_cell=True
    )

    greeting = agent.get_greeting()
    print(f"\nАссистент: {greeting}\n")

    try:
        while True:
            user_input = input("Вы: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit", "q", "выход"]:
                print(f"\nАссистент: До встречи! Все факты надёжно сохранены в {args.file}.\n")
                break

            result = agent.run(user_input)
            answer = result["final_answer"]
            print(f"\nАссистент: {answer}\n")

    except (KeyboardInterrupt, EOFError):
        print(f"\n\nАссистент: Сессия завершена. Память сохранена в {args.file}.\n")
    finally:
        if provider is not None and hasattr(provider, "close"):
            provider.close()

if __name__ == "__main__":
    main()
