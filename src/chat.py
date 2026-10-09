"""
Interactive Console REPL Chat.
Fulfills Requirement 3 & Requirement 4:
- Консольный чат в терминале
- История сообщений между репликами
- Стриминг ответа
- Вызов инструментов (tools) и отображение шагов рассуждения
- Корректный выход (/exit, /quit, Ctrl+C)
"""
import sys
import time
from typing import List, Dict, Optional
from pathlib import Path

# Configure UTF-8 for Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stdin.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.config import LOCAL_MODELS
from src.providers.llama_provider import LlamaCppProvider
from src.providers.maple_server_provider import MapleServerProvider
from src.agent import ToolAgent
from src.tools import TOOLS_REGISTRY


class ConsoleREPLChat:
    def __init__(
        self,
        model_key: str = "qwen",
        enable_tools: bool = True,
        system_prompt: Optional[str] = None
    ):
        self.model_key = model_key
        self.enable_tools = enable_tools
        self.history: List[Dict[str, str]] = []

        if model_key not in LOCAL_MODELS:
            raise ValueError(f"Неизвестный ключ модели: {model_key}. Доступны: {list(LOCAL_MODELS.keys())}")

        info = LOCAL_MODELS[model_key]
        if not info["path"].exists():
            raise FileNotFoundError(
                f"Файл модели {model_key} ({info['path'].name}) не найден. "
                f"Скачайте модель: python scripts/download_models.py --model {model_key}"
            )

        print(f"\n==================================================================")
        print(f"🚀 Запуск локального ассистента на базе: {info['name']}")
        print(f"   Файл весов: {info['path'].name}")
        print(f"   Режим инструментов (Tools): {'ВКЛЮЧЕН' if enable_tools else 'ВЫКЛЮЧЕН'}")
        print(f"==================================================================")

        if model_key == "maple":
            self.provider = MapleServerProvider(model_path=info["path"], model_alias=model_key)
        else:
            self.provider = LlamaCppProvider(
                model_path=info["path"],
                model_alias=model_key,
                chat_format=info["chat_format"]
            )
        self.agent = ToolAgent(provider=self.provider, system_prompt=system_prompt)

    def print_banner(self):
        print("\n" + "="*60)
        print("💡 Команды управления:")
        print("  /exit, /quit   — Завершить работу")
        print("  /clear         — Очистить историю диалога")
        print("  /history       — Показать историю сообщений")
        print("  /tools         — Включить/выключить инструменты")
        print("  /help          — Список команд и доступных инструментов")
        print("="*60 + "\n")

    def print_help(self):
        print("\nДоступные команды:")
        print("  /exit, /quit   — Завершить чат")
        print("  /clear         — Очистить историю сообщений")
        print("  /history       — Вывести текущую историю сообщений")
        print("  /tools         — Переключить статус инструментов")
        print("  /help          — Эта справка\n")
        print("Доступные инструменты агента:")
        for name, tinfo in TOOLS_REGISTRY.items():
            print(f"  • {tinfo['description']}")
        print()

    def run_repl(self):
        self.print_banner()

        while True:
            try:
                user_input = input("\n[Вы] > ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\n\n[Выход] Завершение сеанса чата. До свидания!")
                break

            if not user_input:
                continue

            # Check slash commands
            cmd = user_input.lower()
            if cmd in ["/exit", "/quit", "exit", "quit", ":q"]:
                print("\n[Выход] Завершение сеанса чата. До свидания!")
                break

            if cmd == "/clear":
                self.history.clear()
                print("[Инфо] История сообщений успешно очищена.")
                continue

            if cmd == "/history":
                print(f"\n--- История сообщений ({len(self.history)} сообщений) ---")
                for msg in self.history:
                    role_name = "Вы" if msg["role"] == "user" else "Ассистент"
                    print(f"[{role_name}]: {msg['content']}")
                print("---------------------------------------------------\n")
                continue

            if cmd == "/help":
                self.print_help()
                continue

            if cmd == "/tools":
                self.enable_tools = not self.enable_tools
                print(f"[Инфо] Режим инструментов переключен: {'ВКЛЮЧЕН' if self.enable_tools else 'ВЫКЛЮЧЕН'}")
                continue

            # Process prompt
            if self.enable_tools:
                self._handle_agent_query(user_input)
            else:
                self._handle_direct_query(user_input)

    def _handle_direct_query(self, user_input: str):
        """Direct LLM chat with streaming, no tool execution."""
        self.history.append({"role": "user", "content": user_input})
        messages = [
            {"role": "system", "content": "Ты полезный и вежливый AI-ассистент. Отвечай на русском языке."}
        ] + self.history

        print("\n[Ассистент] > ", end="", flush=True)
        assistant_reply = ""
        try:
            for chunk in self.provider.stream_generate(messages, max_tokens=1024, temperature=0.3):
                print(chunk, end="", flush=True)
                assistant_reply += chunk
            print()
            self.history.append({"role": "assistant", "content": assistant_reply})
        except KeyboardInterrupt:
            print("\n[Прервано пользователем]")

    def _handle_agent_query(self, user_input: str):
        """Tool Agent query with tool execution and step logs."""
        print("\n[Ассистент думает...]")

        assistant_started = False

        def on_tool(func_name: str, kwargs: dict, obs: str):
            print(f"\n  ⚙️ [ВЫЗОВ ИНСТРУМЕНТА] {func_name}({kwargs})")
            print(f"  📥 [РЕЗУЛЬТАТ] {obs}")

        def on_token(text: str):
            nonlocal assistant_started
            if not assistant_started:
                print("\n[Ассистент] > ", end="", flush=True)
                assistant_started = True
            print(text, end="", flush=True)

        try:
            res = self.agent.run(
                user_message=user_input,
                conversation_history=self.history,
                on_token_callback=on_token,
                on_tool_callback=on_tool
            )

            final_text = res["final_answer"]
            if assistant_started:
                print()
            else:
                print(f"\n[Ассистент] > {final_text}")
            print(f"⏱️  (Время: {res['total_time']:.2f} сек, шагов: {len(res['steps'])})")

            # Update history
            self.history.append({"role": "user", "content": user_input})
            self.history.append({"role": "assistant", "content": final_text})

        except KeyboardInterrupt:
            print("\n[Прервано пользователем]")
        except Exception as e:
            print(f"\n[Ошибка агента]: {e}")
