import os
import json
from typing import Dict, Any, List, Optional
from pydantic import Field

try:
    from langchain.memory.chat_memory import BaseChatMemory
except ImportError:
    from langchain_classic.memory.chat_memory import BaseChatMemory

from src.display_cell import render_cell
from src.fact_extractor import FactExtractor

FACT_KEY_LABELS = {
    "name": "Имя",
    "role": "Профессия / Роль",
    "stack": "Технологический стек",
    "level": "Грейд / Уровень",
    "city": "Город",
    "goals": "Цели / Задачи",
    "preferences": "Предпочтения"
}

class FileFactMemory(BaseChatMemory):
    file_path: str = "memory.json"
    session_id: str = "default_user"
    memory_key: str = "facts"
    facts: Dict[str, Any] = Field(default_factory=dict)
    extractor: Optional[Any] = Field(default=None, exclude=True)
    show_service_cell: bool = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.extractor is None:
            self.extractor = FactExtractor()
        self.load_from_file()

    @property
    def memory_variables(self) -> List[str]:
        return [self.memory_key]

    def _read_all_sessions(self) -> Dict[str, Any]:
        if not os.path.exists(self.file_path):
            return {}
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if not content:
                    return {}
                data = json.loads(content)
                if isinstance(data, dict):
                    return data
        except Exception:
            return {}
        return {}

    def _write_all_sessions(self, data: Dict[str, Any]) -> None:
        try:
            with open(self.file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def load_from_file(self) -> Dict[str, Any]:
        all_data = self._read_all_sessions()
        session_data = all_data.get(self.session_id, {})
        if isinstance(session_data, dict):
            self.facts = dict(session_data)
        else:
            self.facts = {}
        return self.facts

    def save_to_file(self) -> None:
        all_data = self._read_all_sessions()
        all_data[self.session_id] = self.facts
        self._write_all_sessions(all_data)

    def format_facts_string(self) -> str:
        if not self.facts:
            return "Факты о пользователе пока не известны."
        lines = []
        for key, value in self.facts.items():
            label = FACT_KEY_LABELS.get(key, key)
            lines.append(f"- {label}: {value}")
        return "\n".join(lines)

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        self.load_from_file()
        return {
            self.memory_key: self.format_facts_string(),
            "facts_dict": dict(self.facts)
        }

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> None:
        user_text = ""
        if isinstance(inputs, dict):
            user_text = str(inputs.get("input") or inputs.get("user_message") or inputs.get("question") or "")
        elif isinstance(inputs, str):
            user_text = inputs

        if not user_text.strip():
            return

        new_facts = {}
        if self.extractor:
            new_facts = self.extractor.extract(user_text)

        if new_facts:
            updated_keys = []
            for k, v in new_facts.items():
                if v and self.facts.get(k) != v:
                    self.facts[k] = v
                    updated_keys.append(k)

            if updated_keys:
                self.save_to_file()
                if self.show_service_cell:
                    summary_dict = {
                        "Сессия": self.session_id,
                        "Файл": self.file_path,
                        "Новые факты": ", ".join(f"{FACT_KEY_LABELS.get(k, k)}={self.facts[k]}" for k in updated_keys),
                        "Всего сохранено фактов": len(self.facts)
                    }
                    render_cell(
                        title="СОХРАНЕНИЕ ФАКТОВ В ПАМЯТЬ ФАЙЛА",
                        content=summary_dict,
                        border_style="bold green",
                        subtitle=f"Сохранено в {self.file_path}"
                    )

    def clear(self) -> None:
        self.facts = {}
        all_data = self._read_all_sessions()
        if self.session_id in all_data:
            del all_data[self.session_id]
            self._write_all_sessions(all_data)
