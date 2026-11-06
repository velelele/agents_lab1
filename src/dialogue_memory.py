from typing import Any, Dict
from langchain_classic.memory import ConversationBufferWindowMemory, ConversationSummaryMemory
from langchain_core.prompts import ChatPromptTemplate
from src.langchain_llm import LocalLlamaChatModel


SUMMARY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Кратко перескажи предыдущую беседу. Обнови сводку с учётом нового обмена. Сохрани имена, предпочтения, цели и важные факты из обоих блоков. Не добавляй новых сведений. Верни только сводку на русском языке."),
    ("human", "Существующая сводка: {summary}\n\nОбмен, который нужно добавить в сводку:\n{new_lines}"),
])


class DialogueMemory:
    def __init__(self, llm: Any, window_size: int = 2):
        if window_size < 1:
            raise ValueError("Размер окна должен быть больше нуля")
        self.llm = llm
        self.window_size = window_size
        self.window_memory = ConversationBufferWindowMemory(
            k=window_size, return_messages=True, input_key="input", output_key="output"
        )
        self.summary_memory = ConversationSummaryMemory(
            llm=llm if llm is not None else LocalLlamaChatModel(model_alias="qwen"),
            prompt=SUMMARY_PROMPT,
            input_key="input",
            output_key="output",
        )

    @property
    def summary(self) -> str:
        return self.summary_memory.buffer

    @property
    def turns(self):
        messages = self.window_memory.load_memory_variables({})["history"]
        return [
            {"user": messages[i].content, "assistant": messages[i + 1].content}
            for i in range(0, len(messages), 2)
        ]

    def save_context(self, user_message: str, answer: str) -> None:
        messages = self.window_memory.chat_memory.messages
        if len(messages) >= self.window_size * 2:
            old_user, old_answer = messages[0].content, messages[1].content
            previous = self.summary_memory.buffer
            try:
                if self.llm is None:
                    raise ValueError("Модель суммаризации не подключена")
                self.summary_memory.save_context({"input": old_user}, {"output": old_answer})
                if not self.summary_memory.buffer.strip():
                    raise ValueError("Модель вернула пустую суммаризацию")
            except Exception:
                self.summary_memory.buffer = "\n".join(
                    part for part in [previous, f"Пользователь: {old_user}\nАссистент: {old_answer}"] if part
                )
            finally:
                self.summary_memory.chat_memory.clear()
            del messages[:2]

        self.window_memory.save_context({"input": user_message}, {"output": answer})

    def load_memory_variables(self) -> Dict[str, Any]:
        return {"summary": self.summary_memory.load_memory_variables({})["history"], "window": self.turns}

    def format_context(self) -> str:
        parts = []
        if self.summary:
            parts.append("Краткое содержание предыдущего диалога:\n" + self.summary)
        if self.turns:
            lines = []
            for turn in self.turns:
                lines.extend([f"Пользователь: {turn['user']}", f"Ассистент: {turn['assistant']}"])
            parts.append("Последние сообщения:\n" + "\n".join(lines))
        return "\n\n".join(parts)

    def clear(self) -> None:
        self.window_memory.clear()
        self.summary_memory.clear()
