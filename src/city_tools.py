import re
from typing import Any, Type
from pydantic import BaseModel, Field, field_validator
from langchain_core.tools import BaseTool


class CityInput(BaseModel):
    city: str = Field(description="Название города пользователя", min_length=2, max_length=80)

    @field_validator("city")
    @classmethod
    def validate_city(cls, value: str) -> str:
        city = " ".join(value.strip().split())
        if not re.fullmatch(r"[А-Яа-яЁёA-Za-z]+(?:[ -][А-Яа-яЁёA-Za-z]+)*", city):
            raise ValueError("Укажите название города буквами")
        return city


class SaveUserCity(BaseTool):
    name: str = "save_user_city"
    description: str = "Сохранить город пользователя для дальнейших запросов погоды"
    args_schema: Type[BaseModel] = CityInput
    memory: Any = Field(exclude=True)

    def _run(self, city: str) -> str:
        self.memory.load_from_file()
        self.memory.facts["city"] = city
        self.memory.save_to_file()
        self.memory.load_from_file()
        if self.memory.facts.get("city") != city:
            raise OSError("Не удалось сохранить город в файл памяти")
        return f"Город сохранён: {city}"


def extract_requested_city(text: str):
    match = re.fullmatch(
        r"\s*(?:запомни|сохрани|установи)\s+(?:мой\s+)?город\s*(?::|—|-|это|как)\s*([А-Яа-яЁёA-Za-z]+(?:[ -][А-Яа-яЁёA-Za-z]+)*)[.!]?\s*",
        text,
        re.IGNORECASE,
    )
    return match.group(1).strip() if match else None
