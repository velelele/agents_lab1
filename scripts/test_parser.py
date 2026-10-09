import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.parser import ToolCallParser
from src.tools import execute_tool


tests = [
    'Thought: Нужно узнать погоду.\nAction: get_weather(city="Москва")',
    'Thought: Курс валюты.\nAction: get_exchange_rate(pair="USD")',
    'Action: calculate(expression="2 + 2 * 2")',
    'Action: get_current_time()',
    'Thought: Посчитаем\nAction: calculate("100 / 4")',
    'Answer: Привет! Я ассистент.'
]

for t in tests:
    fn, kw, th, ans = ToolCallParser.parse(t)
    obs = execute_tool(fn, **kw) if fn else "NO_TOOL"
    print(f"FN: {fn} | KW: {kw} | OBS: {obs} | ANS: {ans}")
