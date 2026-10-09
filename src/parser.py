"""
Text-based tool call parser.
Requirement 4:
- Парсинг вызовов функций прямо из тела ответа модели.
- Без использования OpenAI function calling / tools calling API / MCP.
- Поддержка различных вариантов синтаксиса от моделей <= 4B.
"""
import re
import ast
from typing import Optional, Tuple, Dict, Any


class ToolCallParser:
    """
    Robust parser for extracting tool calls and answers from LLM text responses.
    """

    # Pattern for Action: tool_name(...)
    ACTION_REGEX = re.compile(
        r"(?:Action|Вызов|Tool):\s*([a-zA-Z0-9_]+)\s*\((.*?)\)",
        re.IGNORECASE | re.DOTALL
    )

    # Secondary fallback pattern if model writes just tool_name(...) on a line
    STANDALONE_CALL_REGEX = re.compile(
        r"^\s*([a-zA-Z0-9_]+)\s*\((.*?)\)\s*$",
        re.MULTILINE
    )

    @classmethod
    def parse_arguments(cls, args_str: str) -> Dict[str, Any]:
        """
        Parses function arguments from string.
        Supports:
        - Named kwargs: city="Москва", pair='USD', expression="2 + 2"
        - Positional args: "Москва", 'USD', 2 + 2
        - Empty args: ""
        """
        args_str = args_str.strip()
        if not args_str:
            return {}

        kwargs = {}

        # 1. Try python ast parsing as dummy function call: f(...)
        try:
            tree = ast.parse(f"f({args_str})", mode='eval')
            call_node = tree.body
            if isinstance(call_node, ast.Call):
                # Named args
                for kw in call_node.keywords:
                    kwargs[kw.arg] = ast.literal_eval(kw.value)

                # Positional args (if no named args or in addition)
                if call_node.args and not kwargs:
                    # Collect positional arguments as indexed arg0, arg1 or single value
                    if len(call_node.args) == 1:
                        kwargs["_arg0"] = ast.literal_eval(call_node.args[0])
                    else:
                        for idx, arg in enumerate(call_node.args):
                            kwargs[f"_arg{idx}"] = ast.literal_eval(arg)

                if kwargs:
                    return kwargs
        except Exception:
            pass

        # 2. Regex fallback for key="value" or key='value'
        named_matches = re.findall(r'(\w+)\s*=\s*["\']([^"\']+)["\']', args_str)
        if named_matches:
            for k, v in named_matches:
                kwargs[k] = v
            return kwargs

        # 3. Regex fallback for key=value without quotes
        bare_matches = re.findall(r'(\w+)\s*=\s*([^,)]+)', args_str)
        if bare_matches:
            for k, v in bare_matches:
                kwargs[k] = v.strip().strip("'\"")
            return kwargs

        # 4. Fallback for single raw string / expression inside parentheses
        clean_raw = args_str.strip().strip("'\"")
        kwargs["_arg0"] = clean_raw
        return kwargs

    @classmethod
    def parse(cls, text: str) -> Tuple[Optional[str], Dict[str, Any], Optional[str], Optional[str]]:
        """
        Parses text and returns:
        (tool_name, kwargs, thought, answer)
        """
        thought = None
        answer = None

        # Extract thought if present
        thought_match = re.search(r"Thought:\s*(.*?)(?=\n(?:Action|Answer|Observation):|\Z)", text, re.IGNORECASE | re.DOTALL)
        if thought_match:
            thought = thought_match.group(1).strip()

        # Check for Answer
        answer_match = re.search(r"Answer:\s*(.*)", text, re.IGNORECASE | re.DOTALL)
        if answer_match:
            answer = answer_match.group(1).strip()

        # Search for Action call
        match = cls.ACTION_REGEX.search(text)
        if not match:
            # Try secondary match
            match = cls.STANDALONE_CALL_REGEX.search(text)

        if match:
            func_name = match.group(1).strip()
            args_raw = match.group(2).strip()
            kwargs = cls.parse_arguments(args_raw)
            return func_name, kwargs, thought, answer

        # No tool action found
        return None, {}, thought, answer
