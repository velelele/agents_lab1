import re
import ast
from typing import Optional, Tuple, Dict, Any

class ToolCallParser:
    ACTION_REGEX = re.compile(
        r"(?:Action|Вызов|Tool):\s*([a-zA-Z0-9_]+)\s*\((.*?)\)",
        re.IGNORECASE | re.DOTALL
    )

    STANDALONE_CALL_REGEX = re.compile(
        r"^\s*([a-zA-Z0-9_]+)\s*\((.*?)\)\s*$",
        re.MULTILINE
    )

    @classmethod
    def parse_arguments(cls, args_str: str) -> Dict[str, Any]:
        args_str = args_str.strip()
        if not args_str:
            return {}

        kwargs = {}

        try:
            tree = ast.parse(f"f({args_str})", mode='eval')
            call_node = tree.body
            if isinstance(call_node, ast.Call):
                for kw in call_node.keywords:
                    kwargs[kw.arg] = ast.literal_eval(kw.value)

                if call_node.args and not kwargs:
                    if len(call_node.args) == 1:
                        kwargs["_arg0"] = ast.literal_eval(call_node.args[0])
                    else:
                        for idx, arg in enumerate(call_node.args):
                            kwargs[f"_arg{idx}"] = ast.literal_eval(arg)

                if kwargs:
                    return kwargs
        except Exception:
            pass

        named_matches = re.findall(r'(\w+)\s*=\s*["\']([^"\']+)["\']', args_str)
        if named_matches:
            for k, v in named_matches:
                kwargs[k] = v
            return kwargs

        bare_matches = re.findall(r'(\w+)\s*=\s*([^,)]+)', args_str)
        if bare_matches:
            for k, v in bare_matches:
                kwargs[k] = v.strip().strip("'\"")
            return kwargs

        clean_raw = args_str.strip().strip("'\"")
        kwargs["_arg0"] = clean_raw
        return kwargs

    @classmethod
    def parse(cls, text: str) -> Tuple[Optional[str], Dict[str, Any], Optional[str], Optional[str]]:
        thought = None
        answer = None

        thought_match = re.search(r"Thought:\s*(.*?)(?=\n(?:Action|Answer|Observation):|\Z)", text, re.IGNORECASE | re.DOTALL)
        if thought_match:
            thought = thought_match.group(1).strip()

        answer_match = re.search(r"Answer:\s*(.*)", text, re.IGNORECASE | re.DOTALL)
        if answer_match:
            answer = answer_match.group(1).strip()

        match = cls.ACTION_REGEX.search(text)
        if not match:
            match = cls.STANDALONE_CALL_REGEX.search(text)

        if match:
            func_name = match.group(1).strip()
            args_raw = match.group(2).strip()
            kwargs = cls.parse_arguments(args_raw)
            return func_name, kwargs, thought, answer

        return None, {}, thought, answer
