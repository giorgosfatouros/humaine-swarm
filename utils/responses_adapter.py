import json
from typing import Any, Dict, List, Optional


def convert_tools_for_responses(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    converted: List[Dict[str, Any]] = []
    for tool in tools:
        if tool.get("type") == "function" and "function" in tool:
            fn = tool["function"]
            converted.append(
                {
                    "type": "function",
                    "name": fn["name"],
                    "description": fn.get("description"),
                    "parameters": fn.get("parameters"),
                    "strict": fn.get("strict", False),
                }
            )
        else:
            converted.append(tool)
    return converted


def extract_instructions(history: List[Dict[str, Any]]) -> Optional[str]:
    if history and history[0].get("role") == "system":
        content = history[0].get("content")
        return content if content else None
    return None


def make_function_call(call_id: str, name: str, arguments: str) -> Dict[str, Any]:
    return {
        "type": "function_call",
        "call_id": call_id,
        "name": name,
        "arguments": arguments,
    }


def make_function_call_output(call_id: str, result: Any) -> Dict[str, Any]:
    output = result if isinstance(result, str) else json.dumps(result)
    return {
        "type": "function_call_output",
        "call_id": call_id,
        "output": output,
    }


def _legacy_function_to_output(item: Dict[str, Any]) -> Dict[str, Any]:
    call_id = item.get("call_id") or item.get("name", "legacy")
    return {
        "type": "function_call_output",
        "call_id": call_id,
        "output": item.get("content", ""),
    }


def build_responses_input(history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for item in history:
        role = item.get("role")
        item_type = item.get("type")

        if role == "system":
            continue
        if role in ("user", "assistant"):
            items.append({"role": role, "content": item.get("content", "")})
        elif role == "function":
            items.append(_legacy_function_to_output(item))
        elif item_type in ("function_call", "function_call_output"):
            items.append(item)
    return items


def item_token_estimate(item: Dict[str, Any], encoding) -> int:
    text = item_text_for_tokens(item)
    if not text:
        return 0
    return len(encoding.encode(text))


def item_text_for_tokens(item: Dict[str, Any]) -> str:
    role = item.get("role")
    item_type = item.get("type")

    if role in ("user", "assistant", "system"):
        return item.get("content") or ""
    if item_type == "function_call":
        return f"{item.get('name', '')}{item.get('arguments', '')}"
    if item_type == "function_call_output":
        return item.get("output") or ""
    if role == "function":
        return item.get("content") or ""
    return json.dumps(item)
