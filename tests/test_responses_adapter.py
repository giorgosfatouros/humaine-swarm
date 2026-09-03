import tiktoken

from utils.responses_adapter import (
    build_responses_input,
    convert_tools_for_responses,
    extract_instructions,
    item_token_estimate,
    make_function_call_output,
)


def test_convert_tools_for_responses_flattens_chat_completion_shape():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_docs",
                "description": "Search docs",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    converted = convert_tools_for_responses(tools)
    assert converted == [
        {
            "type": "function",
            "name": "get_docs",
            "description": "Search docs",
            "parameters": {"type": "object", "properties": {}},
            "strict": False,
        }
    ]


def test_extract_instructions():
    history = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hi"},
    ]
    assert extract_instructions(history) == "You are helpful."


def test_build_responses_input_skips_system_and_maps_messages():
    history = [
        {"role": "system", "content": "System"},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
        {
            "type": "function_call",
            "call_id": "call_1",
            "name": "get_docs",
            "arguments": '{"query": "test"}',
        },
        {
            "type": "function_call_output",
            "call_id": "call_1",
            "output": '{"ok": true}',
        },
    ]
    items = build_responses_input(history)
    assert items == [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
        {
            "type": "function_call",
            "call_id": "call_1",
            "name": "get_docs",
            "arguments": '{"query": "test"}',
        },
        {
            "type": "function_call_output",
            "call_id": "call_1",
            "output": '{"ok": true}',
        },
    ]


def test_build_responses_input_migrates_legacy_function_role():
    history = [
        {"role": "system", "content": "System"},
        {"role": "function", "name": "get_docs", "content": '{"ok": true}'},
    ]
    items = build_responses_input(history)
    assert items == [
        {
            "type": "function_call_output",
            "call_id": "get_docs",
            "output": '{"ok": true}',
        }
    ]


def test_make_function_call_output_serializes_dict():
    item = make_function_call_output("call_1", {"ok": True})
    assert item["type"] == "function_call_output"
    assert item["call_id"] == "call_1"
    assert item["output"] == '{"ok": true}'


def test_item_token_estimate_handles_function_items():
    encoding = tiktoken.get_encoding("o200k_base")
    function_call = {
        "type": "function_call",
        "call_id": "call_1",
        "name": "get_docs",
        "arguments": '{"query": "test"}',
    }
    assert item_token_estimate(function_call, encoding) > 0
