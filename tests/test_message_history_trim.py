"""Tests for message-history trimming: tool calls and outputs must stay paired."""

from unittest.mock import MagicMock, patch

import tiktoken

from classes.user_handler import UserSessionManager
from utils import config
from utils.responses_adapter import (
    drop_orphan_tool_items,
    item_token_estimate,
    make_function_call,
    make_function_call_output,
)


def _turn(call_id, output_text):
    return [
        make_function_call(call_id, "get_model_metrics", '{"run_name": "r1"}'),
        make_function_call_output(call_id, output_text),
    ]


def _trim(history, budget):
    session = MagicMock()
    with patch("classes.user_handler.MAX_INPUT_TOKENS", budget), patch(
        "classes.user_handler.cl.user_session", session
    ):
        UserSessionManager.set_message_history(history)
    return session.set.call_args.args[1]


def test_trim_never_keeps_an_output_without_its_call():
    history = [
        {"role": "system", "content": "System"},
        {"role": "user", "content": "show me the results"},
        *_turn("call_A", "metrics " * 50),
        {"role": "assistant", "content": "Here are the results."},
    ]
    # Exactly enough room for the assistant reply and the large output, not for the call before it
    encoding = tiktoken.get_encoding("o200k_base")
    budget = sum(item_token_estimate(item, encoding) for item in history[-2:])

    kept = _trim(history, budget)

    assert kept[0]["role"] == "system"
    call_ids = {i["call_id"] for i in kept if i.get("type") == "function_call"}
    output_ids = {i["call_id"] for i in kept if i.get("type") == "function_call_output"}
    assert output_ids <= call_ids
    assert call_ids <= output_ids


def test_trim_keeps_complete_pairs():
    history = [
        {"role": "system", "content": "System"},
        {"role": "user", "content": "show me the results"},
        *_turn("call_A", "ok"),
        {"role": "assistant", "content": "Done."},
    ]

    kept = _trim(history, 10_000)

    assert kept == history


def test_drop_orphan_tool_items_removes_unpaired_calls_and_outputs():
    items = [
        make_function_call_output("call_orphan_output", "x"),
        *_turn("call_ok", "y"),
        make_function_call("call_orphan_call", "get_docs", "{}"),
        {"role": "user", "content": "hi"},
    ]

    assert drop_orphan_tool_items(items) == [*_turn("call_ok", "y"), {"role": "user", "content": "hi"}]


def test_history_budget_leaves_room_for_the_response():
    assert config.MAX_INPUT_TOKENS == config.MAX_CONTEXT_LENGTH - config.OUTPUT_TOKEN_RESERVE
    assert config.MAX_INPUT_TOKENS > 10_000
