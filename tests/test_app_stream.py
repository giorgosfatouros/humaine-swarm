import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import app
from app import process_responses_stream


class FakeMessage:
    """Mirrors chainlit.Message.stream_token, which appends the token to content."""

    def __init__(self):
        self.content = ""
        self.streaming = False
        self.elements = []

    async def stream_token(self, token, is_sequence=False):
        if not token:
            return
        if is_sequence:
            self.content = token
        else:
            self.content += token
        self.streaming = True


async def text_only_stream(deltas):
    for delta in deltas:
        yield SimpleNamespace(type="response.output_text.delta", delta=delta)


async def tool_call_stream():
    item = SimpleNamespace(
        type="function_call",
        id="item_1",
        call_id="call_1",
        name="get_docs",
        arguments='{"query": "paradigms"}',
    )
    yield SimpleNamespace(type="response.output_item.added", item=item)


class TestProcessResponsesStream(unittest.IsolatedAsyncioTestCase):
    async def test_streamed_text_is_not_duplicated(self):
        msg = FakeMessage()
        await process_responses_stream(
            text_only_stream(["Hello", " ", "world"]), [], msg
        )
        self.assertEqual(msg.content, "Hello world")

    async def test_follow_up_text_after_tool_call_is_not_duplicated(self):
        msg = FakeMessage()

        async def fake_create_response_stream(message_history):
            return text_only_stream(["Answer", " text"])

        with (
            patch.object(app, "create_response_stream", fake_create_response_stream),
            patch.object(app, "function_map", {"get_docs": lambda query: "docs"}),
            patch.object(app, "is_tool_enabled", lambda name, packs: True),
            patch.object(app, "get_enabled_pack_ids", lambda: []),
            patch.object(app, "UserSessionManager", MagicMock()),
        ):
            await process_responses_stream(tool_call_stream(), [], msg)

        self.assertEqual(msg.content, "Answer text")


if __name__ == "__main__":
    unittest.main()
