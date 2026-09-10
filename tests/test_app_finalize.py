import unittest
from unittest.mock import AsyncMock, MagicMock

from app import finalize_assistant_message


class FakeMessage:
    def __init__(self, content="", streaming=False, persisted=False, elements=None):
        self.content = content
        self.streaming = streaming
        self.persisted = persisted
        self.elements = elements or []
        self.update = AsyncMock()
        self.send = AsyncMock()


class TestFinalizeAssistantMessage(unittest.IsolatedAsyncioTestCase):
    async def test_skips_empty_message(self):
        msg = FakeMessage(content="", streaming=False)
        await finalize_assistant_message(msg)
        msg.update.assert_not_called()
        msg.send.assert_not_called()

    async def test_sends_streamed_message_so_it_is_persisted(self):
        msg = FakeMessage(content="Hello", streaming=True)
        await finalize_assistant_message(msg)
        msg.send.assert_awaited_once()
        msg.update.assert_not_called()

    async def test_sends_non_streamed_message(self):
        msg = FakeMessage(content="Hello", streaming=False)
        await finalize_assistant_message(msg)
        msg.send.assert_awaited_once()
        msg.update.assert_not_called()

    async def test_updates_already_sent_message(self):
        msg = FakeMessage(content="Hello", streaming=True, persisted=True)
        await finalize_assistant_message(msg)
        msg.update.assert_awaited_once()
        msg.send.assert_not_called()

    async def test_sends_message_with_elements_only(self):
        msg = FakeMessage(content="", streaming=False, elements=[MagicMock()])
        await finalize_assistant_message(msg)
        msg.send.assert_awaited_once()
        msg.update.assert_not_called()


if __name__ == "__main__":
    unittest.main()
