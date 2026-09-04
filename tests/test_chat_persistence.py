import unittest

from utils.chat_persistence import (
    needs_history_rebuild,
    rebuild_llm_history_from_thread,
)


class TestChatPersistence(unittest.TestCase):
    def test_rebuild_llm_history_from_thread(self):
        thread = {
            "id": "thread-1",
            "steps": [
                {"type": "user_message", "output": "Hello"},
                {"type": "assistant_message", "output": "Hi there"},
                {"type": "user_message", "output": "What is HAIC?"},
                {"type": "assistant_message", "output": "HAIC is a benchmark suite."},
                {"type": "run", "output": "ignored"},
                {"type": "user_message", "output": ""},
            ],
        }
        history = rebuild_llm_history_from_thread(thread, "You are helpful.")
        self.assertEqual(
            history,
            [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"},
                {"role": "user", "content": "What is HAIC?"},
                {"role": "assistant", "content": "HAIC is a benchmark suite."},
            ],
        )

    def test_needs_history_rebuild_empty(self):
        self.assertTrue(needs_history_rebuild([]))

    def test_needs_history_rebuild_system_only(self):
        self.assertTrue(
            needs_history_rebuild([{"role": "system", "content": "prompt"}])
        )

    def test_needs_history_rebuild_with_messages(self):
        self.assertFalse(
            needs_history_rebuild(
                [
                    {"role": "system", "content": "prompt"},
                    {"role": "user", "content": "hello"},
                ]
            )
        )


if __name__ == "__main__":
    unittest.main()
