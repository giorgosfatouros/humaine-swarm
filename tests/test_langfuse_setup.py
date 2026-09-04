import os
import unittest
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

from utils.langfuse_setup import (
    _configure_langfuse_env,
    langfuse_enabled,
    trace_chat_turn,
)


class DummyRoot:
    def __init__(self):
        self.updates = []

    def update(self, **kwargs):
        self.updates.append(kwargs)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class TestLangfuseSetup(unittest.TestCase):
    def test_langfuse_disabled_without_keys(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(langfuse_enabled())

    def test_host_copied_from_base_url(self):
        with patch.dict(
            os.environ,
            {"LANGFUSE_BASE_URL": "https://cloud.langfuse.com"},
            clear=True,
        ):
            _configure_langfuse_env()
            self.assertEqual(os.environ["LANGFUSE_HOST"], "https://cloud.langfuse.com")

    def test_strips_quotes_from_langfuse_env(self):
        with patch.dict(
            os.environ,
            {
                "LANGFUSE_BASE_URL": '"https://cloud.langfuse.com"',
                "LANGFUSE_PUBLIC_KEY": '"pk-lf-test"',
                "LANGFUSE_SECRET_KEY": "'sk-lf-test'",
            },
            clear=True,
        ):
            _configure_langfuse_env()
            self.assertEqual(
                os.environ["LANGFUSE_BASE_URL"], "https://cloud.langfuse.com"
            )
            self.assertEqual(os.environ["LANGFUSE_HOST"], "https://cloud.langfuse.com")
            self.assertEqual(os.environ["LANGFUSE_PUBLIC_KEY"], "pk-lf-test")

    def test_trace_chat_turn_noop_when_disabled(self):
        with patch.dict(os.environ, {}, clear=True):
            with trace_chat_turn("user-1", "thread-1", "hello") as root:
                self.assertIsNone(root)

    def test_trace_chat_turn_sets_root_io_and_string_metadata(self):
        dummy = DummyRoot()
        captured = {}

        @contextmanager
        def fake_propagate(**kwargs):
            captured.update(kwargs)
            yield

        fake_client = MagicMock()
        fake_client.start_as_current_observation.return_value = dummy

        with (
            patch.dict(
                os.environ,
                {
                    "LANGFUSE_PUBLIC_KEY": "pk-lf-test",
                    "LANGFUSE_SECRET_KEY": "sk-lf-test",
                },
                clear=False,
            ),
            patch("langfuse.get_client", return_value=fake_client),
            patch("langfuse.propagate_attributes", fake_propagate),
        ):
            with trace_chat_turn("user-1", "thread-1", "hello") as root:
                self.assertIs(root, dummy)
                root.update(output="world")

        fake_client.start_as_current_observation.assert_called_once_with(
            as_type="span",
            name="handle-chat-turn",
        )
        self.assertEqual(dummy.updates[0], {"input": "hello"})
        self.assertEqual(dummy.updates[1], {"output": "world"})
        self.assertEqual(captured["user_id"], "user-1")
        self.assertEqual(captured["session_id"], "thread-1")
        self.assertEqual(captured["trace_name"], "handle-chat-turn")
        self.assertEqual(captured["tags"], ["humaine-swarm"])
        self.assertEqual(captured["metadata"], {"source": "chainlit"})
        self.assertTrue(all(isinstance(v, str) for v in captured["metadata"].values()))

    def test_trace_chat_turn_omits_blank_session_id(self):
        dummy = DummyRoot()
        captured = {}

        @contextmanager
        def fake_propagate(**kwargs):
            captured.update(kwargs)
            yield

        fake_client = MagicMock()
        fake_client.start_as_current_observation.return_value = dummy

        with (
            patch.dict(
                os.environ,
                {
                    "LANGFUSE_PUBLIC_KEY": "pk-lf-test",
                    "LANGFUSE_SECRET_KEY": "sk-lf-test",
                },
                clear=False,
            ),
            patch("langfuse.get_client", return_value=fake_client),
            patch("langfuse.propagate_attributes", fake_propagate),
        ):
            with trace_chat_turn("user-1", None, "hello"):
                pass

        self.assertNotIn("session_id", captured)


if __name__ == "__main__":
    unittest.main()
