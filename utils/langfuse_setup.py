"""Optional Langfuse tracing for the main OpenAI Responses API path."""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Iterator, Optional

from openai import AsyncOpenAI

_MAX_ATTR_LEN = 200


def _configure_langfuse_env() -> None:
    base_url = os.environ.get("LANGFUSE_BASE_URL")
    if base_url and not os.environ.get("LANGFUSE_HOST"):
        os.environ["LANGFUSE_HOST"] = base_url


def langfuse_enabled() -> bool:
    return bool(
        os.environ.get("LANGFUSE_PUBLIC_KEY")
        and os.environ.get("LANGFUSE_SECRET_KEY")
    )


def _attr(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return text[:_MAX_ATTR_LEN]


_configure_langfuse_env()


def get_openai_client() -> AsyncOpenAI:
    if langfuse_enabled():
        from langfuse.openai import AsyncOpenAI as LangfuseAsyncOpenAI

        return LangfuseAsyncOpenAI()
    return AsyncOpenAI()


@contextmanager
def trace_chat_turn(
    user_id: Optional[str],
    session_id: Optional[str],
    user_input: str,
) -> Iterator[Optional[Any]]:
    """One root observation per chat turn; correlating attrs apply to child generations."""
    if not langfuse_enabled():
        yield None
        return

    from langfuse import get_client, propagate_attributes

    langfuse = get_client()
    uid = _attr(user_id) or "anonymous"
    sid = _attr(session_id)
    metadata = {"source": "chainlit"}
    propagate_kwargs: dict[str, Any] = {
        "trace_name": "handle-chat-turn",
        "user_id": uid,
        "tags": ["humaine-swarm"],
        "metadata": metadata,
    }
    if sid:
        propagate_kwargs["session_id"] = sid

    with langfuse.start_as_current_observation(
        as_type="span",
        name="handle-chat-turn",
    ) as root:
        root.update(input=user_input)
        with propagate_attributes(**propagate_kwargs):
            yield root
