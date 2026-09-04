"""Helpers for restoring persisted Chainlit threads into LLM session state."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import chainlit as cl

from classes.user_handler import UserSessionManager

logger = logging.getLogger(__name__)


def rebuild_llm_history_from_thread(
    thread: Dict[str, Any],
    system_content: str,
) -> List[Dict[str, str]]:
    """Rebuild OpenAI-style message history from persisted Chainlit thread steps."""
    history: List[Dict[str, str]] = [{"role": "system", "content": system_content}]
    for step in thread.get("steps") or []:
        step_type = step.get("type")
        output = (step.get("output") or "").strip()
        if not output:
            continue
        if step_type == "user_message":
            history.append({"role": "user", "content": output})
        elif step_type == "assistant_message":
            history.append({"role": "assistant", "content": output})
    return history


def needs_history_rebuild(message_history: List[Dict[str, Any]]) -> bool:
    """True when session has no usable LLM history beyond an optional system prompt."""
    if not message_history:
        return True
    non_system = [
        item
        for item in message_history
        if item.get("role") != "system"
    ]
    return len(non_system) == 0


async def refresh_oauth_for_user(user: cl.User) -> bool:
    """
    Refresh OAuth-derived session state from the live user metadata.

    Returns False when the OAuth token is expired and the session was cleared.
    """
    if not user.metadata or "oauth_token" not in user.metadata:
        return True

    UserSessionManager.set_oauth_token(user.metadata["oauth_token"])
    logger.info("Stored OAuth token for user %s", user.identifier)

    try:
        await UserSessionManager.fetch_and_store_minio_credentials()
        logger.info("Fetched MinIO credentials for user %s", user.identifier)
    except Exception as exc:
        error_msg = str(exc)
        logger.error("Failed to fetch MinIO credentials: %s", error_msg)
        if "expired" in error_msg.lower() or "token is expired" in error_msg.lower():
            logger.warning("OAuth token expired on resume - clearing session")
            UserSessionManager.clear_session()
            await cl.Message(
                content="Your session has expired. Please refresh the page to log in again.",
                author="System",
            ).send()
            return False

    try:
        UserSessionManager.extract_and_store_namespace()
        UserSessionManager.extract_and_store_token_info()
    except Exception as exc:
        logger.error("Failed to extract token information: %s", exc)

    return True
