import tiktoken
import chainlit as cl
from utils.helper_functions import setup_logging
from utils.config import settings, MAX_INPUT_TOKENS
import logging
import uuid
# from literalai import AsyncLiteralClient

# lai = AsyncLiteralClient()

logger = setup_logging('USER', level=logging.ERROR)

# Helper class for managing user session state
class UserSessionManager:
    @staticmethod
    def set_user_id(user_id):
        cl.user_session.set("user_id", user_id)
        logger.info(f"Set user ID: {user_id}")

    @staticmethod
    def get_user_id():
        app_user = cl.user_session.get("user")
        user_id = app_user.metadata["email"]
        logger.info(f"Retrieved user ID: {user_id}")  
        return user_id
    
    @staticmethod
    def get_message_history() -> list:
        message_history = cl.user_session.get("message_history", [])
        if not isinstance(message_history, list):
            return []
        return message_history

    @staticmethod
    def increment_function_call_count():
        count = cl.user_session.get("function_call_count", 0)
        cl.user_session.set("function_call_count", (count or 0) + 1)

    @staticmethod
    def reset_function_call_count():
        cl.user_session.set("function_call_count", 0)

    @staticmethod
    def get_function_call_count() -> int:
        return cl.user_session.get("function_call_count", 0) or 0

    @staticmethod
    def get_thread_id():
        return cl.context.session.thread_id

    # @staticmethod
    # async def get_literalai_user():
    #     user_id = UserSessionManager.get_user_id()
    #     try:
    #         user = await lai.api.get_or_create_user(identifier=user_id)
    #         logger.info(f"Fetched or created user: {user_id}")
    #         return user
    #     except Exception as e:
    #         logger.error(f"Failed to fetch or create user: {e}")
    #         return {"error": str(e)}

    @staticmethod
    def set_message_history(message_history):
        encoding = tiktoken.encoding_for_model(settings["model"])
        total_tokens = 0
        truncated_messages = []
        system_message = message_history[0] if message_history and message_history[0]['role'] == 'system' else None

        for message in reversed(message_history[1:]):
            message_tokens = len(encoding.encode(message.get('content', '') or ''))
            total_tokens += message_tokens
            if total_tokens > MAX_INPUT_TOKENS:
                break
            truncated_messages.insert(0, message)
        
        if system_message:
            truncated_messages.insert(0, system_message)
        
        cl.user_session.set("message_history", truncated_messages)

    @staticmethod
    def print_stored_info():
        user_id = UserSessionManager.get_user_id()
        function_call_count = UserSessionManager.get_function_call_count()
        thread_id = UserSessionManager.get_thread_id()
        logger.info(f"User ID: {user_id}")
        logger.info(f"Function Call Count: {function_call_count}")
        logger.info(f"Thread ID: {thread_id}")
