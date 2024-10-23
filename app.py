import asyncio
from openai import AsyncOpenAI
import chainlit as cl
from chainlit import User
from classes.user_handler import UserSessionManager
import json
import logging
from utils.starters import set_starters
from agents.code import function_map, read_prompt
from utils.helper_functions import setup_logging, decode_jwt, extract_token_from_headers, extract_user_from_payload
from utils.api_functions import send_follow_up_questions
from utils.config import settings
# from literalai import AsyncLiteralClient
from typing import Optional, Dict
from chainlit.types import ThreadDict

# Setup logging
logger = setup_logging('CHAT', level=logging.ERROR)
logging.getLogger("httpx").setLevel("WARNING")

client = AsyncOpenAI()
# lai = AsyncLiteralClient()
# lai.instrument_openai()

system_prompt = read_prompt('system')

# Action callback to handle follow-up questions
@cl.action_callback("follow_up_question")
async def on_follow_up_question(action):
    await main(cl.Message(content=action.value))
    await action.remove()

@cl.password_auth_callback
async def auth_callback(username: str, password: str):
    # Fetch the user matching username from your database
    # and compare the hashed password with the value stored in the database
    # if (username, password) == ("george", "admin"):
    return cl.User(
            identifier="gfatouros@innov-acts.com", metadata={"role": "admin", "email": "gfatouros@innov-acts.com", "provider": "credentials"}
        )
  

# Chat initialization
@cl.on_chat_start
async def start_chat():
    # Initialize the system prompt message
    init_message = [{"role": "system", "content": system_prompt}]
    UserSessionManager.set_message_history(init_message)

    # Retrieve the user from the session
    user = cl.user_session.get("user")

    # Set the user ID in the session and in LiterAI
    UserSessionManager.set_user_id(user.identifier)
    # participant = await lai.api.get_or_create_user(identifier=user.identifier)

    # Print stored info for debugging
    logger.info(UserSessionManager.print_stored_info())

@cl.on_chat_resume
async def on_chat_resume(thread: ThreadDict):
    thread_id = thread['id']
    logger.info(f"Thread ID on resume: {thread_id}")

# Process assistant's response stream and handle function calls
async def process_stream(stream, message_history, msg):
    function_call_data = None

    async for part in stream:
        delta = part.choices[0].delta

        if delta.function_call:
            if function_call_data is None:
                function_call_data = {"name": delta.function_call.name, "arguments": ""}
            function_call_data["arguments"] += delta.function_call.arguments or ""
        elif delta.content:
            if function_call_data:
                try:
                    json.loads(function_call_data["arguments"])
                    await process_function_call(function_call_data, message_history, msg)
                    function_call_data = None
                except json.JSONDecodeError:
                    pass
            await msg.stream_token(delta.content)

    if function_call_data:
        await process_function_call(function_call_data, message_history, msg)

# @lai.step(name="Function Processor", type="tool")
async def process_function_call(function_call_data, message_history, msg):
    try:
        function_name = function_call_data["name"]
        arguments = json.loads(function_call_data["arguments"])
        
        if function_name not in function_map:
            await msg.stream_token(f"Unknown function: {function_name}\n")
            return
        

        logger.info(f"Executing function: {function_name} with arguments: {arguments}")
        
        func = function_map[function_name]
        result = await func(**arguments) if asyncio.iscoroutinefunction(func) else func(**arguments)

        UserSessionManager.increment_function_call_count()

        message_history.append({"role": "function", "name": function_name, "content": json.dumps(result)})
        UserSessionManager.set_message_history(message_history)

        follow_up_stream = await client.chat.completions.create(messages=message_history, **settings)

        await process_stream(follow_up_stream, message_history, msg)
        await msg.send()

    except Exception as e:
        logger.error(f"Error in {function_name}: {str(e)}")
        await msg.stream_token("We are currently experiencing high traffic. Please try again later.\n")
        await msg.send()

# Main function that handles user messages
@cl.on_message
async def main(message: cl.Message):
    # Increment the function call count for the current session
    UserSessionManager.increment_function_call_count()
    
    thread_id = UserSessionManager.get_thread_id()

    # async with lai.run(name="Assistant Response", thread_id=thread_id):
    # Send an empty message to acknowledge receipt
    msg = cl.Message(content="")
    await msg.send()

    logger.info(f"New message from user {UserSessionManager.get_user_id()}")

    # Get the message history and append the new user message
    message_history = UserSessionManager.get_message_history()
    message_history.append({"role": "user", "content": message.content})

    # Create the OpenAI chat completion with the message history
    completion = await client.chat.completions.create(messages=message_history, **settings)

    # Copy the message history for further use
    temp_history = message_history.copy()

    # Process the completion stream, attaching it to the current session and message
    await process_stream(completion, temp_history, msg)
    await msg.update()
    
    # Only append the assistant's response if it's not empty
    if msg.content.strip():
        message_history.append({"role": "assistant", "content": msg.content})

    # Save the updated message history in the session
    UserSessionManager.set_message_history(message_history)

    # Log message details for debugging
    total_length = sum(len(json.dumps(msg)) for msg in message_history)
    logger.info(f"Message finished with {UserSessionManager.get_function_call_count()} function calls and {len(message_history)} messages in history with total length of {total_length} characters")
