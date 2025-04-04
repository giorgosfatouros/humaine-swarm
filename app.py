import asyncio
from openai import AsyncOpenAI
import chainlit as cl
from chainlit import User
from classes.user_handler import UserSessionManager
from starters import set_starters
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

# Process assistant's response stream and handle tool calls
async def process_stream(stream, message_history, msg):
    current_tool_call = None
    tool_calls_data = []

    async for part in stream:
        delta = part.choices[0].delta

        if delta.tool_calls:
            tool_call = delta.tool_calls[0]  # Handle the first tool call in the list
            logger.info(f"Tool call: {tool_call}")
            
            if tool_call.index is not None and tool_call.function.name:  # New tool call starting
                current_tool_call = {
                    "id": tool_call.id,
                    "type": tool_call.type,
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": ""
                    }
                }
                tool_calls_data.append(current_tool_call)
            
            if current_tool_call and tool_call.function.arguments:
                current_tool_call["function"]["arguments"] += tool_call.function.arguments

        elif delta.content:
            if current_tool_call:
                try:
                    # Only process if we have complete argument data
                    args = current_tool_call["function"]["arguments"]
                    if args.strip() and args[-1] == "}":  # Check if arguments string is complete
                        json.loads(args)  # Validate JSON
                        await process_function_call(
                            {
                                "name": current_tool_call["function"]["name"],
                                "arguments": args
                            },
                            message_history,
                            msg
                        )
                        current_tool_call = None
                except json.JSONDecodeError:
                    # Continue accumulating if JSON is incomplete
                    pass
            await msg.stream_token(delta.content)

    # Process any remaining tool call after the stream ends
    if current_tool_call and current_tool_call["function"]["name"]:  # Only process if we have a valid function name
        try:
            args = current_tool_call["function"]["arguments"]
            json.loads(args)  # Validate JSON
            logger.info(f"Processing final tool call: {current_tool_call}")
            await process_function_call(
                {
                    "name": current_tool_call["function"]["name"],
                    "arguments": args
                },
                message_history,
                msg
            )
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in final tool call: {e}")

# @lai.step(name="Function Processor", type="tool")
@cl.step(name="Agent Processor", type="tool")
async def process_function_call(function_call_data, message_history, msg):
    logger.info(f"Function call data: {function_call_data}")
    try:
        function_name = function_call_data["name"]
        arguments = json.loads(function_call_data["arguments"])
        
        logger.debug(f"Function call data: {function_call_data}")
        
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
@cl.step(name="HumAIne Swarm", type="run", show_input=False)
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
    if msg.content.strip():
        message_history.append({"role": "assistant", "content": msg.content})
        # plotly_figure = await generate_plotly_figure(msg.content)
        # if plotly_figure:
        #     msg.elements = plotly_figure
            
    await msg.update()

    # Save the updated message history in the session
    UserSessionManager.set_message_history(message_history)

    # Log message details for debugging
    total_length = sum(len(json.dumps(msg)) for msg in message_history)
    logger.info(f"Message finished with {UserSessionManager.get_function_call_count()} function calls and {len(message_history)} messages in history with total length of {total_length} characters")

    # Generate follow-up questions for the user
    # follow_up_questions = await generate_follow_up_questions(message_history)
    
    # Send the follow-up questions to the UI
    # await send_follow_up_questions(follow_up_questions)
