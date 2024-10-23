from openai import AsyncOpenAI
# from literalai import AsyncLiteralClient
import asyncio
from typing import Dict, Any, List, Optional
import json
import chainlit as cl
from agents.code import function_map
from utils.helper_functions import setup_logging
import logging

logger = setup_logging('API-FUNCTIONS', level=logging.INFO)

client = AsyncOpenAI()
# lai = AsyncLiteralClient()




async def execute_function(function_name: str, arguments: Dict):
    func = function_map[function_name]
    is_async = asyncio.iscoroutinefunction(func)

    # Use lai.step context manager
    # with lai.step(name=function_name, type="tool") as step:
    if is_async:
        result = await func(**arguments)
    else:
        result = func(**arguments)

    # step.metadata = {"arguments": arguments}
    # step.tags = ["function_call"]

    return result


async def handle_function_error(error: Exception, function_name: str, arguments: Optional[Dict], msg):
    logger.error(f"Error in {function_name}: {str(error)}")
    user_friendly_message = "I'm sorry, but I encountered an issue while processing your request."
    if isinstance(error, json.JSONDecodeError):
        user_friendly_message = "There was an error parsing the function arguments. Please ensure your input is correct."
    new_msg = cl.Message(content=user_friendly_message)
    await new_msg.send()




def update_function_call_data(function_call, function_call_data):
    if function_call.name:
        function_call_data["name"] = function_call.name
    if function_call.arguments:
        function_call_data["arguments"] += function_call.arguments

def is_valid_json(data):
    try:
        json.loads(data)
        return True
    except json.JSONDecodeError:
        return False
    

# Helper function to send follow-up questions
async def send_follow_up_questions(follow_up_questions):
    if follow_up_questions:
        actions = [cl.Action(name="follow_up_question", value=question, label=question) for question in follow_up_questions]
        actions_msg = cl.Message(content="**Follow-up Questions:**", actions=actions, disable_feedback=True)
        await actions_msg.send()


# Send an error message
async def send_error_message(error_message, msg):
    logger.warning(error_message)
    await msg.update(content=error_message, )