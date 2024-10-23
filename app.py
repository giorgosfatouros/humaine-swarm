import asyncio
import logging
from openai import AsyncOpenAI
import chainlit as cl
from chainlit import User
from typing import Dict, Optional
from utils.config import read_prompt
from agents.code import function_map
from agents.definition import functions
import os, json

client = AsyncOpenAI()

settings = {
    "model": "gpt-4o-2024-08-06",
    "temperature": 0,
    "max_tokens": 8000,
    "top_p": 1,
    "frequency_penalty": 0,
    "presence_penalty": 0,
    # "parallel_tool_calls": True,
    # "tools": functions,
    # "tool_choice": "auto",    
    "functions": functions,
    "function_call": "auto",
    "stream": True
}

system_prompt = read_prompt("system")



# @cl.oauth_callback
# def oauth_callback(
#     provider_id: str,
#     token: str,
#     raw_user_data: Dict[str, str],
#     default_user: cl.User):
#     logging.info(f"OAuth callback called with provider_id: {provider_id}, token: {token}, raw_user_data: {raw_user_data}")
#     logging.info(f"Default user: {default_user.identifier}")
#     return default_user


@cl.on_chat_start
async def start_chat():
    cl.user_session.set("function_call_count", 0)
    cl.user_session.set("message_history", [])
    init_message = [{"role": "system", "content": system_prompt}]
    cl.user_session.set("message_history", init_message)    
    await upload_file(show_welcome=True)



async def upload_file(show_welcome=False):
    if show_welcome:
        app_user = cl.user_session.get('user')
        image_path = "public/wealthcraft.png"  # Adjust the path as needed
        welcome_image = cl.Image(name="WealthCraft+ Logo", path=image_path, display="page", size="small")

        await cl.Message(
            content="",
            elements=[welcome_image],
        ).send()

        welcome_message = f"""
### 👋 Welcome to **WealthCraft+** 👋
Your advanced AI assistant for crafting professional, MIFID compliant, and insightful investment proposals. \n\n
I am here to streamline your workflow and support you in delivering tailored financial strategies for high-net-worth clients. Simply upload the portfolio statement, and I'll take care of the analysis, gathering relevant market data, and providing a comprehensive overview. Together, we'll ensure that your proposals align with client objectives, risk profiles, and the latest macroeconomic trends. 
    """ 

    file_message = await cl.AskFileMessage(
        content=welcome_message,
        accept=["application/pdf"],
        max_size_mb=10
    ).send()
    
    if file_message:
        file = file_message

        # Send a loading message
        loading_message = await cl.Message(content="Loading, please wait...").send()    
        client_data = ClientData()
        client_data.process_statement(file[0].path)
        print('================================================================================================================')
        result = await client_data.initialize_portfolio()

        if result['status'] == "error":
            errors = result['errors']
            msg = cl.Message(content=f"I have encountered the following errors while processing the statement: {errors}.")
        else:
            cl.user_session.set("client_data", client_data)
            msg = cl.Message(content=f"I have proceed the statement. Please proceed with your requests.")

        await msg.send()
        await loading_message.remove()  # Remove the loading message after upload is complete        

        return {"success": True, "path": file[0].path}
    else:
        await cl.Message(content="No file was uploaded. You can still ask questions or discuss investment strategies.").send()
        return {"success": False}
    


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

async def process_function_call(function_call_data, message_history, msg):
    try:
        function_name = function_call_data["name"]
        arguments = json.loads(function_call_data["arguments"])
        
        if function_name not in function_map:
            await msg.stream_token(f"Unknown function: {function_name}\n")
            return
        
        function_call_count = cl.user_session.get("function_call_count", 0)
        if function_call_count >= 7:
            await msg.stream_token("Maximum number of function calls reached. Please rephrase your request.\n")
            await msg.send()
            return

        logging.info(f"Executing function: {function_name} with arguments: {arguments}")
        
        client_data = cl.user_session.get("client_data")
        func = function_map[function_name]['function']
        custom_llm_prompt = function_map[function_name]['custom_llm_prompt']

        result = await func(**arguments) if asyncio.iscoroutinefunction(func) else func(**arguments)

        cl.user_session.set("function_call_count", function_call_count + 1)
        
        # Create a temporary message history for the follow-up request
        # temp_message_history = message_history.copy()
        message_history.append({"role": "function", "name": function_name, "content": custom_llm_prompt + "\n\n" + json.dumps(result)})
        
        # print(f"\n\n\ntemp_message_history: {temp_message_history}")

        follow_up_stream = await client.chat.completions.create(messages= message_history, **settings)

        await process_stream(follow_up_stream, message_history, msg)
        await msg.send()

    except Exception as e:
        error_message = f"Error in {function_name}: {str(e)}"
        logging.error(error_message)
        await msg.stream_token(f"{error_message}\n")
        await msg.send()

def manage_chat_history(message_history):      
    total_length = sum(len(json.dumps(msg)) for msg in message_history)
    while total_length > 100000:
        if len(message_history) > 2:
            removed_msg = message_history.pop(2)  # Remove the third message (index 2)
            total_length -= len(json.dumps(removed_msg))
            logging.info(f"Reducing conversation history. Current length: {total_length} characters")
        else:
            break  # If we only have two or fewer messages, stop removing
    
    return message_history


@cl.on_message
async def main(message: cl.Message):
    cl.user_session.set("function_call_count", 0)

    logging.info(f"\n\n===================== THIS IS A NEW MESSAGE REQUEST ======================")  
    message_history = cl.user_session.get("message_history")

    # create meta_prompt
    # meta_prompt = await create_meta_prompt(message.content)
    meta_prompt = message.content
    # append meta_prompt to message_history
    message_history.append({"role": "user", "content": meta_prompt})
    print(f"MESSAGE HISTORY: {message_history}")

    completion = await client.chat.completions.create(
        messages=message_history,
        **settings
    )
    temp_history = message_history.copy()
    
    msg = cl.Message(content="")
    await msg.send()
    await process_stream(completion, temp_history, msg)
    
    # Only append the assistant's response if it's not empty
    if msg.content.strip():
        message_history.append({"role": "assistant", "content": msg.content})
    
    message_history = manage_chat_history(message_history)
    cl.user_session.set("message_history", message_history)


    await msg.update()

    final_count = cl.user_session.get("function_call_count")
    message_history_length = len(cl.user_session.get("message_history"))
    total_length = sum(len(json.dumps(msg)) for msg in message_history)
    print(f"       ============ MESSAGE REQUEST FINISHED with {final_count} function calls and {message_history_length} messages in history with total length of {total_length} characters =============")

