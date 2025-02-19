import json
import asyncio
import chainlit as cl

class StreamProcessor:
    def __init__(self, function_handler, client, settings):
        self.function_handler = function_handler
        self.client = client
        self.settings = settings

    async def process_stream(self, stream, message_history, msg):
        function_call_data = {"name": None, "arguments": ""}
        
        async for part in stream:
            delta = part.choices[0].delta
            
            if function_call := delta.function_call:
                if function_call.name:
                    function_call_data["name"] = function_call.name
                if function_call.arguments:
                    function_call_data["arguments"] += function_call.arguments
                    
                    # Attempt to parse the JSON after each update
                    try:
                        json.loads(function_call_data["arguments"])
                        # If successful, process the function call
                        await self.function_handler.process_function_call(function_call_data, message_history, msg)
                        function_call_data = {"name": None, "arguments": ""}
                    except json.JSONDecodeError:
                        # If it's not valid JSON yet, continue accumulating
                        pass
            elif content := delta.content:
                await msg.stream_token(content)

        # Handle any remaining function call data
        if function_call_data["name"] and function_call_data["arguments"]:
            try:
                json.loads(function_call_data["arguments"])
                await self.function_handler.process_function_call(function_call_data, message_history, msg)
            except json.JSONDecodeError:
                print(f"Incomplete function call data: {function_call_data}")

        # Ensure the message is sent
        await msg.send()