import json
import asyncio
import chainlit as cl
from tools.functions_implementation import get_current_date, use_rag

class FunctionHandler:
    def __init__(self, functions):
        self.function_map = {
            "get_current_date": get_current_date,
            "use_rag": use_rag
        }
        self.functions = functions

    async def process_function_call(self, function_call_data, message_history, msg):
        try:
            function_name = function_call_data["name"]
            arguments = json.loads(function_call_data["arguments"])
            
            print(f"Function called: {function_name}")
            print(f"Arguments: {arguments}")
            
            if function_name not in self.function_map:
                print(f"Unknown function: {function_name}")
                await msg.stream_token(f"Unknown function: {function_name}\n")
                return

            function_call_count = cl.user_session.get("function_call_count", 0)
            print(f"Current function call count: {function_call_count}")
            if function_call_count >= 10:
                print("Maximum number of function calls reached")
                await msg.stream_token("Maximum number of function calls reached. Please rephrase your request.\n")
                await msg.send()
                return

            print(f"Executing function: {function_name}")
            
            # Check if the function is a coroutine (async function)
            if asyncio.iscoroutinefunction(self.function_map[function_name]):
                result = await self.function_map[function_name](**arguments)
            else:
                result = self.function_map[function_name](**arguments)
            print(f"Function result: \n ----------\n{str(result)[:500]}\n----------\n")

            cl.user_session.set("function_call_count", function_call_count + 1)
            print(f"Updated function call count: {function_call_count + 1}")

            message_history.append({
                "role": "function", 
                "name": function_name, 
                "content": json.dumps(result)
            })
            print("Added function result to message history")
            
            print("Making follow-up call to LLM")
            follow_up_stream = await self.client.chat.completions.create(
                messages=message_history,
                stream=True,
                functions=self.functions,
                function_call="auto",
                **self.settings
            )
            
            print("Processing LLM's response")
            await self.process_stream(follow_up_stream, message_history, msg)

            await msg.send()
            print("Final message sent")

        except json.JSONDecodeError as e:
            print(f"JSON decode error: {str(e)}")
            await msg.stream_token(f"Error: Failed to parse arguments. {str(e)}\n")
            await msg.send()
        except TypeError as e:
            print(f"Type error in {function_name}: {str(e)}")
            await msg.stream_token(f"Error in {function_name}: {str(e)}. Arguments received: {arguments}\n")
            await msg.send()
        except Exception as e:
            print(f"Unexpected error in {function_name}: {str(e)}")
            await msg.stream_token(f"Unexpected error in {function_name}: {str(e)}\n")
            await msg.send()