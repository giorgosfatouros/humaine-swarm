from agents.definition import functions


settings = {
    "model": "gpt-4o-mini",
    "temperature": 0,
    "max_tokens": 2000,
    "top_p": 1,
    "frequency_penalty": 0,
    "presence_penalty": 0.6,
    # "parallel_tool_calls": True,
    # "tools": functions,
    # "tool_choice": "auto",    
    "functions": functions,
    "function_call": "auto",
    "stream": True
}

MAX_CONTEXT_LENGTH = 32000  # For models with an 32K context window
MAX_INPUT_TOKENS = MAX_CONTEXT_LENGTH - settings["max_tokens"]