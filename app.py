from utils.dotenv_local import load_swarm_dotenv

load_swarm_dotenv()

import asyncio
from openai import AsyncOpenAI
import chainlit as cl
from chainlit import User
from chainlit.input_widget import MultiSelect
from starlette.datastructures import Headers

from classes.user_handler import UserSessionManager
from starters import set_starters
import json
import logging
from agents.code import function_map, query_haic_benchmark, _get_user_pilot_context
from agents.definition import functions
from agents.tool_packs import (
    ENABLED_PACKS_SESSION_KEY,
    build_system_prompt,
    filter_functions,
    get_default_enabled_pack_ids,
    is_haic_pack_enabled,
    is_tool_enabled,
    multiselect_initial,
    multiselect_items,
    resolve_enabled_pack_ids,
    update_message_history_system_prompt,
)
from utils.haic_routing import detect_haic_live_query
from utils.haic_client import format_haic_result_markdown
from utils.helper_functions import setup_logging
from utils.config import settings
from utils.responses_adapter import (
    build_responses_input,
    convert_tools_for_responses,
    extract_instructions,
    make_function_call,
    make_function_call_output,
)
from utils.keycloak_header_auth import (
    claims_to_display_name,
    claims_to_identifier,
    validate_bearer_token,
)
from typing import Optional, Dict
from chainlit.types import ThreadDict
import plotly.graph_objects as go
import plotly.io as pio

# Setup logging
logger = setup_logging('CHAT', level=logging.ERROR)
logging.getLogger("httpx").setLevel("WARNING")

client = AsyncOpenAI()


def get_enabled_pack_ids() -> list:
    stored = cl.user_session.get(ENABLED_PACKS_SESSION_KEY)
    pilot_context = _get_user_pilot_context()
    return resolve_enabled_pack_ids(stored, pilot_context)


def set_enabled_pack_ids(pack_ids: list) -> None:
    cl.user_session.set(ENABLED_PACKS_SESSION_KEY, list(pack_ids))


async def send_tool_pack_settings(pilot_context: dict, enabled_packs: list) -> list:
    """Show ChatSettings and return the resolved enabled pack ids."""
    items = multiselect_items(pilot_context)
    if not items:
        return enabled_packs

    ui_settings = await cl.ChatSettings(
        [
            MultiSelect(
                id="enabled_packs",
                label="Enabled capabilities",
                items=items,
                initial=multiselect_initial(enabled_packs, pilot_context),
                description="Choose which tool groups the assistant may use in this chat.",
            )
        ]
    ).send()
    selected = ui_settings.get("enabled_packs")
    return resolve_enabled_pack_ids(selected, pilot_context)

# Action callback to handle follow-up questions
@cl.action_callback("follow_up_question")
async def on_follow_up_question(action):
    await main(cl.Message(content=action.value))
    await action.remove()


@cl.header_auth_callback
async def header_auth_callback(headers: Headers) -> Optional[User]:
    """
    Authenticate embedded Common FE: Bearer Keycloak access token -> Chainlit session cookie.
    Must mirror oauth_callback metadata so UserSessionManager receives oauth_token.
    """
    auth = headers.get("authorization") or ""
    if not auth.lower().startswith("bearer "):
        logger.error("Header auth: missing or non-Bearer Authorization header")
        return None
    token = auth[7:].strip()
    if not token:
        logger.error("Header auth: empty Bearer token")
        return None

    claims = validate_bearer_token(token)
    if not claims:
        logger.error(
            "Header auth: Keycloak token rejected (see Keycloak/JWKS connectivity and token issuer)"
        )
        return None

    identifier = claims_to_identifier(claims)
    display_name = claims_to_display_name(claims)
    raw_flat = {str(k): "" if v is None else str(v) for k, v in claims.items()}

    user = User(
        identifier=identifier,
        display_name=display_name,
        metadata={
            "oauth_token": token,
            "provider": "keycloak",
            "raw_user_data": raw_flat,
        },
    )
    return user


@cl.oauth_callback
def oauth_callback(
    provider_id: str,
    token: str,
    raw_user_data: Dict[str, str],
    default_user: cl.User,
) -> Optional[cl.User]:
    """
    Handle OAuth authentication callback from Keycloak.
    Stores the OAuth token in user metadata for later use with MinIO and Kubeflow.
    """
    if provider_id == "keycloak":
        # Store the OAuth token in user metadata for session use
        if default_user.metadata is None:
            default_user.metadata = {}
        
        default_user.metadata["oauth_token"] = token
        default_user.metadata["provider"] = "keycloak"
        default_user.metadata["raw_user_data"] = raw_user_data
        
        logger.info(f"Successfully authenticated user via Keycloak: {default_user.identifier}")
        return default_user
    
    logger.warning(f"Unknown OAuth provider: {provider_id}")
    return None
  

# Chat initialization
@cl.on_chat_start
async def start_chat():
    user = cl.user_session.get("user")
    UserSessionManager.set_user_id(user.identifier)

    # Store OAuth token if available
    if user.metadata and "oauth_token" in user.metadata:
        UserSessionManager.set_oauth_token(user.metadata["oauth_token"])
        logger.info(f"Stored OAuth token for user {user.identifier}")

        try:
            await UserSessionManager.fetch_and_store_minio_credentials()
            logger.info(f"Successfully fetched MinIO credentials for user {user.identifier}")
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Failed to fetch MinIO credentials: {error_msg}")

            if "expired" in error_msg.lower() or "token is expired" in error_msg.lower():
                logger.warning("OAuth token expired on chat start - clearing session to force logout")
                UserSessionManager.clear_session()
                await cl.Message(
                    content="Your session has expired. Please refresh the page to log in again.",
                    author="System"
                ).send()
                return

        try:
            UserSessionManager.extract_and_store_namespace()
            logger.info(f"Extracted Kubeflow namespace for user {user.identifier}")
            UserSessionManager.extract_and_store_token_info()
            logger.info(f"Extracted comprehensive token info for user {user.identifier}")
        except Exception as e:
            logger.error(f"Failed to extract token information: {str(e)}")

    pilot_context = _get_user_pilot_context()
    enabled_packs = get_default_enabled_pack_ids(pilot_context)
    enabled_packs = await send_tool_pack_settings(pilot_context, enabled_packs)
    set_enabled_pack_ids(enabled_packs)

    init_message = [
        {"role": "system", "content": build_system_prompt(enabled_packs)}
    ]
    UserSessionManager.set_message_history(init_message)

    logger.info(UserSessionManager.print_stored_info())


@cl.on_settings_update
async def on_settings_update(settings: dict):
    pilot_context = _get_user_pilot_context()
    selected = settings.get("enabled_packs")
    enabled_packs = resolve_enabled_pack_ids(selected, pilot_context)
    set_enabled_pack_ids(enabled_packs)

    message_history = UserSessionManager.get_message_history()
    update_message_history_system_prompt(message_history, enabled_packs)
    UserSessionManager.set_message_history(message_history)
    logger.info(f"Tool packs updated: {enabled_packs}")


@cl.on_chat_resume
async def on_chat_resume(thread: ThreadDict):
    thread_id = thread['id']
    logger.info(f"Thread ID on resume: {thread_id}")

async def create_response_stream(message_history):
    request_kwargs = dict(settings)
    enabled_packs = get_enabled_pack_ids()
    request_kwargs["tools"] = convert_tools_for_responses(
        filter_functions(functions, enabled_packs)
    )
    instructions = extract_instructions(message_history)
    if instructions:
        request_kwargs["instructions"] = instructions
    return await client.responses.create(
        input=build_responses_input(message_history),
        **request_kwargs,
    )


async def finalize_assistant_message(msg: cl.Message) -> None:
    has_content = bool(msg.content and msg.content.strip())
    has_elements = bool(getattr(msg, "elements", None))
    if not has_content and not has_elements:
        return
    if getattr(msg, "streaming", False):
        await msg.update()
    else:
        await msg.send()


# Process assistant's response stream and handle tool calls (Responses API)
async def process_responses_stream(stream, message_history, msg):
    tool_calls_by_item_id = {}

    async for event in stream:
        event_type = getattr(event, "type", None)

        if event_type == "response.output_text.delta":
            # stream_token already appends the delta to msg.content
            await msg.stream_token(event.delta)
        elif event_type == "response.output_item.added":
            item = event.item
            if getattr(item, "type", None) == "function_call":
                tool_calls_by_item_id[item.id] = {
                    "call_id": item.call_id,
                    "name": item.name,
                    "arguments": item.arguments or "",
                    "item_id": item.id,
                }
        elif event_type == "response.function_call_arguments.delta":
            tool_call = tool_calls_by_item_id.get(event.item_id)
            if tool_call:
                tool_call["arguments"] += event.delta
        elif event_type == "response.function_call_arguments.done":
            tool_call = tool_calls_by_item_id.get(event.item_id)
            if tool_call:
                tool_call["arguments"] = event.arguments

    valid_tool_calls = []
    for tool_call in tool_calls_by_item_id.values():
        try:
            args = tool_call["arguments"]
            if args.strip() and args[-1] == "}":
                json.loads(args)
                valid_tool_calls.append(tool_call)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in tool call: {e}")

    if not valid_tool_calls:
        return

    logger.info(f"Processing {len(valid_tool_calls)} function calls concurrently")

    async def call_function(tool_call):
        try:
            function_name = tool_call["name"]
            arguments = json.loads(tool_call["arguments"])

            if function_name not in function_map:
                logger.warning(f"Unknown function: {function_name}")
                return None

            enabled_packs = get_enabled_pack_ids()
            if not is_tool_enabled(function_name, enabled_packs):
                logger.warning(f"Disabled tool blocked: {function_name}")
                disabled_result = {
                    "error": (
                        f"Tool '{function_name}' is not enabled for this session. "
                        "Enable the relevant capability in chat settings."
                    )
                }
                return (
                    make_function_call(
                        tool_call["call_id"],
                        function_name,
                        tool_call["arguments"],
                    ),
                    make_function_call_output(tool_call["call_id"], disabled_result),
                )

            logger.info(f"Executing function: {function_name} with arguments: {arguments}")

            func = function_map[function_name]
            result = await func(**arguments) if asyncio.iscoroutinefunction(func) else func(**arguments)

            UserSessionManager.increment_function_call_count()

            try:
                if not hasattr(msg, "elements") or msg.elements is None:
                    msg.elements = []

                if function_name == "plot_data" and isinstance(result, dict):
                    if result.get("success") and "figure_json" in result:
                        try:
                            figure_json = result["figure_json"]
                            fig = pio.from_json(figure_json)

                            display = result.get("display", "inline")
                            size = result.get("size", "medium")
                            chart_type = result.get("chart_type", "chart")
                            title = result.get("title", "Data Visualization")

                            plotly_element = cl.Plotly(
                                name=f"{chart_type}_{title}",
                                figure=fig,
                                display=display,
                                size=size,
                            )
                            msg.elements.append(plotly_element)
                            logger.info(f"Added Plotly chart: {chart_type} - {title}")
                        except Exception as plot_error:
                            logger.error(f"Error creating Plotly element: {str(plot_error)}")
            except Exception as e:
                logger.error(f"Error creating figure for {function_name}: {str(e)}")

            return (
                make_function_call(
                    tool_call["call_id"],
                    function_name,
                    tool_call["arguments"],
                ),
                make_function_call_output(tool_call["call_id"], result),
            )
        except Exception as e:
            logger.error(f"Error in {tool_call['name']}: {str(e)}")
            return None

    function_results = await asyncio.gather(
        *(call_function(tool_call) for tool_call in valid_tool_calls)
    )

    history_items = []
    for result in function_results:
        if result is None:
            continue
        function_call_item, function_output_item = result
        history_items.extend([function_call_item, function_output_item])

    if not history_items:
        return

    message_history.extend(history_items)
    UserSessionManager.set_message_history(message_history)

    elements = msg.elements
    if elements and len(elements) > 0:
        follow_up_msg = cl.Message(content="")
        follow_up_msg.elements = elements.copy()
        await follow_up_msg.send()
    else:
        follow_up_msg = msg

    follow_up_stream = await create_response_stream(message_history)
    await process_responses_stream(follow_up_stream, message_history, follow_up_msg)

    if follow_up_msg is not msg:
        await finalize_assistant_message(follow_up_msg)
        msg.elements = []


# Main function that handles user messages
@cl.on_message
async def main(message: cl.Message):
    # Increment the function call count for the current session
    UserSessionManager.increment_function_call_count()
    
    thread_id = UserSessionManager.get_thread_id()

    # Create message but don't send it yet - this is key to avoiding empty messages
    msg = cl.Message(author="Swarm Agent", content="")
    
    logger.info(f"New message from user {UserSessionManager.get_user_id()}")

    # Get the message history and append the new user message
    message_history = UserSessionManager.get_message_history()
    message_history.append({"role": "user", "content": message.content})

    haic_action = detect_haic_live_query(message.content)
    if haic_action and is_haic_pack_enabled(get_enabled_pack_ids()):
        logger.info(f"HAIC pre-router: action={haic_action} for user message")
        result = await query_haic_benchmark(action=haic_action)
        reply = format_haic_result_markdown(result)
        msg.content = reply
        message_history.append({"role": "assistant", "content": reply})
        await msg.send()
        UserSessionManager.set_message_history(message_history)
        return

    stream = await create_response_stream(message_history)

    await process_responses_stream(stream, message_history, msg)
    if msg.content.strip():
        message_history.append({"role": "assistant", "content": msg.content})
        
    await finalize_assistant_message(msg)

    # Save the updated message history in the session
    UserSessionManager.set_message_history(message_history)

    # Log message details for debugging
    total_length = sum(len(json.dumps(msg)) for msg in message_history)
    logger.info(f"Message finished with {UserSessionManager.get_function_call_count()} function calls and {len(message_history)} messages in history with total length of {total_length} characters")

    # # Generate follow-up questions for the user
    # follow_up_questions = await generate_follow_up_questions(message_history)
    
    # # Send the follow-up questions to the UI
    # await send_follow_up_questions(follow_up_questions)