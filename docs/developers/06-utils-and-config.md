# Utils and config

## config.py

[utils/config.py](../../utils/config.py) imports **`functions`** from [agents/definition.py](../../agents/definition.py) and builds env-derived settings.

### Environment-derived constants

| Name | Env var | Default / note |
|------|---------|----------------|
| KUBEFLOW_HOST | KUBEFLOW_HOST | `http://huanew-kubeflow.ddns.net/pipeline` |
| MINIO_ENDPOINT | MINIO_ENDPOINT | `s3-minio.humaine-horizon.eu` |
| MINIO_SECURE | MINIO_SECURE | `"true"` → True |
| MINIO_API_ENDPOINT | MINIO_API_ENDPOINT | None |
| HAIC_BASE_URL | HAIC_BASE_URL | `https://benchmark.humaine-horizon.eu/api` |
| HAIC_API_KEY | HAIC_API_KEY | Optional; unused while HAIC API is public |
| PINECONE_API_KEY | PINECONE_API_KEY | Required (no default) |
| PINECONE_INDEX | — | `"humaine"` (hardcoded) |
| LLM_MODEL | — | `"gpt-5.6-luna"` |
| LLM_REASONING_EFFORT | LLM_REASONING_EFFORT | `"medium"` |
| EMBEDDING_MODEL | — | `"text-embedding-3-small"` |

### settings dict

Passed as **kwargs** to **OpenAI `responses.create()`**:

- **model**
- **tools**: `functions` (from agents/definition.py)
- **tool_choice**: `"auto"`
- **parallel_tool_calls**: True
- **stream**: True
- **reasoning**: `{"effort": LLM_REASONING_EFFORT}`
- **max_output_tokens**: OUTPUT_TOKEN_RESERVE

### responses_adapter.py

[utils/responses_adapter.py](../../utils/responses_adapter.py) converts session history to Responses API input items:

- **extract_instructions(history)**: system prompt for top-level `instructions`
- **build_responses_input(history)**: user/assistant messages plus `function_call` / `function_call_output` items
- **convert_tools_for_responses(functions)**: flattens Chat Completions tool schemas for Responses API
- **item_token_estimate(item, encoding)**: token counting for truncation across item shapes

### Context length

- **MAX_CONTEXT_LENGTH**: 60000  
- **OUTPUT_TOKEN_RESERVE**: 8000 (headroom for response and reasoning tokens)  
- **MAX_INPUT_TOKENS**: MAX_CONTEXT_LENGTH - OUTPUT_TOKEN_RESERVE  
Used in [classes/user_handler.py](../../classes/user_handler.py) for message-history truncation.

---

## helper_functions.py

[utils/helper_functions.py](../../utils/helper_functions.py) provides shared utilities.

### Logging

- **setup_logging(logger_name, level)**: Creates logger with console handler; level can be overridden by env **LOG_LEVEL** (DEBUG, INFO, WARNING, ERROR, CRITICAL).

### JWT / token (optional use)

- **decode_jwt(token)**: Decodes with CHAINLIT_AUTH_SECRET; used for custom auth flows if needed.
- **extract_token_from_headers(headers)**: Tries Authorization Bearer then referer query param.
- **extract_user_from_payload(payload)**: Builds Chainlit User from payload (e.g. userId, email).

### Tool helpers

- **get_required_arguments(function_name)**: Returns required parameter names for a tool in `functions`.
- **rag_extract_deliverables(retrieved_nodes)**: Maps LlamaIndex nodes to list of `{text: node.get_content()}` for RAG responses.
- **manage_chat_history(message_history)**: Character-based truncation (sliding window 100k); **not used** by the app (app uses UserSessionManager token-based truncation).

### MinIO

- **fetch_minio_credentials_from_keycloak(access_token)**: STS AssumeRoleWithWebIdentity; returns access_key, secret_key, session_token, expiry. See [Session and auth](04-session-and-auth.md).
- **get_minio_client(user_credentials=None)**: Returns **Minio** client; endpoint from MINIO_API_ENDPOINT or MINIO_ENDPOINT/MINIO_SECURE; credentials from argument or env (MINIO_ACCESS_KEY, MINIO_SECRET_KEY, MINIO_SESSION_TOKEN).

### Kubeflow

- **extract_user_namespace_from_token(access_token)**: Decodes JWT (no verify), extracts namespace from claims/groups/roles starting with `kubeflow-`. Does **not** invent a namespace from email. Returns `None` when no claim is present (unscoped / all-namespaces listing).
- **normalize_kubeflow_host(host)**: Ensures `KUBEFLOW_HOST` ends with `/pipeline` (the Pipelines API, not the dashboard).
- **get_kubeflow_client(user_namespace, user_token)**: Bearer-only. Requires **user_token** and builds **kfp.Client(existing_token=...)** with normal TLS certificate verification. Raises **ValueError** if the token is missing.

### Artifact helpers (KFP DSL)

Used when authoring Kubeflow pipeline components (e.g. in [kubeflow/](../../kubeflow/)), not by the chat app:

- **create_dataset_artifact**, **create_model_artifact**, **create_metrics_artifact**, **create_classification_metrics_artifact**
- **read_artifact_metadata**, **log_artifact_properties**
- **read_dataset_from_artifact**, **save_model_to_artifact**, **save_metrics_to_artifact**

Reference only; no effect on the assistant runtime.
